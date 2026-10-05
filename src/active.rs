// Copyright (c) 2026 Oleksii PELYKH
// SPDX-License-Identifier: MIT

//! Active-account resolution — WHICH roster account the machine is currently
//! logged into, resolved TOKEN-FIRST.
//!
//! The canonical keychain credential is the authoritative bearer of the active
//! session; `~/.claude.json`'s `oauthAccount` is only the clobberable,
//! last-writer-wins DISPLAY half — Claude Code rewrites or clears it out-of-band on
//! a forced logout (see [`crate::claude_state`]). So resolution consults the two
//! signals in that order of trust:
//!   1. the canonical token byte-matches an account's stashed credential → that
//!      account (exact right after a swap or a re-stash), then
//!   2. the canonical's non-empty refresh token byte-matches an account's stashed
//!      one → that account (Claude Code rewrote the item without rotating the grant),
//!      then
//!   3. the displayed `accountUuid` maps to a roster account — the only signal when
//!      the grant itself is new and no stash holds it yet (a fresh `/login` or a
//!      rotating refresh).
//!
//! Both the daemon's poll loop ([`crate::daemon`]) and the manual `use` swap
//! ([`crate::use_account`]) resolve the outgoing account through these SAME two
//! functions, so the recovery verb `use` no longer hard-fails on a cleared display
//! while a healthy token still sits in the keychain (issue #207). The resolvers
//! return a roster INDEX only — never a token, email, or `accountUuid` — so the
//! output-redaction invariant (issue #15) holds by construction.

use std::path::Path;

use crate::claude_state;
use crate::config::Account;
use crate::keychain::Credential;
use crate::stash::AccountStash;

/// Identify which roster account the given `canonical` credential belongs to, using
/// three signals in order: (1) the canonical blob byte-matches an account's stash —
/// exact right after a swap or a re-stash; (2) the canonical's non-empty REFRESH
/// token byte-matches an account's stashed one — the same OAuth grant, rewritten in
/// place by Claude Code without rotating it; (3) the displayed `~/.claude.json`
/// `accountUuid` maps to a roster account — the signal when the grant itself is new
/// (a fresh `/login`, or a rotation) and no stash holds it yet. `None` if none resolves.
///
/// Token-first because the keychain credential is the authoritative bearer and
/// `~/.claude.json` is the clobberable display half (issue #207): a cleared or stale
/// display still resolves as long as the canonical token matches a stash. Step (2)
/// exists because Claude Code rewrites the item without touching the refresh token,
/// and every such rewrite used to fall straight to the display: a display naming a
/// different account then made the re-auth re-stash write one account's grant into
/// another's stash, destroying the second account's own grant. A refresh token is
/// one grant's bearer, so it attributes the blob more reliably than the display can.
/// An EMPTY refresh token is the cleared-in-place dead signal and matches nothing.
/// Shared by the daemon's active-resolution and re-auth re-stash paths and the manual
/// `use` swap's outgoing-account resolution.
pub(crate) async fn resolve_account_for<S: AccountStash>(
    roster: &[Account],
    stash: &S,
    claude_json: &Path,
    canonical: &Credential,
) -> Option<usize> {
    let canonical_rt = live_refresh_token(canonical);
    let mut same_grant = None;
    for (i, account) in roster.iter().enumerate() {
        let Ok(stashed) = stash.read(&account.stash()).await else {
            continue;
        };
        if stashed.credential.matches(canonical) {
            return Some(i);
        }
        if same_grant.is_none() {
            if let (Some(want), Some(have)) = (
                canonical_rt.as_ref(),
                live_refresh_token(&stashed.credential),
            ) {
                if want.as_slice() == have.as_slice() {
                    same_grant = Some(i);
                }
            }
        }
    }
    same_grant.or_else(|| resolve_via_display(roster, claude_json))
}

/// The credential's refresh token when it is present and non-empty — `None` for an
/// unparseable blob and for the cleared-in-place dead signal alike, so two cleared
/// credentials never attribute to each other. Only ever byte-compared, never logged.
fn live_refresh_token(credential: &Credential) -> Option<zeroize::Zeroizing<Vec<u8>>> {
    crate::refresh::refresh_token(credential.expose()).filter(|rt| !rt.is_empty())
}

/// The `~/.claude.json` DISPLAY-only fallback: the roster account whose
/// `account_uuid` matches the displayed `oauthAccount.accountUuid`, or `None` when
/// the file is unreadable / absent or names an account not in the roster.
///
/// The weaker of the two signals ([`resolve_account_for`]'s step 2), factored out
/// because it is ALSO the only signal available when the canonical token cannot be
/// read at all — the daemon's locked-keychain poll degrades to display-only rather
/// than swap blindly (`use`, by contrast, treats a locked keychain as a safety
/// abort, so it never reaches this fallback under a lock).
pub(crate) fn resolve_via_display(roster: &[Account], claude_json: &Path) -> Option<usize> {
    let oauth = claude_state::read_oauth_account_from(claude_json).ok()?;
    roster
        .iter()
        .position(|account| account.account_uuid == oauth.account_uuid())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::claude_state::OauthAccount;
    use crate::stash::{FakeAccountStash, StashedAccount};

    fn acct(label: &str, uuid: &str) -> Account {
        Account {
            account_uuid: uuid.to_owned(),
            label: label.to_owned(),
            enabled: true,
        }
    }

    fn cred(blob: &[u8]) -> Credential {
        Credential::new(blob.to_vec())
    }

    fn stashed(token: &[u8], uuid: &str) -> StashedAccount {
        StashedAccount {
            credential: cred(token),
            oauth_account: OauthAccount::from_object_bytes(
                format!(r#"{{"accountUuid":"{uuid}","emailAddress":"{uuid}@example.com"}}"#)
                    .as_bytes(),
            )
            .unwrap(),
        }
    }

    /// A two-account roster: `work` (`u-A`) and `spare` (`u-B`).
    fn roster_ab() -> Vec<Account> {
        vec![acct("work", "u-A"), acct("spare", "u-B")]
    }

    /// A stash holding both accounts' tokens (`A-token` / `B-token`).
    async fn stash_ab() -> FakeAccountStash {
        let stash = FakeAccountStash::empty();
        stash
            .write("Sessiometer/u-A", &stashed(b"A-token", "u-A"))
            .await
            .unwrap();
        stash
            .write("Sessiometer/u-B", &stashed(b"B-token", "u-B"))
            .await
            .unwrap();
        stash
    }

    /// A `~/.claude.json` displaying `active_uuid`, returned with its tempdir guard.
    fn claude_json_for(active_uuid: &str) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(".claude.json");
        std::fs::write(
            &path,
            format!(
                r#"{{"numStartups":1,"oauthAccount":{{"accountUuid":"{active_uuid}","emailAddress":"{active_uuid}@x.com"}}}}"#
            ),
        )
        .unwrap();
        (dir, path)
    }

    #[tokio::test]
    async fn token_match_resolves_even_when_the_display_is_stale() {
        // The #207 core: ~/.claude.json points at an account NOT in the roster
        // (Claude Code cleared/rewrote the display), yet the canonical token
        // byte-matches u-A's stash → resolve token-first to u-A regardless.
        let roster = roster_ab();
        let stash = stash_ab().await;
        let (_dir, json) = claude_json_for("u-UNKNOWN");
        let resolved = resolve_account_for(&roster, &stash, &json, &cred(b"A-token")).await;
        assert_eq!(
            resolved,
            Some(0),
            "token byte-match wins over the stale display"
        );
    }

    #[tokio::test]
    async fn display_fallback_when_the_token_matches_no_stash() {
        // A fresh /login (or in-place refresh) minted a token no stash holds yet;
        // the display still names a roster account → resolve via the display half.
        let roster = roster_ab();
        let stash = stash_ab().await;
        let (_dir, json) = claude_json_for("u-B");
        let resolved = resolve_account_for(&roster, &stash, &json, &cred(b"ORPHAN-token")).await;
        assert_eq!(
            resolved,
            Some(1),
            "no stash matches → display maps u-B to index 1"
        );
    }

    #[tokio::test]
    async fn unresolved_when_token_matches_no_stash_and_display_is_unresolvable() {
        // Fail-closed: neither signal resolves — an orphan token AND a display
        // naming an account not in the roster → None (the caller fails cleanly).
        let roster = roster_ab();
        let stash = stash_ab().await;
        let (_dir, json) = claude_json_for("u-UNKNOWN");
        let resolved = resolve_account_for(&roster, &stash, &json, &cred(b"ORPHAN-token")).await;
        assert_eq!(
            resolved, None,
            "no token match and no display match → unresolved"
        );
    }

    #[test]
    fn resolve_via_display_maps_a_displayed_uuid_to_its_roster_index() {
        let roster = roster_ab();
        let (_dir, json) = claude_json_for("u-B");
        assert_eq!(resolve_via_display(&roster, &json), Some(1));
    }

    #[test]
    fn resolve_via_display_is_none_when_the_file_is_absent() {
        let roster = roster_ab();
        let missing = std::path::Path::new("/nonexistent/.claude.json");
        assert_eq!(resolve_via_display(&roster, missing), None);
    }

    /// A Claude Code-shaped credential blob: `access` / `refresh` tokens plus a `tier` field
    /// standing in for the non-token bytes Claude Code rewrites in place.
    fn oauth_blob(access: &str, refresh: &str, tier: &str) -> Vec<u8> {
        format!(
            r#"{{"claudeAiOauth":{{"accessToken":"{access}","refreshToken":"{refresh}","expiresAt":1,"rateLimitTier":"{tier}"}}}}"#
        )
        .into_bytes()
    }

    /// A stash holding `work` (`u-A`) and `spare` (`u-B`) as the given blobs.
    async fn stash_with_blobs(a: &[u8], b: &[u8]) -> FakeAccountStash {
        let stash = FakeAccountStash::empty();
        stash
            .write("Sessiometer/u-A", &stashed(a, "u-A"))
            .await
            .unwrap();
        stash
            .write("Sessiometer/u-B", &stashed(b, "u-B"))
            .await
            .unwrap();
        stash
    }

    #[tokio::test]
    async fn a_grant_rewritten_in_place_resolves_to_its_owner_over_a_stale_display() {
        // Claude Code rewrote u-A's item without rotating the refresh token, so the blob
        // matches no stash; the display names u-B. Attributing by the display would re-stash
        // u-A's grant over u-B's — the shared refresh token resolves it to u-A instead.
        let roster = roster_ab();
        let stash = stash_with_blobs(
            &oauth_blob("at-A1", "rt-A", "t1"),
            &oauth_blob("at-B1", "rt-B", "t1"),
        )
        .await;
        let (_dir, json) = claude_json_for("u-B");
        let canonical = cred(&oauth_blob("at-A1", "rt-A", "t2"));
        assert_eq!(
            resolve_account_for(&roster, &stash, &json, &canonical).await,
            Some(0),
            "the refresh token names the grant's owner; the stale display does not"
        );
    }

    #[tokio::test]
    async fn a_blob_match_outranks_an_earlier_refresh_token_match() {
        // u-A's stash shares u-B's refresh token (a stale copy) but only u-B's stash is the
        // exact blob — the exact match keeps its first rank.
        let roster = roster_ab();
        let exact = oauth_blob("at-B2", "rt-B", "t2");
        let stash = stash_with_blobs(&oauth_blob("at-B1", "rt-B", "t1"), &exact).await;
        let (_dir, json) = claude_json_for("u-A");
        assert_eq!(
            resolve_account_for(&roster, &stash, &json, &cred(&exact)).await,
            Some(1)
        );
    }

    #[tokio::test]
    async fn a_cleared_refresh_token_never_attributes_by_grant() {
        // Both the canonical and u-A's stash carry the cleared (empty) refresh token. Empty is
        // the dead signal, not a shared grant, so resolution falls through to the display.
        let roster = roster_ab();
        let stash = stash_with_blobs(
            &oauth_blob("", "", "t1"),
            &oauth_blob("at-B1", "rt-B", "t1"),
        )
        .await;
        let (_dir, json) = claude_json_for("u-B");
        let canonical = cred(&oauth_blob("", "", "t2"));
        assert_eq!(
            resolve_account_for(&roster, &stash, &json, &canonical).await,
            Some(1)
        );
    }

    #[tokio::test]
    async fn a_new_grant_still_falls_back_to_the_display() {
        // A fresh /login or a rotation: the refresh token is in no stash, so the display
        // remains the signal, exactly as before.
        let roster = roster_ab();
        let stash = stash_with_blobs(
            &oauth_blob("at-A1", "rt-A", "t1"),
            &oauth_blob("at-B1", "rt-B", "t1"),
        )
        .await;
        let (_dir, json) = claude_json_for("u-B");
        let canonical = cred(&oauth_blob("at-NEW", "rt-NEW", "t1"));
        assert_eq!(
            resolve_account_for(&roster, &stash, &json, &canonical).await,
            Some(1)
        );
    }
}
