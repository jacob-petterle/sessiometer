// Copyright (c) 2026 Oleksii PELYKH
// SPDX-License-Identifier: MIT

//! Per-account usage polling.
//!
//! Reads one account's quota from the Claude usage endpoint, read-only:
//!
//! ```text
//! GET https://api.anthropic.com/api/oauth/usage
//! Authorization: Bearer <accessToken>
//! anthropic-beta: oauth-2025-04-20
//! ```
//!
//! and projects the response into a [`Usage`] reading (both windows) for the swap
//! decision. The bearer is the `accessToken` inside the stored
//! `Claude Code-credentials` blob — read through the existing [`CredentialStore`]
//! seam ([`crate::keychain`]); the poller never re-reads or re-mints a token.
//!
//! ## Transport: the `curl` CLI, not an HTTP crate
//!
//! Like [`crate::keychain`] / [`crate::stash`] drive `/usr/bin/security`, this
//! drives `/usr/bin/curl` (absolute path, so a hijacked `PATH` cannot substitute
//! a binary for this network call). No HTTP crate is pulled into the graph; the
//! request rides the system tool that is always present on macOS. The bearer
//! token is fed on `curl`'s **stdin** (a `--config -` file), so — unlike the
//! keychain CLI's `-w <secret>` argv — it never appears in this process's argv.
//!
//! ## HTTP outcome taxonomy (issue #5)
//!
//! Each poll is classified — these are runtime states, not process exits:
//!   - **success** (`2xx`) → a [`Usage`] reading.
//!   - **Transient** (`5xx`, or no HTTP response: DNS / connection / TLS /
//!     timeout) → [`Error::UsageTransient`]. Back off, skip, never swap on
//!     missing data.
//!   - **RateLimited** (`429`) → [`Error::UsageRateLimited`]; other non-401/403
//!     `4xx` → [`Error::UsageRejected`]. Both: back off, skip (design G4).
//!   - **Unauthorized** (`401`) → [`Error::UsageUnauthorized`]; the consecutive
//!     count feeds the re-stash trigger (below).
//!   - **ScopeFailed** (`403`) → [`Error::UsageScopeMissing`], surfaced
//!     **distinctly** from 401 — the hallmark of a non-interactive setup token.
//!
//! ## 401 streak → dead-credential detection (issue #42)
//!
//! The poll path is stateless: each poll classifies one HTTP outcome and returns a
//! [`Usage`] reading or a typed error. Consecutive-401 streak tracking — and the
//! dead-credential quarantine / emergency-swap / recovery it drives — lives in the
//! daemon's per-account health state ([`crate::daemon`]), which is the only place a
//! streak can persist ACROSS polls (a per-poll counter here would reset every tick
//! and never accumulate). The poller never self-refreshes a token.

use std::process::Stdio;
use std::time::Duration;

#[cfg(test)]
use std::cell::RefCell;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use zeroize::Zeroizing;

use crate::error::{Error, Result};
use crate::keychain::CredentialStore;

/// Absolute path to the system `curl`. Absolute (not `$PATH`-resolved) so a
/// hijacked `PATH` cannot substitute a different binary for this network call —
/// the same discipline [`crate::keychain`] applies to `security`.
const CURL: &str = "/usr/bin/curl";

/// The read-only per-account usage endpoint.
const USAGE_URL: &str = "https://api.anthropic.com/api/oauth/usage";

/// The OAuth beta opt-in header the usage endpoint requires.
const BETA_HEADER: &str = "anthropic-beta: oauth-2025-04-20";

/// Hard ceiling (seconds) on a single poll, so a hung request can never wedge the
/// poll loop. Comfortably below the default 300 s (5 min) poll interval.
const POLL_TIMEOUT_SECS: u32 = 30;

/// A point-in-time usage reading for one account, across both quota windows.
///
/// The swap decision compares each dimension against its OWN threshold (issue
/// #41: session vs `session_ceiling`, weekly vs the separate `weekly_ceiling`),
/// so the reading carries both fractions and projects neither to a single
/// worst-case scalar.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Usage {
    /// Fraction in `[0.0, 1.0]` of the rolling 5-hour session window consumed.
    pub(crate) session: f64,
    /// Fraction in `[0.0, 1.0]` of the weekly window consumed — the BINDING
    /// reading: the all-models `weekly_all`, or a per-model `weekly_scoped`
    /// cap when that sits higher.
    pub(crate) weekly: f64,
    /// Fraction in `[0.0, 1.0]` of the ALL-MODELS weekly window consumed — the
    /// `seven_day` reading BEFORE any `weekly_scoped` override. Kept beside
    /// `weekly` so the sample store can persist the overall-vs-scoped split for
    /// external readers.
    pub(crate) weekly_all: f64,
    /// The highest per-model `weekly_scoped` cap's fraction, when the response
    /// carried one; `None` when no scoped entry exists (so the reading is
    /// all-models only). The cap's model name rides the sample-only
    /// channel instead ([`PolledReading::weekly_scoped_model`]) — an owned
    /// `String` here would cost `Usage` its `Copy`.
    pub(crate) weekly_scoped: Option<f64>,
    /// Epoch seconds at which the WEEKLY window resets, when the API reported a
    /// parseable timestamp; `None` otherwise. The all-exhausted terminal logic
    /// (issue #11) consumes it: when every account is weekly-exhausted the daemon
    /// holds on the account whose weekly window resets soonest. Only the weekly
    /// dimension drives that terminal signal — the weekly window is the hard limit
    /// whose reset actually ends the all-exhausted state.
    pub(crate) weekly_resets_at: Option<i64>,
    /// Epoch seconds at which the rolling 5-hour SESSION window resets, when the
    /// API reported a parseable timestamp; `None` otherwise. Carried so `status`
    /// can show a per-account "resets in" (issue #72): in normal rotation an
    /// account is session-exhausted (out for hours) while its weekly window is
    /// fine, so the SESSION reset — not the weekly one — is when it becomes usable
    /// again. The all-exhausted terminal signal ignores it (that keys off the weekly
    /// reset above), but since issue #614 the swap-decision path DOES read it as the
    /// session window's IDENTITY: usage is monotonic within one window, so a session
    /// value that drops while this stamp is unchanged is a stale / cache-lagged
    /// reading rather than real drain (see [`crate::swap::SessionHighWater`]).
    pub(crate) session_resets_at: Option<i64>,
}

/// A poll's full reading: the lean swap-decision [`Usage`] plus the sample-only
/// fields the decision path does not consume (issue #156). Kept SEPARATE from
/// [`Usage`] so the pervasive, `Copy` decision type stays lean — the daemon
/// projects a `PolledReading` straight to its `Usage` on the decision path, and the
/// usage-sample collector reads the extras here.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PolledReading {
    /// The swap-decision reading (both quota dimensions + resets).
    pub(crate) usage: Usage,
    /// The provider-reported severity label for this reading (e.g. `"critical"`),
    /// when the API supplied one — the WEEKLY window's label, falling back to the
    /// SESSION window's. Persisted to the usage-sample store
    /// ([`crate::usage_store::Sample`]); a tolerant free string, never parsed to an
    /// enum, so an unrecognised future value is retained rather than dropped.
    pub(crate) severity: Option<String>,
    /// The winning `weekly_scoped` entry's `scope.model.display_name` (e.g.
    /// `"Fable"`), when the response named one — persisted to the sample store
    /// alongside [`Usage::weekly_scoped`]. Kept here rather than on [`Usage`] so
    /// the pervasive `Copy` decision type carries no owned `String`.
    pub(crate) weekly_scoped_model: Option<String>,
    /// Epoch seconds at which the winning `weekly_scoped` entry's window resets,
    /// when it carried a parseable timestamp. Kept BESIDE
    /// [`Usage::weekly_resets_at`] (which is the scoped reset only while the
    /// scoped cap overrides) so a consumer can chart the per-model reset even
    /// when the all-models window is the binding one.
    pub(crate) weekly_scoped_resets_at: Option<i64>,
    /// Epoch seconds at which the ALL-MODELS weekly window resets — the
    /// `seven_day` `resets_at` BEFORE the scoped override [`Usage::weekly`]
    /// describes. Kept here (not on [`Usage`]) so
    /// [`PolledReading::with_weekly_scoped_binding`] can restore the all-models
    /// reset when the scoped override is switched off.
    pub(crate) weekly_all_resets_at: Option<i64>,
    /// The account's paid-extra-usage posture, when the response reported one —
    /// top-level `extra_usage`, falling back to the `spend` object when the API
    /// omits the former. Persisted to the sample store and `status --json`; the
    /// swap-decision path does not consume it. Non-secret throughout (amounts
    /// and flags only — never identifiers).
    pub(crate) extra_usage: Option<ExtraUsage>,
}

impl PolledReading {
    /// Apply `[tunables] weekly_scoped_binding`: when `false`, the all-models
    /// window is the binding weekly reading — `weekly` and `weekly_resets_at`
    /// revert to the pre-override `seven_day` values; the scoped breakdown
    /// (`weekly_scoped*`) is untouched so samples keep the split.
    pub(crate) fn with_weekly_scoped_binding(mut self, binding: bool) -> Self {
        if !binding {
            self.usage.weekly = self.usage.weekly_all;
            self.usage.weekly_resets_at = self.weekly_all_resets_at;
        }
        self
    }
}

/// The provider's paid-extra-usage posture for an account (the response's
/// `extra_usage` block, or the `spend` block's equivalent). When enabled, an
/// account that reaches its weekly cap is billed per-credit rather than blocked,
/// which the dashboard surfaces so an operator can see overage before choosing
/// to ride an account past 100 %. All fields are non-secret amounts/flags.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct ExtraUsage {
    /// Whether paid extra usage is enabled for the account.
    pub(crate) is_enabled: bool,
    /// Credits consumed so far this month, in minor units of `currency`.
    pub(crate) used_credits: i64,
    /// The monthly credit limit in minor units, when the API reported one.
    pub(crate) monthly_limit: Option<i64>,
    /// Whether the spend limit has been reached.
    pub(crate) spend_limit_reached: bool,
    /// ISO-style currency code (e.g. `"USD"`), when reported.
    pub(crate) currency: Option<String>,
    /// Decimal places the minor-unit amounts are expressed in (`amount_minor /
    /// 10^decimal_places` = major units) — the `extra_usage` block's
    /// `decimal_places`, or a `spend` money object's `exponent`.
    pub(crate) decimal_places: u8,
}

/// The full parse of a usage response: both dimensions plus each window's reset
/// timestamp, kept VERBATIM as the API rendered them (ISO string or epoch). The
/// reset timestamps are extracted here (issue #5 acceptance: "returns session%,
/// account%, resets_at"); [`to_usage`](UsageReport::to_usage) is the swap-decision
/// projection, normalizing the weekly reset to epoch seconds for the all-exhausted
/// terminal logic (#11). Keeping the raw form here means this extraction stays a
/// faithful mirror of the response; normalization is a separate, tested concern.
#[derive(Debug, Clone, PartialEq)]
struct UsageReport {
    session: f64,
    weekly: f64,
    /// The ALL-MODELS weekly fraction — what `dimension("seven_day", …)` read
    /// BEFORE a higher `weekly_scoped` cap overrode `weekly`. Kept verbatim so
    /// the sample store can persist the overall-vs-scoped split.
    weekly_all: f64,
    /// The highest `weekly_scoped` entry's fraction, `None` when the response
    /// carried no scoped cap.
    weekly_scoped: Option<f64>,
    /// That scoped entry's `scope.model.display_name` (e.g. `"Fable"`), `None`
    /// when the scope or name is missing.
    weekly_scoped_model: Option<String>,
    /// The winning scoped entry's raw `resets_at` (see `weekly_resets_at`),
    /// regardless of whether the scoped cap outvoted the all-models window.
    weekly_scoped_resets_at: Option<String>,
    /// The raw `seven_day` `resets_at` captured BEFORE the scoped override —
    /// what `dimension(&root, "seven_day", …)` returned. Kept beside
    /// `weekly_resets_at` (which is the scoped reset while the override holds)
    /// so `[tunables] weekly_scoped_binding = false` can revert the binding
    /// weekly reading to the all-models window AND its own reset.
    weekly_all_resets_at: Option<String>,
    /// The account's paid-extra-usage posture — top-level `extra_usage`, or the
    /// `spend` block's equivalent when the former is absent; `None` when the
    /// response carried neither.
    extra_usage: Option<ExtraUsage>,
    /// Raw `resets_at` of the session window, as the API rendered it (ISO string
    /// or epoch-as-string); tolerant — `None` if absent/unrecognized. Projected to
    /// epoch seconds by [`to_usage`](UsageReport::to_usage) for the `status`
    /// per-account "resets in" display (issue #72).
    session_resets_at: Option<String>,
    /// Raw `resets_at` of the weekly window (see `session_resets_at`). Projected
    /// to epoch seconds by [`to_usage`](UsageReport::to_usage) for the
    /// all-exhausted terminal logic (#11).
    weekly_resets_at: Option<String>,
    /// The session window's provider-reported `severity`, verbatim, when present.
    /// Kept per-window here (the faithful raw mirror); [`to_usage`](UsageReport::to_usage) collapses the
    /// pair to a single reading-level label (issue #156 parse widening).
    session_severity: Option<String>,
    /// The weekly window's provider-reported `severity`, verbatim, when present
    /// (see `session_severity`).
    weekly_severity: Option<String>,
}

impl UsageReport {
    /// The swap-decision projection: the two usage dimensions the loop acts on,
    /// plus each window's reset normalized to epoch seconds. The weekly reset feeds
    /// the all-exhausted logic (issue #11) so it can compare reset times across
    /// accounts; the session reset feeds the `status` "resets in" display (issue
    /// #72). A reset the API did not supply — or that does not parse — projects to
    /// `None`.
    fn to_usage(&self) -> Usage {
        Usage {
            session: self.session,
            weekly: self.weekly,
            weekly_all: self.weekly_all,
            weekly_scoped: self.weekly_scoped,
            weekly_resets_at: self
                .weekly_resets_at
                .as_deref()
                .and_then(epoch_from_resets_at),
            session_resets_at: self
                .session_resets_at
                .as_deref()
                .and_then(epoch_from_resets_at),
        }
    }

    /// The reading's single severity label (issue #156): prefer the WEEKLY window's
    /// (the hard/binding limit the store emphasises), falling back to the SESSION
    /// window's; `None` when the API supplied neither. Kept off [`to_usage`](UsageReport::to_usage) so the
    /// lean `Usage` decision type is unaffected — only [`PolledReading`] carries it.
    fn severity(&self) -> Option<String> {
        self.weekly_severity
            .clone()
            .or_else(|| self.session_severity.clone())
    }
}

/// Seam: reads one account's usage quota as a [`PolledReading`] — the swap-decision
/// [`Usage`] plus the sample-only `severity`, from a SINGLE usage-API call (issue
/// #156). The only impl ([`RealUsageSource`]) polls the API; it is exercised in tests
/// through its [`UsageTransport`] seam (a fake transport returns scripted responses),
/// not a separate fake source.
pub(crate) trait UsageSource {
    async fn usage(&self) -> Result<PolledReading>;
}

/// Seam: performs the raw usage HTTP GET and reports the `(status, body)`. The
/// real impl ([`CurlTransport`]) reads the token and shells out to `curl`; a
/// test impl returns scripted responses, so the classification / parsing / 401
/// logic in [`RealUsageSource`] is exercised without a network.
pub(crate) trait UsageTransport {
    async fn fetch(&self) -> Result<HttpResponse>;
}

/// One HTTP exchange's result. `status == 0` is the sentinel for "no HTTP
/// response" (curl could not reach the endpoint), classified as Transient.
pub(crate) struct HttpResponse {
    status: u16,
    body: String,
    /// The server-advised `Retry-After` (delta-seconds form) when the response
    /// carried that header; `None` otherwise. Threaded into the rate-limited /
    /// transient errors so the daemon can honour it as a MINIMUM back-off wait
    /// (issue #76). Only a `4xx`/`5xx` carrying it is meaningful; on a `2xx` it is
    /// simply ignored.
    retry_after: Option<Duration>,
}

/// Real usage poller: classify the HTTP outcome and parse a success. Generic over
/// the transport so the whole `usage()` flow is testable against a fake. Stateless
/// — consecutive-401 streak tracking lives in the daemon's per-account health state
/// (issue #42), the only place a streak can persist across polls.
pub(crate) struct RealUsageSource<Tp> {
    transport: Tp,
}

impl<Tp> RealUsageSource<Tp> {
    pub(crate) fn new(transport: Tp) -> Self {
        Self { transport }
    }
}

impl<Tp> UsageSource for RealUsageSource<Tp>
where
    Tp: UsageTransport,
{
    async fn usage(&self) -> Result<PolledReading> {
        let response = self.transport.fetch().await?;
        let status = response.status;
        match classify_status(status) {
            UsageStatus::Unauthorized => Err(Error::UsageUnauthorized),
            // The one fetch+classify success path: parse ONCE and keep both the
            // decision `Usage` and the sample-only `severity` (issue #156), so the
            // daemon's sample piggybacks this poll with no extra API call.
            UsageStatus::Success => {
                let report = parse_usage(&response.body)?;
                Ok(PolledReading {
                    usage: report.to_usage(),
                    severity: report.severity(),
                    weekly_scoped_model: report.weekly_scoped_model.clone(),
                    weekly_scoped_resets_at: report
                        .weekly_scoped_resets_at
                        .as_deref()
                        .and_then(epoch_from_resets_at),
                    weekly_all_resets_at: report
                        .weekly_all_resets_at
                        .as_deref()
                        .and_then(epoch_from_resets_at),
                    extra_usage: report.extra_usage.clone(),
                })
            }
            UsageStatus::ScopeMissing => Err(Error::UsageScopeMissing),
            UsageStatus::RateLimited => Err(Error::UsageRateLimited {
                status,
                retry_after: response.retry_after,
            }),
            UsageStatus::ClientError => Err(Error::UsageRejected { status }),
            UsageStatus::Transient => Err(Error::UsageTransient {
                status,
                retry_after: response.retry_after,
            }),
        }
    }
}

/// Real transport: read the bearer from the stored credential and `curl` the
/// usage endpoint. Generic over [`CredentialStore`] so the token source can be
/// the canonical active item (now) or a per-account stash adapter (#6 / #7).
pub(crate) struct CurlTransport<C> {
    store: C,
}

impl<C: CredentialStore> CurlTransport<C> {
    pub(crate) fn new(store: C) -> Self {
        Self { store }
    }
}

impl<C: CredentialStore> UsageTransport for CurlTransport<C> {
    async fn fetch(&self) -> Result<HttpResponse> {
        let credential = self.store.read().await?;
        let token = access_token_from_blob(credential.expose())?;
        // The token-bearing config rides stdin (never argv) and is zeroized on
        // drop. argv is only `curl --config -`.
        let config = curl_config(&token);

        let mut child = Command::new(CURL)
            .arg("--config")
            .arg("-")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()?;
        {
            let mut stdin = child.stdin.take().expect("stdin was piped");
            stdin.write_all(config.as_bytes()).await?;
            // EOF so `curl` stops reading config and performs the request.
            stdin.shutdown().await?;
        }
        let output = child.wait_with_output().await?;

        // `curl` ran but got no HTTP response (DNS / connection / TLS / timeout):
        // a non-zero exit and no status line. Report the `0` "no response"
        // sentinel → classified Transient. (A failure to *spawn* curl is a true
        // I/O error and propagated via `?` above.)
        if !output.status.success() {
            return Ok(HttpResponse {
                status: 0,
                body: String::new(),
                retry_after: None,
            });
        }
        Ok(parse_curl_output(&output.stdout))
    }
}

/// The internal classification of one HTTP status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UsageStatus {
    Success,
    Transient,
    RateLimited,
    ClientError,
    Unauthorized,
    ScopeMissing,
}

/// Map an HTTP status to its taxonomy class. `401` / `403` / `429` are matched
/// before the `4xx` catch-all; `0` (no response) and any unexpected code fall to
/// Transient (skip + back off, never swap on it).
fn classify_status(status: u16) -> UsageStatus {
    match status {
        200..=299 => UsageStatus::Success,
        401 => UsageStatus::Unauthorized,
        403 => UsageStatus::ScopeMissing,
        429 => UsageStatus::RateLimited,
        400..=499 => UsageStatus::ClientError,
        500..=599 => UsageStatus::Transient,
        _ => UsageStatus::Transient,
    }
}

/// Extract the OAuth bearer from a `Claude Code-credentials` blob.
///
/// Documented shape: `{"claudeAiOauth":{"accessToken":"sk-ant-oat-…"}}` (the
/// canonical credential; `build/version-compat.md`: the token is the bearer).
/// Tolerant of a couple of plausible reshapes — top-level `accessToken` /
/// `access_token` — so a future change degrades to a clear typed error rather
/// than a mis-poll. Never echoes any blob bytes (issue #15 redaction).
fn access_token_from_blob(blob: &[u8]) -> Result<Zeroizing<String>> {
    let value: Value = serde_json::from_slice(blob).map_err(|_| Error::UsageTokenUnreadable)?;
    let token = value
        .get("claudeAiOauth")
        .and_then(|oauth| oauth.get("accessToken"))
        .or_else(|| value.get("accessToken"))
        .or_else(|| value.get("access_token"))
        .and_then(Value::as_str)
        .filter(|token| !token.is_empty())
        .ok_or(Error::UsageTokenUnreadable)?;
    Ok(Zeroizing::new(token.to_owned()))
}

/// Build the `curl --config -` body fed on stdin. Keeps the bearer out of argv;
/// `write-out` appends `\n<status>\n<retry-after>` after the body so the caller
/// can recover the HTTP status and the `Retry-After` response header (empty when
/// absent — `%header{}` requires curl ≥ 7.84, present on every supported macOS;
/// an unparseable value just degrades to `None`, issue #76). The token is an
/// opaque `sk-ant-…` string (no `"`/newline), so it needs no escaping inside the
/// quoted `header` value. Zeroized on drop.
fn curl_config(token: &str) -> Zeroizing<String> {
    let mut config = String::new();
    config.push_str("url = \"");
    config.push_str(USAGE_URL);
    config.push_str("\"\n");
    config.push_str("header = \"Authorization: Bearer ");
    config.push_str(token);
    config.push_str("\"\n");
    config.push_str("header = \"");
    config.push_str(BETA_HEADER);
    config.push_str("\"\n");
    config.push_str("header = \"Accept: application/json\"\n");
    // Quiet progress, but still print transport errors (to stderr, never stdout).
    config.push_str("silent\n");
    config.push_str("show-error\n");
    config.push_str(&format!("max-time = {POLL_TIMEOUT_SECS}\n"));
    // Body, then a newline and the numeric status, then a newline and the
    // `Retry-After` header value (empty when absent), on stdout.
    config.push_str("write-out = \"\\n%{http_code}\\n%header{retry-after}\"\n");
    Zeroizing::new(config)
}

/// Split `curl`'s stdout (`<body>\n<status>\n<retry-after>`, per the `write-out`
/// above) into a [`HttpResponse`]. The two metadata lines are peeled from the END,
/// so a multi-line body is preserved. Tolerant: the `Retry-After` line is empty
/// (→ `None`) on most responses; an unparseable trailing code becomes `0` (→
/// Transient); a no-body output is just the metadata.
fn parse_curl_output(stdout: &[u8]) -> HttpResponse {
    let text = String::from_utf8_lossy(stdout);
    // Peel the last line (the `Retry-After` value, empty when the header was
    // absent), then the new-last line (the status); everything before is the body.
    let Some((rest, retry_after_line)) = text.rsplit_once('\n') else {
        // No newline at all — degenerate; treat the whole output as the code.
        return HttpResponse {
            status: text.trim().parse().unwrap_or(0),
            body: String::new(),
            retry_after: None,
        };
    };
    let retry_after = parse_retry_after(retry_after_line);
    match rest.rsplit_once('\n') {
        Some((body, code)) => HttpResponse {
            status: code.trim().parse().unwrap_or(0),
            body: body.to_owned(),
            retry_after,
        },
        None => HttpResponse {
            status: rest.trim().parse().unwrap_or(0),
            body: String::new(),
            retry_after,
        },
    }
}

/// Parse a `Retry-After` header value into a back-off floor. Only the
/// delta-seconds form (a non-negative integer count of seconds) is honoured →
/// `Some(Duration)`; the HTTP-date form, and any empty or unparseable value,
/// yields `None`, leaving the daemon's exponential back-off to govern the wait
/// (issue #76). A negative or fractional value parses as neither a `u64` → `None`.
fn parse_retry_after(value: &str) -> Option<Duration> {
    let secs: u64 = value.trim().parse().ok()?;
    Some(Duration::from_secs(secs))
}

/// Parse a `200` usage body into both dimensions plus reset timestamps.
///
/// Tolerant of the two documented response shapes for each window:
///   - a top-level object — `five_hour` (session) / `seven_day` (weekly) — with a
///     `utilization` percentage (`0..=100`), or
///   - a `limits[]` entry whose `kind` matches the window, with a `percent`
///     (also `0..=100`).
///
/// A window that cannot be found is a hard [`Error::UsageParse`] (never a
/// fabricated `0`): the loop must skip, not swap, on missing data.
fn parse_usage(body: &str) -> Result<UsageReport> {
    let root: Value =
        serde_json::from_str(body).map_err(|_| Error::UsageParse("body is not JSON".into()))?;

    let (session, session_resets_at, session_severity) =
        dimension(&root, "five_hour", &["session", "five_hour"])
            .ok_or_else(|| Error::UsageParse("no session (five_hour) dimension".into()))?;
    let (weekly_all, mut weekly_resets_at, mut weekly_severity) =
        dimension(&root, "seven_day", &["weekly_all", "seven_day", "weekly"])
            .ok_or_else(|| Error::UsageParse("no weekly (seven_day) dimension".into()))?;
    let mut weekly = weekly_all;
    // The pre-override all-models reset, captured before the scoped override can
    // overwrite `weekly_resets_at` — `weekly_scoped_binding = false` restores it.
    let weekly_all_resets_at = weekly_resets_at.clone();

    // A per-model `weekly_scoped` cap above the all-models window is the binding
    // constraint — take it over the top-level `seven_day` reading. Either way the
    // breakdown is recorded: `weekly_all` keeps the pre-override all-models
    // fraction and the scoped fields carry the cap + its model name. `is_active`
    // is ignored: it marks the limit Claude's UI currently highlights, not
    // whether the cap applies — an inactive scoped entry still applies.
    let mut weekly_scoped = None;
    let mut weekly_scoped_model = None;
    let mut weekly_scoped_resets_at = None;
    if let Some(scoped) = max_weekly_scoped(&root) {
        weekly_scoped = Some(scoped.fraction);
        weekly_scoped_model = scoped.model;
        weekly_scoped_resets_at = scoped.resets_at.clone();
        if scoped.fraction > weekly {
            weekly = scoped.fraction;
            weekly_resets_at = scoped.resets_at;
            weekly_severity = scoped.severity;
        }
    }

    Ok(UsageReport {
        session,
        weekly,
        weekly_all,
        weekly_scoped,
        weekly_scoped_model,
        weekly_scoped_resets_at,
        weekly_all_resets_at,
        extra_usage: extra_usage_of(&root),
        session_resets_at,
        weekly_resets_at,
        session_severity,
        weekly_severity,
    })
}

/// Find one window's `(fraction, resets_at, severity)`, trying the top-level
/// `{top_key: {...}}` object first, then a `limits[]` entry whose `kind` is one
/// of `kinds`. Among matching `limits[]` entries an active one (`is_active` true
/// or absent) is preferred, but an explicitly inactive entry is used as a
/// fallback rather than dropped: `is_active: false` means "not the currently
/// binding limit" — the live API marks `weekly_all` so — NOT "absent" (issue
/// #66). Dropping it entirely would lose a window whenever it is the only match.
/// `severity` is the entry's optional provider label, extracted from the SAME
/// matched entry as the fraction (issue #156 parse widening).
fn dimension(
    root: &Value,
    top_key: &str,
    kinds: &[&str],
) -> Option<(f64, Option<String>, Option<String>)> {
    if let Some(obj) = root.get(top_key) {
        if let Some(fraction) = fraction_of(obj) {
            return Some((fraction, resets_at_of(obj), severity_of(obj)));
        }
    }
    let limits = root.get("limits").and_then(Value::as_array)?;
    // An active matching entry wins; an inactive one is kept only as a fallback
    // (is_active:false = "not the binding limit", not "absent" — issue #66).
    let mut fallback: Option<(f64, Option<String>, Option<String>)> = None;
    for entry in limits {
        let kind = entry.get("kind").and_then(Value::as_str);
        if !kind.is_some_and(|kind| kinds.contains(&kind)) {
            continue;
        }
        let Some(fraction) = fraction_of(entry) else {
            continue;
        };
        let active = entry
            .get("is_active")
            .and_then(Value::as_bool)
            .unwrap_or(true);
        if active {
            return Some((fraction, resets_at_of(entry), severity_of(entry)));
        }
        if fallback.is_none() {
            fallback = Some((fraction, resets_at_of(entry), severity_of(entry)));
        }
    }
    fallback
}

/// One `limits[]` entry of `kind == "weekly_scoped"` — a per-model weekly
/// cap — distilled to what [`parse_usage`] needs. Named fields rather than a
/// four-element tuple, so the three optional strings (`resets_at` / `severity` /
/// `model`) cannot be confused positionally.
struct WeeklyScoped {
    /// Fraction in `[0.0, 1.0]` of the scoped weekly cap consumed.
    fraction: f64,
    /// Raw `resets_at` as the API rendered it (see [`UsageReport::weekly_resets_at`]).
    resets_at: Option<String>,
    /// The entry's provider `severity` label, when present.
    severity: Option<String>,
    /// `scope.model.display_name` (e.g. `"Fable"`), `None` when the scope or the
    /// name is absent — a scoped cap with no model name still counts.
    model: Option<String>,
}

/// The highest [`WeeklyScoped`] among `limits[]` entries with
/// `kind == "weekly_scoped"` that carry a readable fraction. `is_active` is
/// NOT consulted: it marks which limit Claude's UI currently highlights (the
/// most severe one), not whether the cap applies — the live API routinely marks
/// a still-applicable `weekly_scoped` entry `is_active: false` (e.g. while the
/// session limit is the highlighted one), so skipping it would under-report the
/// account's true per-model weekly load. A scoped per-model weekly cap is the
/// BINDING constraint when it is higher than the all-models window:
/// `weekly_all` aggregates every model, so a Fable-heavy account can sit at
/// 44 % overall while its Fable cap is already at 74 %, and it is the 74 %
/// that exhausts the account. `None` when no scoped entry carries a readable
/// fraction.
fn max_weekly_scoped(root: &Value) -> Option<WeeklyScoped> {
    let limits = root.get("limits").and_then(Value::as_array)?;
    let mut max: Option<WeeklyScoped> = None;
    for entry in limits {
        let kind = entry.get("kind").and_then(Value::as_str);
        if kind != Some("weekly_scoped") {
            continue;
        }
        let Some(fraction) = fraction_of(entry) else {
            continue;
        };
        if max.as_ref().is_none_or(|best| fraction > best.fraction) {
            max = Some(WeeklyScoped {
                fraction,
                resets_at: resets_at_of(entry),
                severity: severity_of(entry),
                model: scoped_model_of(entry),
            });
        }
    }
    max
}

/// The scoped model's display name from a `limits[]` entry —
/// `scope.model.display_name`; `None` when the scope, the model object, or the
/// name itself is absent or not a string.
fn scoped_model_of(entry: &Value) -> Option<String> {
    entry
        .get("scope")?
        .get("model")?
        .get("display_name")?
        .as_str()
        .map(str::to_owned)
}

/// Read the account's paid-extra-usage posture: the top-level `extra_usage`
/// object when present, otherwise the `spend` block's equivalent (`used` /
/// `limit` money objects plus `enabled`); `None` when the response carried
/// neither. Tolerant — a missing field takes its neutral default rather than
/// failing the parse, and no identifiers or amounts are ever logged here.
fn extra_usage_of(root: &Value) -> Option<ExtraUsage> {
    if let Some(obj) = root.get("extra_usage") {
        return Some(ExtraUsage {
            is_enabled: obj
                .get("is_enabled")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            used_credits: obj.get("used_credits").and_then(Value::as_i64).unwrap_or(0),
            monthly_limit: obj.get("monthly_limit").and_then(Value::as_i64),
            spend_limit_reached: obj
                .get("spend_limit_reached")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            currency: obj
                .get("currency")
                .and_then(Value::as_str)
                .map(str::to_owned),
            decimal_places: obj
                .get("decimal_places")
                .and_then(Value::as_u64)
                .map(|places| places.min(u64::from(u8::MAX)) as u8)
                .unwrap_or(0),
        });
    }
    let spend = root.get("spend")?;
    let money_field = |name: &str| {
        spend
            .get(name)
            .and_then(|money| money.get("amount_minor"))
            .and_then(Value::as_i64)
    };
    let money_str = |name: &str, key: &str| {
        spend
            .get(name)
            .and_then(|money| money.get(key))
            .and_then(Value::as_str)
    };
    let money_num = |name: &str, key: &str| {
        spend
            .get(name)
            .and_then(|money| money.get(key))
            .and_then(Value::as_u64)
    };
    Some(ExtraUsage {
        is_enabled: spend
            .get("enabled")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        used_credits: money_field("used").unwrap_or(0),
        monthly_limit: money_field("limit"),
        // `spend` carries no explicit limit-reached flag; derive it from the
        // reported limit and usage.
        spend_limit_reached: match (money_field("used"), money_field("limit")) {
            (Some(used), Some(limit)) => used >= limit,
            _ => false,
        },
        currency: money_str("used", "currency")
            .or_else(|| money_str("limit", "currency"))
            .map(str::to_owned),
        decimal_places: money_num("used", "exponent")
            .or_else(|| money_num("limit", "exponent"))
            .map(|exp| exp.min(u64::from(u8::MAX)) as u8)
            .unwrap_or(0),
    })
}

/// Read a usage fraction in `[0.0, 1.0]` from an object: a `utilization`
/// percentage or a `percent`, each divided by 100. Both fields are on the same
/// `0..=100` scale — `utilization` is the top-level window's spelling, `percent`
/// the `limits[]` entry's (issue #66) — and `utilization` is tried first, with
/// `percent` the fallback when it is absent. Clamped, so a stray `> 100` reading
/// can never exceed a full window.
fn fraction_of(obj: &Value) -> Option<f64> {
    let percent = obj
        .get("utilization")
        .and_then(Value::as_f64)
        .or_else(|| obj.get("percent").and_then(Value::as_f64))?;
    Some((percent / 100.0).clamp(0.0, 1.0))
}

/// Read `resets_at` as the API rendered it — a string verbatim, or a number
/// stringified — tolerant of either; `None` if absent or another type.
fn resets_at_of(obj: &Value) -> Option<String> {
    match obj.get("resets_at") {
        Some(Value::String(at)) => Some(at.clone()),
        Some(Value::Number(at)) => Some(at.to_string()),
        _ => None,
    }
}

/// Read a window's optional `severity` label verbatim (issue #156 parse widening).
/// A tolerant free string — only a JSON string is taken; any other type (or an
/// absent field) is `None`, so a reading without a severity still parses cleanly.
fn severity_of(obj: &Value) -> Option<String> {
    match obj.get("severity") {
        Some(Value::String(label)) => Some(label.clone()),
        _ => None,
    }
}

/// Normalize a raw `resets_at` (as [`resets_at_of`] captured it) to epoch seconds,
/// so the all-exhausted logic (#11) can order reset times across accounts.
/// Tolerant of the two shapes the API uses: a whole-second epoch rendered as
/// digits, or an RFC 3339 instant (`2026-06-30T00:00:00Z`). `None` for anything
/// it cannot parse — a missing reset time is never fatal; the terminal signal just
/// omits it.
fn epoch_from_resets_at(raw: &str) -> Option<i64> {
    let raw = raw.trim();
    if let Ok(epoch) = raw.parse::<i64>() {
        return Some(epoch);
    }
    epoch_from_rfc3339(raw)
}

/// Parse an RFC 3339 / ISO 8601 instant to epoch seconds, second-granular.
/// Tolerant: accepts a `Z`/`z` suffix, an explicit `±HH:MM` offset, or none
/// (treated as UTC); a fractional-seconds part is dropped. `None` on any deviation
/// from the expected `YYYY-MM-DDTHH:MM:SS` shape, so a surprising format degrades
/// to "reset time unknown" rather than a wrong instant.
///
/// `pub(crate)` so [`crate::observability::last_swap_at`] can parse the event log's
/// `ts=` field back to an instant through this SAME canonical parser (the log
/// renders it via `observability::rfc3339`), rather than hand-rolling a second
/// copy of the leap-year arithmetic for the cooldown gate (#63).
pub(crate) fn epoch_from_rfc3339(s: &str) -> Option<i64> {
    let (date, rest) = s.split_once('T').or_else(|| s.split_once(' '))?;

    let mut date_parts = date.split('-');
    let year: i64 = date_parts.next()?.parse().ok()?;
    let month: u32 = date_parts.next()?.parse().ok()?;
    let day: u32 = date_parts.next()?.parse().ok()?;
    // Reject anything the civil calendar does not admit, so a malformed upstream
    // value degrades to "reset unknown" rather than a fabricated instant (#177):
    // a 4-digit year bound keeps `days_from_civil` clear of `i64` overflow, and
    // the per-month day length rejects `2026-02-30` / `2025-02-29` instead of
    // silently normalizing them forward. `||` short-circuits, so `days_in_month`
    // only sees a `1..=12` month.
    if date_parts.next().is_some()
        || !(0..=9999).contains(&year)
        || !(1..=12).contains(&month)
        || !(1..=days_in_month(year, month)).contains(&day)
    {
        return None;
    }

    let (time, offset_secs) = split_offset(rest)?;
    let time = time.split('.').next().unwrap_or(time);
    let mut time_parts = time.split(':');
    let hour: i64 = time_parts.next()?.parse().ok()?;
    let minute: i64 = time_parts.next()?.parse().ok()?;
    let second: i64 = time_parts.next().unwrap_or("0").parse().ok()?;
    if time_parts.next().is_some()
        || !(0..=23).contains(&hour)
        || !(0..=59).contains(&minute)
        || !(0..=60).contains(&second)
    {
        return None;
    }

    let days = days_from_civil(year, month, day);
    Some(days * 86_400 + hour * 3_600 + minute * 60 + second - offset_secs)
}

/// Split a time-of-day from its timezone suffix, returning the bare time and the
/// offset in seconds EAST of UTC (so `epoch = local - offset`). `Z`/`z` or no
/// suffix is UTC (offset 0); an explicit `±HH:MM` / `±HHMM` / `±HH` is parsed.
fn split_offset(rest: &str) -> Option<(&str, i64)> {
    if let Some(time) = rest.strip_suffix(['Z', 'z']) {
        return Some((time, 0));
    }
    let Some(pos) = rest.rfind(['+', '-']) else {
        return Some((rest, 0)); // no offset → UTC
    };
    let (time, tz) = rest.split_at(pos);
    let sign = if tz.starts_with('-') { -1 } else { 1 };
    let tz = &tz[1..];
    let (hours, minutes) = match tz.split_once(':') {
        Some((h, m)) => (h.parse::<i64>().ok()?, m.parse::<i64>().ok()?),
        // `±HHMM`: slice via `get` (not `tz[..2]`) so a non-ASCII byte landing
        // mid-char yields None instead of panicking on the daemon poll loop (#177).
        None if tz.len() == 4 => (tz.get(..2)?.parse().ok()?, tz.get(2..)?.parse().ok()?),
        None => (tz.parse::<i64>().ok()?, 0),
    };
    Some((time, sign * (hours * 3_600 + minutes * 60)))
}

/// Days since 1970-01-01 for a proleptic-Gregorian civil date — Howard Hinnant's
/// `days_from_civil`, the inverse of the `civil_from_days` the event log uses to
/// render the reset back. Correct across leap years and the 100/400 century rules
/// for the post-epoch dates the usage API returns.
fn days_from_civil(year: i64, month: u32, day: u32) -> i64 {
    let year = if month <= 2 { year - 1 } else { year };
    let era = (if year >= 0 { year } else { year - 399 }) / 400;
    let yoe = year - era * 400; // [0, 399]
    let month = i64::from(month);
    let day = i64::from(day);
    let doy = (153 * (if month > 2 { month - 3 } else { month + 9 }) + 2) / 5 + (day - 1); // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    era * 146_097 + doe - 719_468
}

/// Whether `year` is a Gregorian leap year — the 4 / 100 / 400 rule (divisible by
/// 4, except whole centuries, except those divisible by 400, so 2000 is a leap
/// year and 2100 is not). Only bounds February in [`days_in_month`]; the epoch
/// arithmetic itself lives in [`days_from_civil`].
fn is_leap_year(year: i64) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

/// The length of `month` (`1..=12`) in `year`, honouring leap February. Lets
/// [`epoch_from_rfc3339`] reject an impossible calendar date instead of letting
/// [`days_from_civil`] normalize it to a wrong instant (#177). Callers guarantee
/// `month ∈ 1..=12`; any other value is treated as zero-length so it can never
/// masquerade as a valid day count.
fn days_in_month(year: i64, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap_year(year) => 29,
        2 => 28,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    // --- classify_status (pure, the taxonomy) ---

    #[test]
    fn classify_maps_each_status_to_its_taxonomy_class() {
        assert_eq!(classify_status(200), UsageStatus::Success);
        assert_eq!(classify_status(204), UsageStatus::Success);
        assert_eq!(classify_status(401), UsageStatus::Unauthorized);
        assert_eq!(classify_status(403), UsageStatus::ScopeMissing);
        assert_eq!(classify_status(429), UsageStatus::RateLimited);
        // Other non-401/403 4xx are ClientError (G4: back off + skip).
        assert_eq!(classify_status(400), UsageStatus::ClientError);
        assert_eq!(classify_status(404), UsageStatus::ClientError);
        assert_eq!(classify_status(422), UsageStatus::ClientError);
        // 5xx and the no-response sentinel are Transient.
        assert_eq!(classify_status(500), UsageStatus::Transient);
        assert_eq!(classify_status(503), UsageStatus::Transient);
        assert_eq!(classify_status(0), UsageStatus::Transient);
    }

    #[test]
    fn forbidden_is_classified_distinctly_from_unauthorized() {
        // Issue #5 acceptance: 403 (scope) must be distinct from 401.
        assert_ne!(classify_status(403), classify_status(401));
    }

    // --- parse_usage (tolerant, both shapes) ---

    #[test]
    fn parses_the_limits_array_shape_with_percent() {
        let body = r#"{
            "limits": [
                {"kind": "session", "percent": 42, "resets_at": "2026-06-26T18:00:00Z", "is_active": true},
                {"kind": "weekly_all", "percent": 88, "severity": "critical", "resets_at": "2026-06-30T00:00:00Z", "is_active": true}
            ]
        }"#;
        let report = parse_usage(body).unwrap();
        assert!((report.session - 0.42).abs() < 1e-9);
        assert!((report.weekly - 0.88).abs() < 1e-9);
        assert_eq!(
            report.session_resets_at.as_deref(),
            Some("2026-06-26T18:00:00Z")
        );
        assert_eq!(
            report.weekly_resets_at.as_deref(),
            Some("2026-06-30T00:00:00Z")
        );
    }

    #[test]
    fn parse_widening_retains_severity_and_prefers_weekly() {
        // Issue #156 parse widening: the `limits[]` `severity` (previously dropped)
        // is retained per window, and the reading's single label prefers the WEEKLY
        // window's, falling back to the session window's.
        let both = r#"{
            "limits": [
                {"kind": "session", "percent": 42, "severity": "warning", "is_active": true},
                {"kind": "weekly_all", "percent": 88, "severity": "critical", "is_active": true}
            ]
        }"#;
        let report = parse_usage(both).unwrap();
        assert_eq!(report.session_severity.as_deref(), Some("warning"));
        assert_eq!(report.weekly_severity.as_deref(), Some("critical"));
        assert_eq!(
            report.severity().as_deref(),
            Some("critical"),
            "weekly wins"
        );

        // Only the session window labels its severity → it is used.
        let session_only = r#"{
            "limits": [
                {"kind": "session", "percent": 20, "severity": "warning", "is_active": true},
                {"kind": "weekly_all", "percent": 10, "is_active": true}
            ]
        }"#;
        assert_eq!(
            parse_usage(session_only).unwrap().severity().as_deref(),
            Some("warning"),
            "falls back to session"
        );

        // Neither window labels a severity → None; the reading still parses cleanly
        // (issue #156 AC: a sample is valid when the optional field is absent).
        let neither = r#"{
            "limits": [
                {"kind": "session", "percent": 20, "is_active": true},
                {"kind": "weekly_all", "percent": 10, "is_active": true}
            ]
        }"#;
        let report = parse_usage(neither).unwrap();
        assert_eq!(report.severity(), None);
        assert!(
            (report.session - 0.20).abs() < 1e-9,
            "dimensions still parse"
        );
    }

    #[test]
    fn parses_the_top_level_window_shape_with_utilization() {
        // `utilization` is a 0..=100 percentage (issue #66): 10.0 → 0.10 fraction.
        let body = r#"{
            "five_hour": {"utilization": 10.0, "resets_at": "2026-06-26T18:00:00Z"},
            "seven_day": {"utilization": 55.0, "resets_at": "2026-06-30T00:00:00Z"}
        }"#;
        let report = parse_usage(body).unwrap();
        assert!((report.session - 0.10).abs() < 1e-9);
        assert!((report.weekly - 0.55).abs() < 1e-9);
    }

    #[test]
    fn parses_the_live_response_shape_utilization_on_a_0_to_100_scale() {
        // Regression fixture (issue #66): the verbatim shape captured from a live
        // `GET /api/oauth/usage` (HTTP 200). `utilization` is on a 0..=100 scale
        // — matching the sibling `limits[].percent` — NOT a 0..=1 fraction, so
        // `82.0` is 82 % (fraction 0.82), not a saturated full window. Before the
        // fix, `fraction_of` clamped `82.0 → 1.0`, reporting every account `100 %`.
        let body = r#"{
            "five_hour": { "utilization": 82.0, "resets_at": "2026-06-29T13:40:00.475727+00:00" },
            "seven_day": { "utilization": 15.0, "resets_at": "2026-07-06T00:00:00.475752+00:00" },
            "limits": [
                { "kind": "session",   "percent": 82, "is_active": true  },
                { "kind": "weekly_all", "percent": 15, "is_active": false }
            ]
        }"#;
        let report = parse_usage(body).unwrap();
        assert!(
            (report.session - 0.82).abs() < 1e-9,
            "session fraction was {}",
            report.session
        );
        assert!(
            (report.weekly - 0.15).abs() < 1e-9,
            "weekly fraction was {}",
            report.weekly
        );
    }

    #[test]
    fn weekly_scoped_higher_than_weekly_all_wins() {
        // Regression fixture: the live payload shape carries BOTH a top-level
        // `seven_day` window (44 %) and an active `limits[]` entry of kind
        // `weekly_scoped` for one model (Fable, 74 %). The scoped per-model cap
        // is the binding constraint — it is what exhausts the account — so it
        // outvotes the lower all-models window, resets_at included.
        let body = r#"{
            "five_hour": { "utilization": 0.0, "resets_at": null },
            "seven_day": { "utilization": 44.0, "resets_at": "2026-09-24T16:00:00.485164+00:00" },
            "limits": [
                { "kind": "session", "group": "session", "percent": 0, "severity": "normal",
                  "resets_at": null, "scope": null, "is_active": false },
                { "kind": "weekly_all", "group": "weekly", "percent": 44, "severity": "normal",
                  "resets_at": "2026-09-24T16:00:00.485164+00:00", "scope": null, "is_active": false },
                { "kind": "weekly_scoped", "group": "weekly", "percent": 74, "severity": "normal",
                  "resets_at": "2026-09-24T16:00:00.485335+00:00",
                  "scope": { "model": { "id": null, "display_name": "Fable" }, "surface": null },
                  "is_active": true }
            ]
        }"#;
        let report = parse_usage(body).unwrap();
        assert!((report.weekly - 0.74).abs() < 1e-9);
        // The breakdown keeps the pre-override all-models reading alongside the
        // winning scoped cap + its model name, so a consumer can chart the split.
        assert!((report.weekly_all - 0.44).abs() < 1e-9);
        assert_eq!(report.weekly_scoped, Some(0.74));
        assert_eq!(report.weekly_scoped_model.as_deref(), Some("Fable"));
        assert_eq!(
            report.weekly_resets_at.as_deref(),
            Some("2026-09-24T16:00:00.485335+00:00")
        );
        assert!((report.session - 0.0).abs() < 1e-9);
    }

    #[test]
    fn weekly_scoped_lower_than_weekly_all_is_ignored() {
        // A scoped cap BELOW the all-models window is not the binding constraint;
        // the weekly triple stays the `seven_day` reading.
        let body = r#"{
            "five_hour": { "utilization": 0.0, "resets_at": null },
            "seven_day": { "utilization": 44.0, "resets_at": "2026-09-24T16:00:00.485164+00:00" },
            "limits": [
                { "kind": "weekly_scoped", "group": "weekly", "percent": 10, "severity": "normal",
                  "resets_at": "2026-09-24T16:00:00.485335+00:00",
                  "scope": { "model": { "id": null, "display_name": "Fable" }, "surface": null },
                  "is_active": true }
            ]
        }"#;
        let report = parse_usage(body).unwrap();
        assert!((report.weekly - 0.44).abs() < 1e-9);
        assert_eq!(
            report.weekly_resets_at.as_deref(),
            Some("2026-09-24T16:00:00.485164+00:00")
        );
    }

    #[test]
    fn inactive_weekly_scoped_still_applies() {
        // `is_active: false` marks which limit Claude's UI currently highlights
        // — it does NOT switch the cap off. An inactive 90 % scoped entry is
        // still the binding constraint over the 44 % all-models window.
        let body = r#"{
            "five_hour": { "utilization": 0.0, "resets_at": null },
            "seven_day": { "utilization": 44.0, "resets_at": "2026-09-24T16:00:00.485164+00:00" },
            "limits": [
                { "kind": "weekly_scoped", "group": "weekly", "percent": 90, "severity": "normal",
                  "resets_at": "2026-09-24T16:00:00.485335+00:00",
                  "scope": { "model": { "id": null, "display_name": "Fable" }, "surface": null },
                  "is_active": false }
            ]
        }"#;
        let report = parse_usage(body).unwrap();
        assert!((report.weekly - 0.90).abs() < 1e-9);
        assert_eq!(report.weekly_scoped, Some(0.90));
        assert_eq!(report.weekly_scoped_model.as_deref(), Some("Fable"));
        // The scoped cap overrode → its reset becomes the weekly reset.
        assert_eq!(
            report.weekly_resets_at.as_deref(),
            Some("2026-09-24T16:00:00.485335+00:00")
        );
        assert_eq!(
            report.weekly_scoped_resets_at.as_deref(),
            Some("2026-09-24T16:00:00.485335+00:00")
        );
    }

    #[test]
    fn the_highest_scoped_entry_wins_regardless_of_is_active() {
        // Two scoped caps: Fable 70 % inactive, Sonnet 20 % active. `is_active`
        // marks the highlighted limit, not applicability, so the higher cap
        // (Fable) still wins.
        let body = r#"{
            "five_hour": { "utilization": 0.0, "resets_at": null },
            "seven_day": { "utilization": 44.0, "resets_at": "2026-09-24T16:00:00.485164+00:00" },
            "limits": [
                { "kind": "weekly_scoped", "group": "weekly", "percent": 70, "severity": "normal",
                  "resets_at": "2026-09-24T16:00:00.485335+00:00",
                  "scope": { "model": { "id": null, "display_name": "Fable" }, "surface": null },
                  "is_active": false },
                { "kind": "weekly_scoped", "group": "weekly", "percent": 20, "severity": "normal",
                  "resets_at": "2026-09-25T16:00:00.485335+00:00",
                  "scope": { "model": { "id": null, "display_name": "Sonnet" }, "surface": null },
                  "is_active": true }
            ]
        }"#;
        let report = parse_usage(body).unwrap();
        assert!((report.weekly - 0.70).abs() < 1e-9);
        assert_eq!(report.weekly_scoped, Some(0.70));
        assert_eq!(report.weekly_scoped_model.as_deref(), Some("Fable"));
    }

    /// The scrubbed live capture of acct-14's `GET /api/oauth/usage` response
    /// (token/ids removed): session 90 % active, weekly_all 53 % inactive, and a
    /// `weekly_scoped` Fable cap at 70 % — ALSO inactive, because `is_active`
    /// marks the highlighted limit (the session one), not applicability. This is
    /// the payload that exposed the bug: the scoped cap must still read 0.70 and
    /// override the binding `weekly`.
    const ACCT14_USAGE_SCRUBBED: &str = r#"{
        "five_hour": {
            "utilization": 90,
            "resets_at": "2026-09-22T20:00:00.618246+00:00"
        },
        "seven_day": {
            "utilization": 53,
            "resets_at": "2026-09-27T04:00:00.618269+00:00"
        },
        "extra_usage": {
            "is_enabled": true,
            "monthly_limit": 40000,
            "used_credits": 0,
            "currency": "USD",
            "decimal_places": 2,
            "spend_limit_reached": false
        },
        "limits": [
            {
                "kind": "session",
                "percent": 90,
                "severity": "critical",
                "resets_at": "2026-09-22T20:00:00.618246+00:00",
                "is_active": true
            },
            {
                "kind": "weekly_all",
                "percent": 53,
                "severity": "normal",
                "resets_at": "2026-09-27T04:00:00.618269+00:00",
                "is_active": false
            },
            {
                "kind": "weekly_scoped",
                "percent": 70,
                "severity": "normal",
                "resets_at": "2026-09-27T04:00:00.618462+00:00",
                "scope": {
                    "model": {
                        "id": "model-id-redacted",
                        "display_name": "Fable"
                    },
                    "surface": null
                },
                "is_active": false
            }
        ],
        "spend": {
            "used": {
                "amount_minor": 0,
                "currency": "USD",
                "exponent": 2
            },
            "limit": {
                "amount_minor": 40000,
                "currency": "USD",
                "exponent": 2
            },
            "percent": 0,
            "severity": "normal",
            "enabled": true
        }
    }"#;

    #[test]
    fn parses_the_scrubbed_acct14_capture_inactive_scoped_included() {
        let report = parse_usage(ACCT14_USAGE_SCRUBBED).unwrap();
        assert!(
            (report.weekly_all - 0.53).abs() < 1e-9,
            "weekly_all was {}",
            report.weekly_all
        );
        assert_eq!(report.weekly_scoped, Some(0.70));
        // The inactive scoped cap is still applicable — and binding.
        assert!(
            (report.weekly - 0.70).abs() < 1e-9,
            "weekly was {}",
            report.weekly
        );
        assert_eq!(report.weekly_scoped_model.as_deref(), Some("Fable"));
        // The scoped entry overrode → the weekly reset is its reset.
        assert_eq!(
            report.weekly_resets_at.as_deref(),
            Some("2026-09-27T04:00:00.618462+00:00")
        );
        assert_eq!(
            report.weekly_scoped_resets_at.as_deref(),
            Some("2026-09-27T04:00:00.618462+00:00")
        );
        // The `extra_usage` block takes precedence over `spend`.
        let extra = report.extra_usage.expect("extra_usage parsed");
        assert!(extra.is_enabled);
        assert_eq!(extra.used_credits, 0);
        assert_eq!(extra.monthly_limit, Some(40000));
        assert!(!extra.spend_limit_reached);
        assert_eq!(extra.currency.as_deref(), Some("USD"));
        assert_eq!(extra.decimal_places, 2);
    }

    #[test]
    fn extra_usage_falls_back_to_the_spend_block() {
        // When `extra_usage` is absent the equivalent `spend` block supplies the
        // same posture (used/limit money objects + `enabled`); the limit-reached
        // flag is derived from used >= limit.
        let body = r#"{
            "five_hour": { "utilization": 10.0 },
            "seven_day": { "utilization": 20.0 },
            "spend": {
                "used": { "amount_minor": 1500, "currency": "USD", "exponent": 2 },
                "limit": { "amount_minor": 40000, "currency": "USD", "exponent": 2 },
                "enabled": true
            }
        }"#;
        let extra = parse_usage(body).unwrap().extra_usage.expect("fallback");
        assert!(extra.is_enabled);
        assert_eq!(extra.used_credits, 1500);
        assert_eq!(extra.monthly_limit, Some(40000));
        assert!(!extra.spend_limit_reached);
        assert_eq!(extra.currency.as_deref(), Some("USD"));
        assert_eq!(extra.decimal_places, 2);

        let reached = r#"{
            "five_hour": { "utilization": 10.0 },
            "seven_day": { "utilization": 20.0 },
            "spend": {
                "used": { "amount_minor": 40000, "currency": "USD", "exponent": 2 },
                "limit": { "amount_minor": 40000, "currency": "USD", "exponent": 2 },
                "enabled": true
            }
        }"#;
        assert!(
            parse_usage(reached)
                .unwrap()
                .extra_usage
                .unwrap()
                .spend_limit_reached,
            "used == limit derives spend_limit_reached"
        );
    }

    #[test]
    fn extra_usage_is_absent_when_neither_block_exists() {
        let body = r#"{
            "five_hour": { "utilization": 10.0 },
            "seven_day": { "utilization": 20.0 }
        }"#;
        assert_eq!(parse_usage(body).unwrap().extra_usage, None);
    }

    #[test]
    fn no_limits_array_still_parses() {
        // Regression guard: a payload with only the top-level windows and no
        // `limits` key parses exactly as before the weekly_scoped scan.
        let body = r#"{
            "five_hour": { "utilization": 0.0, "resets_at": null },
            "seven_day": { "utilization": 44.0, "resets_at": "2026-09-24T16:00:00.485164+00:00" }
        }"#;
        let report = parse_usage(body).unwrap();
        assert!((report.weekly - 0.44).abs() < 1e-9);
        // No `limits[]` → no scoped breakdown: `weekly_all` mirrors `weekly`.
        assert_eq!(report.weekly_scoped, None);
        assert_eq!(report.weekly_scoped_model, None);
        assert!((report.weekly_all - 0.44).abs() < 1e-9);
        assert_eq!(
            report.weekly_resets_at.as_deref(),
            Some("2026-09-24T16:00:00.485164+00:00")
        );
    }

    #[test]
    fn parses_kind_aliases_in_the_limits_array() {
        // five_hour / seven_day as `kind` values inside limits[].
        let body = r#"{
            "limits": [
                {"kind": "five_hour", "percent": 25},
                {"kind": "seven_day", "percent": 60}
            ]
        }"#;
        let report = parse_usage(body).unwrap();
        assert!((report.session - 0.25).abs() < 1e-9);
        assert!((report.weekly - 0.60).abs() < 1e-9);
    }

    #[test]
    fn prefers_an_active_limit_entry_over_an_inactive_duplicate() {
        // When the same kind appears twice, the active entry wins over the
        // inactive one (issue #66: inactive is a fallback, not a hard skip).
        let body = r#"{
            "limits": [
                {"kind": "session", "percent": 99, "is_active": false},
                {"kind": "session", "percent": 12, "is_active": true},
                {"kind": "weekly_all", "percent": 30}
            ]
        }"#;
        let report = parse_usage(body).unwrap();
        assert!((report.session - 0.12).abs() < 1e-9);
    }

    #[test]
    fn falls_back_to_an_inactive_limit_when_no_active_one_exists() {
        // `is_active: false` means "not the currently-binding limit", NOT "absent"
        // (issue #66). When the ONLY matching entry is inactive, read it rather
        // than fabricating a missing-dimension error — the live API marks
        // `weekly_all` inactive, and here no top-level `seven_day` masks it.
        let body = r#"{
            "limits": [
                {"kind": "session", "percent": 40, "is_active": true},
                {"kind": "weekly_all", "percent": 15, "is_active": false}
            ]
        }"#;
        let report = parse_usage(body).unwrap();
        assert!((report.session - 0.40).abs() < 1e-9);
        assert!((report.weekly - 0.15).abs() < 1e-9);
    }

    #[test]
    fn an_over_full_reading_clamps_to_a_full_window() {
        // Both spellings are a 0..=100 percentage (issue #66); an over-100 stray
        // reading clamps to a single full window, never beyond — whether it
        // arrives as `percent` or `utilization`.
        let body = r#"{
            "limits": [
                {"kind": "session", "percent": 150},
                {"kind": "weekly_all", "utilization": 150}
            ]
        }"#;
        let report = parse_usage(body).unwrap();
        assert_eq!(report.session, 1.0);
        assert_eq!(report.weekly, 1.0);
    }

    #[test]
    fn resets_at_tolerates_a_numeric_timestamp() {
        let body = r#"{
            "five_hour": {"utilization": 10.0, "resets_at": 1750960800},
            "seven_day": {"utilization": 20.0}
        }"#;
        let report = parse_usage(body).unwrap();
        assert_eq!(report.session_resets_at.as_deref(), Some("1750960800"));
        // An absent resets_at is tolerated as None, not an error.
        assert_eq!(report.weekly_resets_at, None);
    }

    #[test]
    fn a_missing_dimension_is_a_parse_error_not_a_fabricated_zero() {
        // Only session present → weekly missing → hard error (never swap on
        // missing data).
        let body = r#"{"limits": [{"kind": "session", "percent": 10}]}"#;
        assert!(matches!(parse_usage(body), Err(Error::UsageParse(_))));
    }

    #[test]
    fn non_json_body_is_a_parse_error() {
        assert!(matches!(parse_usage("not json"), Err(Error::UsageParse(_))));
    }

    // --- resets_at normalization to epoch (issue #11) ---

    #[test]
    fn epoch_from_resets_at_parses_a_digit_epoch_string() {
        // The numeric API shape arrives stringified (see `resets_at_of`).
        assert_eq!(epoch_from_resets_at("1750960800"), Some(1_750_960_800));
        assert_eq!(epoch_from_resets_at("  1750960800  "), Some(1_750_960_800));
    }

    #[test]
    fn epoch_from_resets_at_parses_an_rfc3339_instant() {
        // Cross-checked against the event log's `rfc3339` renderer, which maps
        // 1_750_960_800 -> 2025-06-26T18:00:00Z (see the observability tests), so
        // the two halves of the round-trip agree.
        assert_eq!(
            epoch_from_resets_at("2025-06-26T18:00:00Z"),
            Some(1_750_960_800)
        );
        // A lower-case `z` and a fractional-seconds part are both tolerated; the
        // fraction is dropped (the log is second-granular).
        assert_eq!(
            epoch_from_resets_at("2025-06-26T18:00:00.512z"),
            Some(1_750_960_800)
        );
    }

    #[test]
    fn epoch_from_rfc3339_applies_a_timezone_offset() {
        // 20:00:00+02:00 and 13:00:00-05:00 are both the 18:00:00Z instant.
        assert_eq!(
            epoch_from_rfc3339("2025-06-26T20:00:00+02:00"),
            Some(1_750_960_800)
        );
        assert_eq!(
            epoch_from_rfc3339("2025-06-26T13:00:00-05:00"),
            Some(1_750_960_800)
        );
    }

    #[test]
    fn epoch_from_rfc3339_handles_a_leap_day() {
        // 2024-02-29 exists (a leap year): pins the leap-day arithmetic.
        assert_eq!(
            epoch_from_rfc3339("2024-02-29T00:00:00Z"),
            Some(1_709_164_800)
        );
        // 2000-02-29 exists too (the 400-year century rule is a leap year), so a
        // valid leap day survives regardless of which leap sub-rule qualifies it.
        assert!(epoch_from_rfc3339("2000-02-29T00:00:00Z").is_some());
    }

    #[test]
    fn epoch_from_rfc3339_rejects_impossible_calendar_dates() {
        // Issue #177: the day was validated only as `1..=31`, so a day past the
        // month's real length silently normalized to a WRONG instant instead of
        // degrading to "reset unknown". A total parser rejects each of these.
        for bad in [
            "2026-02-30T00:00:00Z", // February never has 30 days
            "2025-02-29T00:00:00Z", // 2025 is not a leap year — no Feb 29
            "2100-02-29T00:00:00Z", // century non-leap (÷100 but not ÷400)
            "2026-04-31T00:00:00Z", // April has 30 days
            "2026-06-31T00:00:00Z", // June has 30 days
            "2026-00-15T00:00:00Z", // month 0 is not a real month
            "2026-01-00T00:00:00Z", // day 0 is not a real day
        ] {
            assert_eq!(epoch_from_rfc3339(bad), None, "{bad} must be rejected");
        }
    }

    #[test]
    fn epoch_from_rfc3339_does_not_panic_on_a_non_ascii_timezone() {
        // Issue #177 (follow-up): the `±HHMM` offset was byte-sliced (`tz[..2]`),
        // which panics when byte index 2 lands mid-char. A `from_utf8_lossy`-decoded
        // upstream `resets_at` can carry such a byte and must degrade to None, not
        // abort the daemon poll loop.
        assert_eq!(epoch_from_rfc3339("2026-06-26T00:00:00+1é2"), None);
        assert_eq!(epoch_from_rfc3339("2026-06-26T00:00:00+é0"), None);
    }

    #[test]
    fn epoch_from_rfc3339_does_not_panic_on_an_overflowing_year() {
        // Issue #177 (follow-up): an unbounded year overflowed the civil-day
        // arithmetic (`era * 146_097`) — a panic on the poll loop. A bounded parser
        // rejects an out-of-range year instead of overflowing.
        assert_eq!(
            epoch_from_rfc3339("99999999999999999-02-15T00:00:00Z"),
            None
        );
        assert_eq!(epoch_from_rfc3339("10000-01-01T00:00:00Z"), None);
    }

    #[test]
    fn epoch_from_resets_at_rejects_unparseable_input() {
        for bad in [
            "",
            "not-a-date",
            "2025-13-01T00:00:00Z", // month out of range
            "2025-06-26",           // date only, no time
            "2025-06-26T25:00:00Z", // hour out of range
        ] {
            assert_eq!(epoch_from_resets_at(bad), None, "{bad} should not parse");
        }
    }

    #[test]
    fn to_usage_normalizes_both_resets_to_epoch() {
        let report = UsageReport {
            session: 0.1,
            weekly: 0.2,
            weekly_all: 0.2,
            weekly_scoped: None,
            weekly_scoped_model: None,
            weekly_scoped_resets_at: None,
            weekly_all_resets_at: None,
            extra_usage: None,
            session_resets_at: Some("2025-01-01T00:00:00Z".to_owned()),
            weekly_resets_at: Some("2025-06-26T18:00:00Z".to_owned()),
            session_severity: None,
            weekly_severity: None,
        };
        let usage = report.to_usage();
        assert_eq!(usage.weekly_resets_at, Some(1_750_960_800));
        // Both windows' resets are projected for the `status` display (issue #72).
        assert_eq!(usage.session_resets_at, Some(1_735_689_600));
        // An absent reset (either window) projects to None (no fabricated value).
        let no_reset = UsageReport {
            session_resets_at: None,
            weekly_resets_at: None,
            ..report.clone()
        };
        let usage = no_reset.to_usage();
        assert_eq!(usage.weekly_resets_at, None);
        assert_eq!(usage.session_resets_at, None);
    }

    // --- access_token_from_blob (pure) ---

    #[test]
    fn extracts_the_nested_access_token() {
        let blob = br#"{"claudeAiOauth":{"accessToken":"sk-ant-oat-EXAMPLE","refreshToken":"sk-ant-ort-EXAMPLE"}}"#;
        let token = access_token_from_blob(blob).unwrap();
        assert_eq!(&*token, "sk-ant-oat-EXAMPLE");
    }

    #[test]
    fn tolerates_a_top_level_access_token() {
        assert_eq!(
            &*access_token_from_blob(br#"{"accessToken":"top-level"}"#).unwrap(),
            "top-level"
        );
        assert_eq!(
            &*access_token_from_blob(br#"{"access_token":"snake"}"#).unwrap(),
            "snake"
        );
    }

    #[test]
    fn a_blob_without_a_token_is_unreadable() {
        assert!(matches!(
            access_token_from_blob(br#"{"claudeAiOauth":{"refreshToken":"x"}}"#),
            Err(Error::UsageTokenUnreadable)
        ));
        assert!(matches!(
            access_token_from_blob(br#"{"claudeAiOauth":{"accessToken":""}}"#),
            Err(Error::UsageTokenUnreadable)
        ));
        assert!(matches!(
            access_token_from_blob(b"not json"),
            Err(Error::UsageTokenUnreadable)
        ));
    }

    // --- curl_config (pure, the request shape + token-on-stdin) ---

    #[test]
    fn curl_config_carries_the_url_headers_and_writeout() {
        let config = curl_config("sk-ant-oat-TESTTOKEN");
        assert!(config.contains(&format!("url = \"{USAGE_URL}\"")));
        assert!(config.contains("header = \"Authorization: Bearer sk-ant-oat-TESTTOKEN\""));
        assert!(config.contains(&format!("header = \"{BETA_HEADER}\"")));
        assert!(config.contains("write-out = \"\\n%{http_code}\\n%header{retry-after}\""));
        assert!(config.contains(&format!("max-time = {POLL_TIMEOUT_SECS}")));
    }

    // --- parse_curl_output (pure) ---

    #[test]
    fn parse_curl_output_splits_body_status_and_empty_retry_after() {
        // write-out shape: <body>\n<status>\n<retry-after> — the trailing
        // retry-after line is empty on a normal response.
        let resp = parse_curl_output(b"{\"limits\":[]}\n200\n");
        assert_eq!(resp.status, 200);
        assert_eq!(resp.body, "{\"limits\":[]}");
        assert_eq!(resp.retry_after, None);
    }

    #[test]
    fn parse_curl_output_handles_a_multiline_body() {
        // The two metadata lines are peeled from the end, so newlines inside the
        // body survive.
        let resp = parse_curl_output(b"{\n  \"a\": 1\n}\n403\n");
        assert_eq!(resp.status, 403);
        assert_eq!(resp.body, "{\n  \"a\": 1\n}");
        assert_eq!(resp.retry_after, None);
    }

    #[test]
    fn parse_curl_output_captures_a_numeric_retry_after() {
        // A 429 with `Retry-After: 30` and an empty body.
        let resp = parse_curl_output(b"\n429\n30");
        assert_eq!(resp.status, 429);
        assert_eq!(resp.body, "");
        assert_eq!(resp.retry_after, Some(Duration::from_secs(30)));
    }

    #[test]
    fn parse_curl_output_ignores_an_http_date_retry_after() {
        // The HTTP-date form is not honoured (delta-seconds only) → None, leaving
        // exponential back-off to govern the wait.
        let resp = parse_curl_output(b"\n503\nWed, 21 Oct 2015 07:28:00 GMT");
        assert_eq!(resp.status, 503);
        assert_eq!(resp.retry_after, None);
    }

    #[test]
    fn parse_curl_output_tolerates_a_missing_or_unparseable_status() {
        // No newline at all + non-numeric → status 0 (→ Transient).
        assert_eq!(parse_curl_output(b"garbage").status, 0);
        // A lone newline → empty status + empty retry-after → status 0.
        assert_eq!(parse_curl_output(b"\n").status, 0);
    }

    #[test]
    fn parse_retry_after_honours_only_the_delta_seconds_form() {
        assert_eq!(parse_retry_after("30"), Some(Duration::from_secs(30)));
        assert_eq!(parse_retry_after("  7 "), Some(Duration::from_secs(7)));
        assert_eq!(parse_retry_after("0"), Some(Duration::from_secs(0)));
        // Empty, negative, fractional, and HTTP-date forms all decline to None.
        assert_eq!(parse_retry_after(""), None);
        assert_eq!(parse_retry_after("-5"), None);
        assert_eq!(parse_retry_after("1.5"), None);
        assert_eq!(parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT"), None);
    }

    // --- RealUsageSource end-to-end (fake transport) ---

    /// Scripts a sequence of transport results, one per `fetch` call.
    struct FakeUsageTransport {
        responses: RefCell<VecDeque<Result<HttpResponse>>>,
    }

    impl FakeUsageTransport {
        fn new(responses: Vec<Result<HttpResponse>>) -> Self {
            Self {
                responses: RefCell::new(responses.into_iter().collect()),
            }
        }

        fn ok(status: u16, body: &str) -> Result<HttpResponse> {
            Ok(HttpResponse {
                status,
                body: body.to_owned(),
                retry_after: None,
            })
        }

        /// A bodyless response carrying a `Retry-After`, for the throttled paths.
        fn ok_retry_after(status: u16, retry_after: Option<Duration>) -> Result<HttpResponse> {
            Ok(HttpResponse {
                status,
                body: String::new(),
                retry_after,
            })
        }
    }

    impl UsageTransport for FakeUsageTransport {
        async fn fetch(&self) -> Result<HttpResponse> {
            self.responses
                .borrow_mut()
                .pop_front()
                .expect("a scripted response for each fetch")
        }
    }

    fn source(responses: Vec<Result<HttpResponse>>) -> RealUsageSource<FakeUsageTransport> {
        RealUsageSource::new(FakeUsageTransport::new(responses))
    }

    #[tokio::test]
    async fn a_success_yields_the_reading() {
        let body =
            r#"{"limits":[{"kind":"session","percent":30},{"kind":"weekly_all","percent":70}]}"#;
        let src = source(vec![FakeUsageTransport::ok(200, body)]);
        let reading = src.usage().await.unwrap();
        assert!((reading.usage.session - 0.30).abs() < 1e-9);
        assert!((reading.usage.weekly - 0.70).abs() < 1e-9);
        // No `severity` on either limit → the reading carries none (issue #156).
        assert_eq!(reading.severity, None);
    }

    #[tokio::test]
    async fn weekly_scoped_binding_false_reverts_to_the_all_models_weekly() {
        // Same fixture as `weekly_scoped_higher_than_weekly_all_wins`: Fable's
        // scoped cap (74 %) outvotes the all-models window (44 %) at parse time.
        let body = r#"{
            "five_hour": { "utilization": 0.0, "resets_at": null },
            "seven_day": { "utilization": 44.0, "resets_at": "2026-09-24T16:00:00.485164+00:00" },
            "limits": [
                { "kind": "session", "group": "session", "percent": 0, "severity": "normal",
                  "resets_at": null, "scope": null, "is_active": false },
                { "kind": "weekly_all", "group": "weekly", "percent": 44, "severity": "normal",
                  "resets_at": "2026-09-24T16:00:00.485164+00:00", "scope": null, "is_active": false },
                { "kind": "weekly_scoped", "group": "weekly", "percent": 74, "severity": "normal",
                  "resets_at": "2026-09-24T16:00:00.485335+00:00",
                  "scope": { "model": { "id": null, "display_name": "Fable" }, "surface": null },
                  "is_active": true }
            ]
        }"#;
        let all_reset = epoch_from_resets_at("2026-09-24T16:00:00.485164+00:00");
        let scoped_reset = epoch_from_resets_at("2026-09-24T16:00:00.485335+00:00");
        let src = source(vec![FakeUsageTransport::ok(200, body)]);
        let reading = src.usage().await.unwrap();
        // The pre-override all-models reset rides the reading either way.
        assert_eq!(reading.weekly_all_resets_at, all_reset);

        // `false`: the all-models window binds — `weekly`/`weekly_resets_at`
        // revert to the `seven_day` values; the scoped breakdown is untouched.
        let off = reading.clone().with_weekly_scoped_binding(false);
        assert!((off.usage.weekly - 0.44).abs() < 1e-9);
        assert_eq!(off.usage.weekly_resets_at, all_reset);
        assert_eq!(off.usage.weekly_scoped, Some(0.74));
        assert!((off.usage.weekly_all - 0.44).abs() < 1e-9);
        assert_eq!(off.weekly_all_resets_at, all_reset);

        // `true`: unchanged — the scoped cap stays the binding reading.
        let on = reading.with_weekly_scoped_binding(true);
        assert!((on.usage.weekly - 0.74).abs() < 1e-9);
        assert_eq!(on.usage.weekly_resets_at, scoped_reset);
        assert_eq!(on.weekly_all_resets_at, all_reset);
    }

    #[tokio::test]
    async fn a_200_with_an_unparseable_body_is_a_parse_error() {
        let src = source(vec![FakeUsageTransport::ok(200, "{}")]);
        assert!(matches!(src.usage().await, Err(Error::UsageParse(_))));
    }

    #[tokio::test]
    async fn forbidden_surfaces_distinctly_from_unauthorized() {
        // Issue #5 acceptance: a 403 (missing usage scope) surfaces as its own typed
        // error, never collapsed into the 401 the dead-credential detection counts.
        let src = source(vec![FakeUsageTransport::ok(403, "")]);
        assert!(matches!(src.usage().await, Err(Error::UsageScopeMissing)));
    }

    #[tokio::test]
    async fn rate_limited_and_server_errors_are_typed() {
        let src = source(vec![
            FakeUsageTransport::ok(429, ""),
            FakeUsageTransport::ok(503, ""),
            FakeUsageTransport::ok(400, ""),
        ]);
        assert!(matches!(
            src.usage().await,
            Err(Error::UsageRateLimited { status: 429, .. })
        ));
        assert!(matches!(
            src.usage().await,
            Err(Error::UsageTransient { status: 503, .. })
        ));
        assert!(matches!(
            src.usage().await,
            Err(Error::UsageRejected { status: 400 })
        ));
    }

    #[tokio::test]
    async fn a_retry_after_threads_into_the_throttled_errors() {
        // Issue #76: the server-advised Retry-After rides the rate-limited and
        // transient errors so the daemon can honour it as a minimum back-off.
        let src = source(vec![
            FakeUsageTransport::ok_retry_after(429, Some(Duration::from_secs(42))),
            FakeUsageTransport::ok_retry_after(503, Some(Duration::from_secs(7))),
            FakeUsageTransport::ok_retry_after(429, None),
        ]);
        assert!(matches!(
            src.usage().await,
            Err(Error::UsageRateLimited { retry_after: Some(d), .. }) if d == Duration::from_secs(42)
        ));
        assert!(matches!(
            src.usage().await,
            Err(Error::UsageTransient { retry_after: Some(d), .. }) if d == Duration::from_secs(7)
        ));
        assert!(matches!(
            src.usage().await,
            Err(Error::UsageRateLimited {
                retry_after: None,
                ..
            })
        ));
    }

    #[tokio::test]
    async fn unauthorized_and_a_transport_failure_each_surface_their_typed_error() {
        let src = source(vec![
            FakeUsageTransport::ok(401, ""),
            Err(Error::Io(std::io::Error::other("boom"))),
        ]);
        assert!(matches!(src.usage().await, Err(Error::UsageUnauthorized)));
        assert!(matches!(src.usage().await, Err(Error::Io(_))));
    }

    // --- no-telemetry / no-other-egress privacy invariant (issue #271) ---
    //
    // Sessiometer phones home for nothing: no analytics, usage telemetry, crash
    // reporting, update pings, or beacons. The ONLY outbound network egress it
    // originates is the documented per-account usage GET the swap logic polls
    // (`USAGE_URL`). These are *capability* guards — deliberately stronger than a
    // behavioural "we observed no egress on this run" check: the same discipline
    // as `scripts/check-no-security-framework.sh`, guarding what the binary CAN
    // do so a refactor cannot silently open a second egress. Because the guarantee
    // is about capability, it holds across every operating mode — start, poll,
    // swap, idle — and everything else: there is simply no code path that could
    // reach the network another way.
    //
    // The tool reaches the network through exactly one mechanism: spawning
    // `/usr/bin/curl` (`CURL`). It links no in-process HTTP/TLS client and opens
    // no raw TCP or UDP socket, so it *cannot* originate a connection except by
    // that one subprocess — which only ever targets `USAGE_URL`. (The daemon's
    // control channel is local on both targets and never leaves the machine: a
    // Unix-domain socket on Unix, and since #1511 a named pipe on Windows, created
    // with `reject_remote_clients` so it is unreachable over SMB. The external
    // `claude` CLI the daemon drives for refresh/login makes its own calls under
    // the user's own credential — that is not Sessiometer's egress.)

    /// Read a repo-root-relative file, anchored at `CARGO_MANIFEST_DIR` so the
    /// scan is tied to the crate under test regardless of the test CWD.
    fn repo_file(rel: &str) -> String {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(rel);
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
    }

    /// Every `.rs` file under `src/`, as `(path, contents)`.
    fn source_files() -> Vec<(std::path::PathBuf, String)> {
        fn walk(dir: &std::path::Path, out: &mut Vec<(std::path::PathBuf, String)>) {
            for entry in std::fs::read_dir(dir).expect("read_dir under src") {
                let path = entry.expect("dir entry").path();
                if path.is_dir() {
                    walk(&path, out);
                } else if path.extension().is_some_and(|ext| ext == "rs") {
                    let text = std::fs::read_to_string(&path).expect("read .rs source");
                    out.push((path, text));
                }
            }
        }
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut out = Vec::new();
        walk(&root, &mut out);
        assert!(!out.is_empty(), "found no source files under src/ to scan");
        out
    }

    #[test]
    fn the_sole_egress_endpoint_is_the_documented_usage_url() {
        // The one allowed destination, and the absolute (PATH-proof) binary that
        // reaches it. A change here is a deliberate change to the privacy contract
        // — keep it in step with the README no-telemetry statement.
        assert_eq!(USAGE_URL, "https://api.anthropic.com/api/oauth/usage");
        assert_eq!(CURL, "/usr/bin/curl");

        // The request we actually build points nowhere else: its only `url = ` line
        // is the documented endpoint.
        let config = curl_config("sk-ant-oat-TESTTOKEN");
        let url_lines: Vec<&str> = config
            .lines()
            .filter(|line| line.trim_start().starts_with("url = "))
            .collect();
        assert_eq!(url_lines.len(), 1, "exactly one request URL is configured");
        assert_eq!(url_lines[0], format!("url = \"{USAGE_URL}\"").as_str());
    }

    #[test]
    fn no_in_process_http_tls_or_telemetry_client_is_linked() {
        // Without an HTTP/TLS client crate the process cannot originate an
        // in-process HTTP(S) request; without a telemetry/analytics/crash SDK it
        // cannot beacon to one. The daemon's local control channel needs none of
        // these on either target — a Unix-domain socket, or a named pipe on Windows,
        // so the dependency graph stays empty of them — the network surface analogue
        // of `scripts/check-no-security-framework.sh` and the CONTRIBUTING.md
        // transport rule ("No TLS / HTTP client ... the one network call rides the
        // system curl"). Scans the committed lockfile, where every package appears
        // as a `name = "<crate>"` line.
        //
        // (`socket2` / `mio` — tokio's transitive socket layer, needed for the LOCAL
        // Unix socket — are intentionally NOT listed; raw-socket *usage* is guarded
        // by kind in the next test, not by banning the layer.)
        const FORBIDDEN_CRATES: &[&str] = &[
            // HTTP clients / stacks
            "reqwest",
            "hyper",
            "h2",
            "ureq",
            "isahc",
            "curl",
            "curl-sys",
            "surf",
            "attohttpc",
            "http-req",
            "minreq",
            // in-process TLS (only needed for in-process HTTPS; the curl subprocess needs none)
            "native-tls",
            "rustls",
            "openssl",
            "openssl-sys",
            "boring",
            "boring-sys",
            "schannel",
            // RPC clients
            "tonic",
            "grpcio",
            // telemetry / analytics / crash-reporting SaaS SDKs
            "sentry",
            "sentry-core",
            "opentelemetry",
            "tracing-opentelemetry",
            "posthog",
            "posthog-rs",
            "segment",
            "amplitude",
            "mixpanel",
            "rudderanalytics",
        ];
        let lockfile = repo_file("Cargo.lock");
        let has_crate = |name: &str| {
            lockfile
                .lines()
                .any(|line| line.trim() == format!("name = \"{name}\""))
        };

        // Positive control: the same predicate MUST find a crate we know is present,
        // so a passing test can never be a silently-broken matcher.
        assert!(
            has_crate("tokio"),
            "lockfile scan is broken: did not find tokio"
        );

        let present: Vec<&str> = FORBIDDEN_CRATES
            .iter()
            .copied()
            .filter(|name| has_crate(name))
            .collect();
        assert!(
            present.is_empty(),
            "forbidden network/telemetry crate(s) entered the dependency graph: {present:?}. \
             Sessiometer must reach the network only by spawning /usr/bin/curl for the usage \
             endpoint — no in-process HTTP/TLS client, no telemetry SDK (issue #271)."
        );
    }

    #[test]
    fn no_raw_tcp_or_udp_socket_primitive_is_used() {
        // tokio's `net` feature is on for the daemon's LOCAL IPC — but that IPC is a
        // Unix-domain socket (UnixStream / UnixListener) on Unix and a NAMED PIPE on
        // Windows since #1511, and neither ever leaves the machine. Every pipe
        // instance sets `reject_remote_clients`, so the Windows half is local-only
        // by construction rather than by convention. A raw TCP or UDP socket WOULD
        // be outbound egress that bypasses the curl seam, so guard the primitives
        // by name; neither the Unix-socket types nor the named-pipe types match.
        // Tokens are assembled from fragments so this guard never matches its own
        // source (usage.rs is itself scanned).
        let forbidden: [String; 4] = [
            concat!("Tcp", "Stream").to_string(),
            concat!("Tcp", "Listener").to_string(),
            concat!("Tcp", "Socket").to_string(),
            concat!("Udp", "Socket").to_string(),
        ];
        let files = source_files();

        // Positive control: the scanner must find a Unix-socket type we know is used,
        // proving a clean result is a real scan, not a scan that read nothing.
        assert!(
            files
                .iter()
                .any(|(_, text)| text.contains(concat!("Unix", "Listener"))),
            "source scan is broken: did not find UnixListener"
        );

        let mut offenders = Vec::new();
        for (path, text) in &files {
            for token in &forbidden {
                if text.contains(token.as_str()) {
                    offenders.push(format!("{}: {token}", path.display()));
                }
            }
        }
        assert!(
            offenders.is_empty(),
            "raw TCP/UDP socket primitive(s) found — outbound egress must ride the curl seam, \
             never a raw socket (issue #271): {offenders:?}"
        );
    }

    #[test]
    fn curl_is_the_only_network_capable_binary_referenced() {
        // The two guards above prove the tool cannot reach the network in-process,
        // so the only residual egress vector is spawning an external network tool.
        // The one sanctioned spawn is `/usr/bin/curl` -> USAGE_URL, in the usage
        // poller (this file). Guard the two realistic shapes of a sneaked-in beacon:
        //   (a) a curl binary literal in any OTHER module, and
        //   (b) another well-known network fetcher anywhere in the tree.
        // The other system tools the daemon drives — /usr/bin/security (keychain),
        // /bin/launchctl, /usr/bin/plutil, osascript — are not network-capable and
        // are not guarded here. The external `claude` CLI (isolated_spawn) is a
        // separate program's egress, resolved dynamically rather than as an absolute
        // literal. Fetcher tokens are assembled from fragments so this guard never
        // matches its own source.
        let other_fetchers: [String; 3] = [
            concat!("wg", "et").to_string(),
            concat!("so", "cat").to_string(),
            concat!("tel", "net").to_string(),
        ];
        let files = source_files();

        // Positive control: the sanctioned poller (skipped below) must itself
        // reference the curl binary — proving the scan actually reads file contents.
        assert!(
            files
                .iter()
                .any(|(path, text)| path.ends_with("usage.rs") && text.contains(CURL)),
            "source scan is broken: usage.rs does not reference the curl binary"
        );

        let mut offenders = Vec::new();
        for (path, text) in &files {
            for tool in &other_fetchers {
                if text.contains(tool.as_str()) {
                    offenders.push(format!("{}: references `{tool}`", path.display()));
                }
            }
            // A curl binary literal is sanctioned ONLY in the usage poller.
            if !path.ends_with("usage.rs") && text.contains(CURL) {
                offenders.push(format!(
                    "{}: references the curl binary `{CURL}`",
                    path.display()
                ));
            }
        }
        assert!(
            offenders.is_empty(),
            "an unsanctioned network CLI reference was found — the only sanctioned network spawn \
             is /usr/bin/curl in the usage poller (issue #271): {offenders:?}"
        );
    }
}
