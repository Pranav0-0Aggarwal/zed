//! What each model charges, and where that number comes from.
//!
//! Three sources, in descending authority: the user's own `agent.model_pricing`
//! entry, the price a provider reports for its own model, and finally LiteLLM's
//! public rate table, fetched over the network. A model that none of them cover
//! is reported as unpriced rather than as free.
//!
//! The fetched table is the only reason this module needs I/O, and it is
//! treated as hostile input throughout: the response is size-capped and
//! deadlined, only four float fields per entry are ever read, and the on-disk
//! cache is re-validated on load rather than trusted for being local.

use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use agent_settings::AgentSettings;
use anyhow::{Context as _, Result, anyhow};
use futures::{AsyncReadExt as _, FutureExt as _, future::Shared, select_biased};
use gpui::{App, AppContext as _, BackgroundExecutor, Global, Task};
use http_client::{AsyncBody, HttpClient};
use language_model::{LanguageModelCostInfo, LanguageModelRegistry};
use settings::{ModelPricing, Settings as _};
use util::ResultExt as _;

/// LiteLLM's price table: a community-maintained document mapping model ids to
/// per-token costs.
///
/// Pinned to a branch rather than a release, because that is the only form
/// LiteLLM publishes. It is therefore mutable by a third party, which is why
/// nothing here trusts its shape and why `agent.fetch_model_prices` exists to
/// turn the fetch off entirely.
const RATES_URL: &str =
    "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json";

/// Refetch at most once a day. Prices move on the order of months.
const TTL: Duration = Duration::from_secs(24 * 60 * 60);

/// The real document is a few megabytes. The cap is generous but bounded, so a
/// runaway or hostile response cannot exhaust memory.
const MAX_RESPONSE_BYTES: u64 = 16 * 1024 * 1024;

/// Long enough for a slow connection, short enough that a hung request does not
/// leave the dashboard waiting forever.
const FETCH_TIMEOUT: Duration = Duration::from_secs(20);

const PER_MILLION: f64 = 1_000_000.0;

/// What a model charges, in dollars per single token.
///
/// Four rates rather than one, because cached and uncached input are billed
/// very differently; a blended rate would misreport any agent that leans on
/// prompt caching, which is most of them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Rate {
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_write: f64,
}

impl Rate {
    fn from_settings(pricing: &ModelPricing) -> Self {
        let input = f64::from(pricing.input) / PER_MILLION;
        Self {
            input,
            output: f64::from(pricing.output) / PER_MILLION,
            // An unspecified cache rate bills cached tokens as plain input
            // rather than as free, which is the safer direction to be wrong in.
            cache_read: pricing
                .cache_read
                .map_or(input, |rate| f64::from(rate) / PER_MILLION),
            cache_write: pricing
                .cache_write
                .map_or(input, |rate| f64::from(rate) / PER_MILLION),
        }
    }

    /// Providers report one input and one output rate, so cache reads are
    /// priced as full input here. That overstates cost wherever cache reads are
    /// discounted; a `model_pricing` entry or the fetched table corrects it.
    fn from_provider(info: &LanguageModelCostInfo) -> Option<Self> {
        match info {
            LanguageModelCostInfo::TokenCost {
                input_token_cost_per_1m,
                output_token_cost_per_1m,
            } => {
                let input = input_token_cost_per_1m / PER_MILLION;
                Some(Self {
                    input,
                    output: output_token_cost_per_1m / PER_MILLION,
                    cache_read: input,
                    cache_write: input,
                })
            }
            // A per-request multiplier is not a dollar figure, and converting
            // one would be inventing a number.
            LanguageModelCostInfo::RequestCost { .. } => None,
        }
    }

    /// `input_tokens` counts only tokens that were *not* served from cache:
    /// every provider Zed talks to reports the three input classes separately,
    /// so these four terms do not double count.
    pub fn cost(&self, usage: &language_model::TokenUsage) -> f64 {
        usage.input_tokens as f64 * self.input
            + usage.output_tokens as f64 * self.output
            + usage.cache_read_input_tokens as f64 * self.cache_read
            + usage.cache_creation_input_tokens as f64 * self.cache_write
    }

    /// What the cache reads would have cost at the full input rate, less what
    /// they did cost. Zero when the rate carries no cache discount.
    pub fn cache_savings(&self, usage: &language_model::TokenUsage) -> f64 {
        usage.cache_read_input_tokens as f64 * (self.input - self.cache_read)
    }
}

/// One LiteLLM entry, narrowed to the fields that carry a price. Every other
/// field in the document is ignored.
#[derive(Debug, serde::Deserialize)]
struct LiteLlmEntry {
    input_cost_per_token: Option<f64>,
    output_cost_per_token: Option<f64>,
    cache_read_input_token_cost: Option<f64>,
    cache_creation_input_token_cost: Option<f64>,
}

/// A price is only usable if it is finite and not negative. Anything else is
/// treated as absent.
fn usable(cost: Option<f64>) -> Option<f64> {
    cost.filter(|cost| cost.is_finite() && *cost >= 0.0)
}

fn normalize_key(model: &str) -> String {
    model.trim().to_lowercase()
}

/// The name after the last `/`, so `anthropic/claude-x` also answers to
/// `claude-x`.
fn bare_name(key: &str) -> &str {
    key.rsplit('/').next().unwrap_or(key)
}

/// Drops a bracketed variant suffix such as `claude-fable-5-1[1m]`, which marks
/// a context tier the table does not price separately.
fn strip_variant(key: &str) -> &str {
    key.split_once('[').map_or(key, |(base, _)| base)
}

/// Names that are never priced, whatever the table claims.
///
/// A bare family name is genuinely ambiguous across generations, and pricing it
/// would silently attribute one generation's cost to another.
const UNPRICEABLE: &[&str] = &[
    "opus",
    "sonnet",
    "haiku",
    "fable",
    "synthetic",
    "<synthetic>",
];

/// Projects the LiteLLM document into a rate table.
///
/// Entries without *both* an input and an output price are dropped: a
/// half-priced model under-reports cost, which is worse than reporting the
/// model as unpriced. A bare name is aliased only when no canonical entry
/// claims it and every qualified entry agrees on the rate, so an ambiguous
/// short name never resolves to one arbitrary variant's price.
pub(crate) fn parse_rate_table(document: &serde_json::Value) -> HashMap<String, Rate> {
    let mut table = HashMap::default();

    let Some(entries) = document.as_object() else {
        return table;
    };

    for (name, raw) in entries {
        let Ok(entry) = serde_json::from_value::<LiteLlmEntry>(raw.clone()) else {
            continue;
        };
        let (Some(input), Some(output)) = (
            usable(entry.input_cost_per_token),
            usable(entry.output_cost_per_token),
        ) else {
            continue;
        };

        let key = normalize_key(name);
        if key.is_empty() {
            continue;
        }

        table.insert(
            key,
            Rate {
                input,
                output,
                cache_read: usable(entry.cache_read_input_token_cost).unwrap_or(input),
                cache_write: usable(entry.cache_creation_input_token_cost).unwrap_or(input),
            },
        );
    }

    // `None` marks a bare name claimed at conflicting rates: it gets no alias.
    let mut aliases: HashMap<String, Option<Rate>> = HashMap::default();
    for (key, rate) in &table {
        let alias = bare_name(key);
        if alias.is_empty() || alias == key || table.contains_key(alias) {
            continue;
        }
        match aliases.get(alias) {
            None => {
                aliases.insert(alias.to_string(), Some(*rate));
            }
            Some(Some(held)) if held != rate => {
                aliases.insert(alias.to_string(), None);
            }
            _ => {}
        }
    }
    table.extend(
        aliases
            .into_iter()
            .filter_map(|(alias, rate)| rate.map(|rate| (alias, rate))),
    );

    table
}

fn lookup(table: &HashMap<String, Rate>, model_id: &str) -> Option<Rate> {
    let key = normalize_key(strip_variant(model_id));
    if key.is_empty() || UNPRICEABLE.contains(&bare_name(&key)) {
        return None;
    }
    table.get(&key).copied()
}

/// The fetched table plus enough state to honour the TTL and to keep concurrent
/// refreshes down to one request.
#[derive(Default)]
struct GlobalModelRates {
    table: Arc<HashMap<String, Rate>>,
    fetched_at: Option<SystemTime>,
    in_flight: Option<Shared<Task<()>>>,
}

impl Global for GlobalModelRates {}

fn cache_path() -> std::path::PathBuf {
    paths::data_dir().join("model_prices.json")
}

#[derive(serde::Serialize, serde::Deserialize)]
struct CachedRates {
    fetched_at_secs: u64,
    /// The document exactly as served. Kept raw, and re-parsed through
    /// [`parse_rate_table`] on load, so a tampered cache file gets the same
    /// validation as a fresh response instead of being trusted for being local.
    document: serde_json::Value,
}

/// Persists the document so the next launch can skip the request.
fn write_cache(cached: &CachedRates) -> Result<()> {
    let path = cache_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(&path, serde_json::to_vec(cached)?).context("writing model price cache")
}

async fn fetch_document(
    http_client: Arc<dyn HttpClient>,
    executor: BackgroundExecutor,
) -> Result<serde_json::Value> {
    let request = async {
        let mut response = http_client
            .get(RATES_URL, AsyncBody::default(), true)
            .await?;
        if !response.status().is_success() {
            return Err(anyhow!("price table returned {}", response.status()));
        }

        // Cap the read rather than trusting `Content-Length`: a truncated or
        // deliberately huge body must not be able to grow without bound.
        let mut body = Vec::new();
        response
            .body_mut()
            .take(MAX_RESPONSE_BYTES + 1)
            .read_to_end(&mut body)
            .await
            .context("reading price table")?;
        if body.len() as u64 > MAX_RESPONSE_BYTES {
            return Err(anyhow!("price table exceeded {MAX_RESPONSE_BYTES} bytes"));
        }

        serde_json::from_slice(&body).context("parsing price table")
    };

    select_biased! {
        result = request.fuse() => result,
        _ = executor.timer(FETCH_TIMEOUT).fuse() => Err(anyhow!("price table request timed out")),
    }
}

fn seconds_since_epoch(time: SystemTime) -> u64 {
    time.duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// The price document and when it was obtained: from disk while that copy is
/// inside the TTL, from the network otherwise.
///
/// Runs entirely on a background thread — it does blocking file I/O — and
/// falls back to a stale cached document when the network fails, because a
/// stale price beats no price at all.
async fn load_document(
    http_client: Arc<dyn HttpClient>,
    executor: BackgroundExecutor,
) -> Option<(serde_json::Value, u64)> {
    let now = seconds_since_epoch(SystemTime::now());
    let cached = std::fs::read(cache_path())
        .ok()
        .and_then(|bytes| serde_json::from_slice::<CachedRates>(&bytes).ok());

    if let Some(cached) = &cached
        && now.saturating_sub(cached.fetched_at_secs) < TTL.as_secs()
    {
        return Some((cached.document.clone(), cached.fetched_at_secs));
    }

    match fetch_document(http_client, executor).await.log_err() {
        Some(document) => {
            write_cache(&CachedRates {
                fetched_at_secs: now,
                document: document.clone(),
            })
            .log_err();
            Some((document, now))
        }
        None => cached.map(|cached| (cached.document, cached.fetched_at_secs)),
    }
}

/// Makes the fetched table available, returning a task that completes once it
/// is (or once the attempt has failed).
///
/// Concurrent callers share one task, so opening several dashboards issues one
/// request. A table younger than [`TTL`] is left alone.
pub(crate) fn refresh(cx: &mut App) -> Shared<Task<()>> {
    if !AgentSettings::get_global(cx).fetch_model_prices {
        return Task::ready(()).shared();
    }

    let global = cx.default_global::<GlobalModelRates>();
    if let Some(in_flight) = global.in_flight.clone() {
        return in_flight;
    }
    if global
        .fetched_at
        .is_some_and(|at| at.elapsed().unwrap_or(TTL) < TTL)
    {
        return Task::ready(()).shared();
    }

    let http_client = cx.http_client();
    let executor = cx.background_executor().clone();
    let load = cx.background_spawn({
        let executor = executor.clone();
        async move { load_document(http_client, executor).await }
    });

    let task = cx
        .spawn(async move |cx| {
            let loaded = load.await;
            cx.update(|cx| {
                let global = cx.default_global::<GlobalModelRates>();
                global.in_flight = None;

                let Some((document, fetched_at_secs)) = loaded else {
                    return;
                };
                let table = parse_rate_table(&document);
                // An empty parse means the document changed shape. Keep what
                // we had rather than replacing prices with nothing.
                if !table.is_empty() {
                    global.table = Arc::new(table);
                    global.fetched_at = Some(UNIX_EPOCH + Duration::from_secs(fetched_at_secs));
                }
            });
        })
        .shared();

    cx.default_global::<GlobalModelRates>().in_flight = Some(task.clone());
    task
}

/// Resolves a rate for every model that can be priced right now.
///
/// Precedence is user setting, then the provider's own reported price, then the
/// fetched table. Inserting in reverse of that order lets each later source
/// simply overwrite the earlier one.
pub(crate) fn resolve(cx: &App) -> HashMap<String, Rate> {
    let mut rates = if cx.has_global::<GlobalModelRates>() {
        (*cx.global::<GlobalModelRates>().table).clone()
    } else {
        HashMap::default()
    };

    for model in LanguageModelRegistry::read_global(cx).available_models(cx) {
        if let Some(rate) = model
            .model_cost_info()
            .as_ref()
            .and_then(Rate::from_provider)
        {
            rates.insert(normalize_key(&model.id().0), rate);
        }
    }

    for (model_id, pricing) in &AgentSettings::get_global(cx).model_pricing {
        rates.insert(normalize_key(model_id), Rate::from_settings(pricing));
    }

    rates
}

/// Looks a model up the same way regardless of which source supplied the rate,
/// so variant suffixes and provider prefixes resolve consistently.
pub(crate) fn rate_for(rates: &HashMap<String, Rate>, model_id: &str) -> Option<Rate> {
    lookup(rates, model_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn drops_entries_missing_either_side_of_the_price() {
        let table = parse_rate_table(&json!({
            "priced": { "input_cost_per_token": 1e-6, "output_cost_per_token": 2e-6 },
            "input-only": { "input_cost_per_token": 1e-6 },
            "output-only": { "output_cost_per_token": 2e-6 },
            "neither": { "max_tokens": 4096 },
        }));

        assert!(table.contains_key("priced"));
        assert_eq!(table.len(), 1);
    }

    #[test]
    fn rejects_prices_that_are_not_real_numbers() {
        let table = parse_rate_table(&json!({
            "negative": { "input_cost_per_token": -1.0, "output_cost_per_token": 2e-6 },
            "stringly": { "input_cost_per_token": "1e-6", "output_cost_per_token": 2e-6 },
        }));

        assert!(table.is_empty());
    }

    #[test]
    fn cached_input_falls_back_to_the_input_price() {
        let table = parse_rate_table(&json!({
            "m": { "input_cost_per_token": 3e-6, "output_cost_per_token": 15e-6 },
        }));

        let rate = table.get("m").unwrap();
        assert_eq!(rate.cache_read, 3e-6);
        assert_eq!(rate.cache_write, 3e-6);
    }

    #[test]
    fn aliases_a_bare_name_only_when_variants_agree() {
        let agreeing = parse_rate_table(&json!({
            "vertex/same": { "input_cost_per_token": 1e-6, "output_cost_per_token": 2e-6 },
            "bedrock/same": { "input_cost_per_token": 1e-6, "output_cost_per_token": 2e-6 },
        }));
        assert!(agreeing.contains_key("same"));

        let conflicting = parse_rate_table(&json!({
            "vertex/diff": { "input_cost_per_token": 1e-6, "output_cost_per_token": 2e-6 },
            "bedrock/diff": { "input_cost_per_token": 9e-6, "output_cost_per_token": 2e-6 },
        }));
        assert!(!conflicting.contains_key("diff"));
    }

    #[test]
    fn a_canonical_entry_is_never_overwritten_by_an_alias() {
        let table = parse_rate_table(&json!({
            "claude": { "input_cost_per_token": 1e-6, "output_cost_per_token": 1e-6 },
            "anthropic/claude": { "input_cost_per_token": 5e-6, "output_cost_per_token": 5e-6 },
        }));

        assert_eq!(table.get("claude").unwrap().input, 1e-6);
    }

    #[test]
    fn lookup_strips_variant_suffixes_and_is_case_insensitive() {
        let table = parse_rate_table(&json!({
            "claude-x": { "input_cost_per_token": 3e-6, "output_cost_per_token": 15e-6 },
        }));

        assert!(lookup(&table, "Claude-X[1m]").is_some());
        assert!(lookup(&table, "claude-x").is_some());
    }

    #[test]
    fn ambiguous_family_names_stay_unpriced() {
        let table = parse_rate_table(&json!({
            "opus": { "input_cost_per_token": 1e-6, "output_cost_per_token": 1e-6 },
        }));

        assert_eq!(lookup(&table, "opus"), None);
        assert_eq!(lookup(&table, "<synthetic>"), None);
    }

    #[test]
    fn a_non_object_document_yields_no_rates() {
        assert!(parse_rate_table(&json!(["not", "an", "object"])).is_empty());
        assert!(parse_rate_table(&json!(null)).is_empty());
    }
}
