//! One place to see what the agent has cost.
//!
//! Every thread already records its `cumulative_token_usage`, but that value
//! lives inside a zstd-compressed blob, so totalling a year of history the
//! obvious way means decompressing and parsing every thread on every open.
//! [`ThreadsDatabase`] keeps the four token counts and the model in plain
//! columns for exactly this view, so the cost here is one small query at open
//! and nothing at all when the reader switches ranges: the rows are folded in
//! memory.

use std::collections::HashMap;

use agent::ThreadUsageRow;
use agent_settings::AgentSettings;
use chrono::{Local, NaiveDate};
use gpui::{App, EventEmitter, FocusHandle, Focusable, Task};
use language_model::{LanguageModelCostInfo, LanguageModelRegistry, TokenUsage};
use settings::{ModelPricing, Settings as _};
use ui::{Table, prelude::*};
use util::ResultExt as _;
use workspace::{Item, Workspace};

use crate::OpenUsageDashboard;

pub fn init(cx: &mut App) {
    cx.observe_new(|workspace: &mut Workspace, _, _| {
        workspace.register_action(|workspace, _: &OpenUsageDashboard, window, cx| {
            let dashboard = cx.new(|cx| UsageDashboard::new(cx));
            workspace.add_item_to_active_pane(Box::new(dashboard), None, true, window, cx);
        });
    })
    .detach();
}

/// What a model charges, in dollars per single token.
///
/// Kept as four separate rates because cached and uncached input are billed
/// very differently; a single blended rate would misreport any agent that
/// leans on prompt caching, which is most of them.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Rate {
    input: f64,
    output: f64,
    cache_read: f64,
    cache_write: f64,
}

const PER_MILLION: f64 = 1_000_000.0;

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
    /// priced as full input here. That overstates cost wherever cache reads
    /// are discounted; a `model_pricing` entry corrects it.
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
            // A per-request multiplier is not a dollar figure, and pretending
            // otherwise would invent a number.
            LanguageModelCostInfo::RequestCost { .. } => None,
        }
    }

    /// `input_tokens` counts only tokens that were *not* served from cache:
    /// every provider Zed talks to reports the three classes separately, so
    /// the four terms below do not double count.
    fn cost(&self, usage: &TokenUsage) -> f64 {
        usage.input_tokens as f64 * self.input
            + usage.output_tokens as f64 * self.output
            + usage.cache_read_input_tokens as f64 * self.cache_read
            + usage.cache_creation_input_tokens as f64 * self.cache_write
    }

    /// What the cache reads would have cost at the full input rate, less what
    /// they did cost. Zero when the rate carries no cache discount.
    fn cache_savings(&self, usage: &TokenUsage) -> f64 {
        usage.cache_read_input_tokens as f64 * (self.input - self.cache_read)
    }
}

/// Rates for every model we can price, keyed by model id.
///
/// Provider-reported rates are inserted first so a user's `model_pricing`
/// entry always wins.
fn rate_table(cx: &App) -> HashMap<String, Rate> {
    let mut rates = HashMap::default();

    for model in LanguageModelRegistry::read_global(cx).available_models(cx) {
        if let Some(rate) = model
            .model_cost_info()
            .as_ref()
            .and_then(Rate::from_provider)
        {
            rates.insert(model.id().0.to_string(), rate);
        }
    }

    for (model_id, pricing) in &AgentSettings::get_global(cx).model_pricing {
        rates.insert(model_id.to_string(), Rate::from_settings(pricing));
    }

    rates
}

/// Everything spent on one model over the selected range.
#[derive(Clone, Debug, PartialEq)]
struct ModelSpend {
    provider_id: SharedString,
    model_id: SharedString,
    usage: TokenUsage,
    threads: usize,
    /// `None` when no rate is known. The row still reports its tokens: a model
    /// we cannot price is not a model that was free.
    cost: Option<f64>,
    cache_savings: f64,
}

#[derive(Clone, Debug, Default, PartialEq)]
struct Summary {
    rows: Vec<ModelSpend>,
    usage: TokenUsage,
    cost: f64,
    cache_savings: f64,
    threads: usize,
    /// Models in range that no rate covers, so `cost` is a floor rather than a
    /// total. Surfaced in the UI instead of being rounded away.
    unpriced_models: usize,
}

/// Folds per-thread rows into per-model spend.
///
/// Pure, and linear in the number of threads: the whole reason the range
/// buttons need no second query.
fn summarize(
    rows: &[ThreadUsageRow],
    since: Option<NaiveDate>,
    rates: &HashMap<String, Rate>,
) -> Summary {
    let mut by_model: HashMap<(&str, &str), ModelSpend> = HashMap::default();
    let mut summary = Summary::default();

    for row in rows {
        // Buckets follow the reader's own calendar, not UTC, so "today" means
        // what they expect it to mean.
        if let Some(since) = since
            && row.updated_at.with_timezone(&Local).date_naive() < since
        {
            continue;
        }

        let provider_id = row.provider_id.as_deref().unwrap_or("unknown");
        let model_id = row.model_id.as_deref().unwrap_or("unknown");
        let rate = rates.get(model_id);

        let entry = by_model
            .entry((provider_id, model_id))
            .or_insert_with(|| ModelSpend {
                provider_id: provider_id.to_string().into(),
                model_id: model_id.to_string().into(),
                usage: TokenUsage::default(),
                threads: 0,
                cost: rate.map(|_| 0.0),
                cache_savings: 0.0,
            });

        entry.usage = entry.usage + row.usage;
        entry.threads += 1;
        if let Some(rate) = rate {
            let cost = rate.cost(&row.usage);
            *entry.cost.get_or_insert(0.0) += cost;
            entry.cache_savings += rate.cache_savings(&row.usage);

            summary.cost += cost;
            summary.cache_savings += rate.cache_savings(&row.usage);
        }

        summary.usage = summary.usage + row.usage;
        summary.threads += 1;
    }

    summary.unpriced_models = by_model.values().filter(|row| row.cost.is_none()).count();
    summary.rows = by_model.into_values().collect();
    // Most expensive first, and unpriced rows last but ordered by volume, so
    // the table always leads with what matters.
    summary.rows.sort_by(|a, b| {
        b.cost
            .unwrap_or(-1.0)
            .total_cmp(&a.cost.unwrap_or(-1.0))
            .then_with(|| b.usage.total_tokens().cmp(&a.usage.total_tokens()))
    });

    summary
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Range {
    Today,
    Week,
    Month,
    All,
}

impl Range {
    const ALL: [Range; 4] = [Range::Today, Range::Week, Range::Month, Range::All];

    fn label(self) -> &'static str {
        match self {
            Range::Today => "Today",
            Range::Week => "7 days",
            Range::Month => "30 days",
            Range::All => "All time",
        }
    }

    /// Inclusive lower bound as a local date; `None` covers all history.
    fn since(self, today: NaiveDate) -> Option<NaiveDate> {
        let days = match self {
            Range::Today => 0,
            Range::Week => 6,
            Range::Month => 29,
            Range::All => return None,
        };
        Some(today - chrono::Duration::days(days))
    }
}

pub struct UsageDashboard {
    focus_handle: FocusHandle,
    /// All of history, loaded once. Ranges are sliced from this.
    rows: Vec<ThreadUsageRow>,
    range: Range,
    loading: bool,
    _load: Task<()>,
}

impl UsageDashboard {
    fn new(cx: &mut Context<Self>) -> Self {
        let usage = agent::thread_usage(cx);

        Self {
            focus_handle: cx.focus_handle(),
            rows: Vec::new(),
            range: Range::Week,
            loading: true,
            _load: cx.spawn(async move |this, cx| {
                let rows = usage.await.log_err().unwrap_or_default();
                this.update(cx, |this, cx| {
                    this.rows = rows;
                    this.loading = false;
                    cx.notify();
                })
                .ok();
            }),
        }
    }

    fn summary(&self, cx: &App) -> Summary {
        let since = self.range.since(Local::now().date_naive());
        summarize(&self.rows, since, &rate_table(cx))
    }
}

/// Dollars, with more precision the smaller the figure, so a cheap day does
/// not render as `$0.00`.
fn format_usd(amount: f64) -> String {
    if amount >= 100.0 {
        format!("${amount:.0}")
    } else if amount >= 1.0 {
        format!("${amount:.2}")
    } else {
        format!("${amount:.4}")
    }
}

fn format_tokens(tokens: u64) -> String {
    match tokens {
        0..=999 => tokens.to_string(),
        1_000..=999_999 => format!("{:.1}K", tokens as f64 / 1_000.0),
        _ => format!("{:.2}M", tokens as f64 / 1_000_000.0),
    }
}

impl UsageDashboard {
    fn render_stat(&self, label: &'static str, value: String, cx: &App) -> impl IntoElement {
        v_flex()
            .gap_1()
            .child(Label::new(label).size(LabelSize::Small).color(Color::Muted))
            .child(
                Label::new(value)
                    .size(LabelSize::Large)
                    .color(Color::Default),
            )
            .text_color(cx.theme().colors().text)
    }
}

impl Render for UsageDashboard {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let summary = self.summary(cx);
        let range = self.range;

        let mut table = Table::new(6).striped().header(vec![
            "Model".into_any_element(),
            "Provider".into_any_element(),
            "Threads".into_any_element(),
            "Input".into_any_element(),
            "Output".into_any_element(),
            "Cost".into_any_element(),
        ]);

        for row in &summary.rows {
            let cost = match row.cost {
                Some(cost) => Label::new(format_usd(cost)).into_any_element(),
                None => Label::new("No rate").color(Color::Muted).into_any_element(),
            };

            table = table.row(vec![
                Label::new(row.model_id.clone()).into_any_element(),
                Label::new(row.provider_id.clone())
                    .color(Color::Muted)
                    .into_any_element(),
                Label::new(row.threads.to_string()).into_any_element(),
                Label::new(format_tokens(
                    row.usage.input_tokens
                        + row.usage.cache_read_input_tokens
                        + row.usage.cache_creation_input_tokens,
                ))
                .into_any_element(),
                Label::new(format_tokens(row.usage.output_tokens)).into_any_element(),
                cost,
            ]);
        }

        v_flex()
            .id("usage-dashboard")
            .track_focus(&self.focus_handle)
            .size_full()
            .overflow_scroll()
            .p_4()
            .gap_4()
            .bg(cx.theme().colors().editor_background)
            .child(
                h_flex()
                    .justify_between()
                    .child(Headline::new("Agent Usage").size(HeadlineSize::Medium))
                    .child(h_flex().gap_1().children(Range::ALL.map(|option| {
                        Button::new(option.label(), option.label())
                            .toggle_state(option == range)
                            .on_click(cx.listener(move |this, _, _, cx| {
                                this.range = option;
                                cx.notify();
                            }))
                    }))),
            )
            .child(
                h_flex()
                    .gap_6()
                    .child(self.render_stat("Spend", format_usd(summary.cost), cx))
                    .child(self.render_stat(
                        "Tokens",
                        format_tokens(summary.usage.total_tokens()),
                        cx,
                    ))
                    .child(self.render_stat(
                        "Saved by caching",
                        format_usd(summary.cache_savings),
                        cx,
                    ))
                    .child(self.render_stat("Threads", summary.threads.to_string(), cx)),
            )
            .when(summary.unpriced_models > 0, |this| {
                this.child(
                    Label::new(format!(
                        "{} model(s) have no known price, so spend is a lower bound. \
                         Set `agent.model_pricing` to include them.",
                        summary.unpriced_models
                    ))
                    .size(LabelSize::Small)
                    .color(Color::Warning),
                )
            })
            .child(if self.loading {
                Label::new("Reading thread history…")
                    .color(Color::Muted)
                    .into_any_element()
            } else if summary.rows.is_empty() {
                Label::new("No agent usage recorded in this range.")
                    .color(Color::Muted)
                    .into_any_element()
            } else {
                table.into_any_element()
            })
    }
}

impl EventEmitter<()> for UsageDashboard {}

impl Focusable for UsageDashboard {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl Item for UsageDashboard {
    type Event = ();

    fn to_item_events(_: &Self::Event, _: &mut dyn FnMut(workspace::item::ItemEvent)) {}

    fn tab_content_text(&self, _detail: usize, _cx: &App) -> SharedString {
        "Agent Usage".into()
    }

    fn tab_tooltip_text(&self, _cx: &App) -> Option<SharedString> {
        Some("Token spend across all agent threads".into())
    }

    fn telemetry_event_text(&self) -> Option<&'static str> {
        None
    }

    fn can_split(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, Utc};

    fn rate() -> Rate {
        Rate {
            input: 3.0 / PER_MILLION,
            output: 15.0 / PER_MILLION,
            cache_read: 0.3 / PER_MILLION,
            cache_write: 3.75 / PER_MILLION,
        }
    }

    fn row(model: &str, days_ago: i64, usage: TokenUsage) -> ThreadUsageRow {
        ThreadUsageRow {
            updated_at: Utc::now() - Duration::days(days_ago),
            provider_id: Some("anthropic".into()),
            model_id: Some(model.into()),
            usage,
        }
    }

    fn usage(input: u64, output: u64, cache_read: u64, cache_write: u64) -> TokenUsage {
        TokenUsage {
            input_tokens: input,
            output_tokens: output,
            cache_read_input_tokens: cache_read,
            cache_creation_input_tokens: cache_write,
        }
    }

    #[test]
    fn prices_each_token_class_at_its_own_rate() {
        // 1M uncached in, 1M out, 1M cache read, 1M cache write.
        let cost = rate().cost(&usage(1_000_000, 1_000_000, 1_000_000, 1_000_000));
        assert!((cost - (3.0 + 15.0 + 0.3 + 3.75)).abs() < 1e-9);
    }

    #[test]
    fn cache_savings_measure_the_discount_not_the_spend() {
        let savings = rate().cache_savings(&usage(0, 0, 1_000_000, 0));
        assert!((savings - 2.7).abs() < 1e-9);
    }

    #[test]
    fn settings_rates_default_cache_to_the_input_rate() {
        let pricing = ModelPricing {
            input: 2.0,
            output: 8.0,
            cache_read: None,
            cache_write: None,
        };
        let rate = Rate::from_settings(&pricing);
        assert_eq!(rate.cache_read, rate.input);
        assert_eq!(rate.cache_write, rate.input);
    }

    #[test]
    fn a_request_multiplier_is_not_a_price() {
        assert_eq!(
            Rate::from_provider(&LanguageModelCostInfo::RequestCost {
                cost_per_request: 1.0
            }),
            None
        );
    }

    #[test]
    fn groups_by_model_and_sums_usage() {
        let rates = HashMap::from_iter([("opus".to_string(), rate())]);
        let rows = vec![
            row("opus", 0, usage(100, 10, 0, 0)),
            row("opus", 0, usage(200, 20, 0, 0)),
        ];

        let summary = summarize(&rows, None, &rates);
        assert_eq!(summary.rows.len(), 1);
        assert_eq!(summary.rows[0].threads, 2);
        assert_eq!(summary.rows[0].usage.input_tokens, 300);
        assert_eq!(summary.threads, 2);
    }

    #[test]
    fn excludes_threads_older_than_the_range() {
        let rates = HashMap::from_iter([("opus".to_string(), rate())]);
        let rows = vec![
            row("opus", 0, usage(100, 0, 0, 0)),
            row("opus", 40, usage(900, 0, 0, 0)),
        ];

        let since = Range::Month.since(Local::now().date_naive());
        let summary = summarize(&rows, since, &rates);
        assert_eq!(summary.usage.input_tokens, 100);
        assert_eq!(summary.threads, 1);
    }

    #[test]
    fn unpriced_models_keep_their_tokens_and_are_counted() {
        let rows = vec![row("mystery-model", 0, usage(1_000, 500, 0, 0))];

        let summary = summarize(&rows, None, &HashMap::default());
        assert_eq!(summary.unpriced_models, 1);
        assert_eq!(summary.cost, 0.0);
        assert_eq!(summary.rows[0].cost, None);
        assert_eq!(summary.rows[0].usage.output_tokens, 500);
    }

    #[test]
    fn priced_models_sort_above_unpriced_ones() {
        let rates = HashMap::from_iter([("opus".to_string(), rate())]);
        let rows = vec![
            row("mystery-model", 0, usage(10_000_000, 0, 0, 0)),
            row("opus", 0, usage(1_000, 0, 0, 0)),
        ];

        let summary = summarize(&rows, None, &rates);
        assert_eq!(summary.rows[0].model_id.as_ref(), "opus");
    }
}
