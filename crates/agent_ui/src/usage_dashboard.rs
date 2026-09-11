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
use chrono::{Local, NaiveDate};
use gpui::{App, EventEmitter, FocusHandle, Focusable, Subscription, Task};
use language_model::TokenUsage;
use settings::SettingsStore;
use ui::{Table, prelude::*};
use util::ResultExt as _;
use workspace::{Item, Workspace};

use crate::OpenUsageDashboard;
use crate::model_rate_table::{self, Rate};

pub fn init(cx: &mut App) {
    cx.observe_new(|workspace: &mut Workspace, _, _| {
        workspace.register_action(|workspace, _: &OpenUsageDashboard, window, cx| {
            let dashboard = cx.new(|cx| UsageDashboard::new(cx));
            workspace.add_item_to_active_pane(Box::new(dashboard), None, true, window, cx);
        });
    })
    .detach();
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
///
/// A thread is attributed entirely to the day it was last active, because
/// that is the granularity the denormalized columns record. Per-model totals
/// are therefore exact, but a thread worked on across a range boundary counts
/// wholly on its last day. Splitting spend by day properly would mean reading
/// per-message usage out of the compressed blobs, which is the cost this whole
/// design exists to avoid.
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
        let rate = model_rate_table::rate_for(rates, model_id);

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
        if let Some(rate) = &rate {
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
    /// Recomputed only when the rows, the range, or the prices change.
    ///
    /// Resolving rates means merging the fetched price table, which has on the
    /// order of a thousand entries; doing that per frame would make scrolling
    /// the table cost more than loading it.
    summary: Summary,
    loading: bool,
    _load: Task<()>,
    _settings: Subscription,
}

impl UsageDashboard {
    fn new(cx: &mut Context<Self>) -> Self {
        let usage = agent::thread_usage(cx);
        // Kicked off alongside the query so a first-run price fetch overlaps
        // with reading history rather than following it.
        let prices = model_rate_table::refresh(cx);

        Self {
            focus_handle: cx.focus_handle(),
            rows: Vec::new(),
            range: Range::Week,
            summary: Summary::default(),
            loading: true,
            _settings: cx.observe_global::<SettingsStore>(|this, cx| {
                this.recompute(cx);
                cx.notify();
            }),
            _load: cx.spawn(async move |this, cx| {
                let rows = usage.await.log_err().unwrap_or_default();
                this.update(cx, |this, cx| {
                    this.rows = rows;
                    this.loading = false;
                    this.recompute(cx);
                    cx.notify();
                })
                .ok();

                // Token counts are shown as soon as they are read. Prices may
                // still be in flight on a first run, so fold them in when they
                // land rather than making history wait on a network request.
                prices.await;
                this.update(cx, |this, cx| {
                    this.recompute(cx);
                    cx.notify();
                })
                .ok();
            }),
        }
    }

    /// Rebuilds the totals. Called when the rows, the range, or the prices
    /// change — never from `render`.
    fn recompute(&mut self, cx: &App) {
        let since = self.range.since(Local::now().date_naive());
        self.summary = summarize(&self.rows, since, &model_rate_table::resolve(cx));
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
        let summary = &self.summary;
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
                    .child(
                        v_flex()
                            .child(Headline::new("Agent Usage").size(HeadlineSize::Medium))
                            .child(
                                Label::new("Grouped by each thread's last activity")
                                    .size(LabelSize::Small)
                                    .color(Color::Muted),
                            ),
                    )
                    .child(h_flex().gap_1().children(Range::ALL.map(|option| {
                        Button::new(option.label(), option.label())
                            .toggle_state(option == range)
                            .on_click(cx.listener(move |this, _, _, cx| {
                                this.range = option;
                                this.recompute(cx);
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
            input: 3.0 / 1_000_000.0,
            output: 15.0 / 1_000_000.0,
            cache_read: 0.3 / 1_000_000.0,
            cache_write: 3.75 / 1_000_000.0,
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
    fn groups_by_model_and_sums_usage() {
        let rates = HashMap::from_iter([("claude-opus-5".to_string(), rate())]);
        let rows = vec![
            row("claude-opus-5", 0, usage(100, 10, 0, 0)),
            row("claude-opus-5", 0, usage(200, 20, 0, 0)),
        ];

        let summary = summarize(&rows, None, &rates);
        assert_eq!(summary.rows.len(), 1);
        assert_eq!(summary.rows[0].threads, 2);
        assert_eq!(summary.rows[0].usage.input_tokens, 300);
        assert_eq!(summary.threads, 2);
    }

    #[test]
    fn excludes_threads_older_than_the_range() {
        let rates = HashMap::from_iter([("claude-opus-5".to_string(), rate())]);
        let rows = vec![
            row("claude-opus-5", 0, usage(100, 0, 0, 0)),
            row("claude-opus-5", 40, usage(900, 0, 0, 0)),
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
        let rates = HashMap::from_iter([("claude-opus-5".to_string(), rate())]);
        let rows = vec![
            row("mystery-model", 0, usage(10_000_000, 0, 0, 0)),
            row("claude-opus-5", 0, usage(1_000, 0, 0, 0)),
        ];

        let summary = summarize(&rows, None, &rates);
        assert_eq!(summary.rows[0].model_id.as_ref(), "claude-opus-5");
    }
}
