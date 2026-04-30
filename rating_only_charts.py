from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from scipy import stats
import json


HERE = Path(__file__).resolve().parent

POS_BY_YEAR_RATING_CSV = HERE / "pos_counts_by_year_and_rating.csv"
WORDS_BY_YEAR_RATING_CSV = HERE / "word_count_by_year_and_rating.csv"

RATING_TICKS = [1, 2, 3, 4, 5]
PLOT_FONT = dict(family="Roboto")
LINE_COLOR = "#1d2c4d"

TITLE_SIZE = 26
FOOTNOTE_SIZE = 11
TITLE_X = 0.03
LEGEND_Y = 1.06

OUT_ADJ_LINE = HERE / "adj_count_by_rating_line.html"
OUT_ADJ_BAR = HERE / "adj_count_by_rating_bar.html"
OUT_ADV_LINE = HERE / "adv_count_by_rating_line.html"
OUT_ADV_BAR = HERE / "adv_count_by_rating_bar.html"
OUT_WORD_LINE = HERE / "word_count_by_rating_line.html"
OUT_WORD_BAR = HERE / "word_count_by_rating_bar.html"

OUT_ADJ_BAR_LINE = HERE / "adj_count_by_rating_bar_line.html"
OUT_ADV_BAR_LINE = HERE / "adv_count_by_rating_bar_line.html"
OUT_WORD_BAR_LINE = HERE / "word_count_by_rating_bar_line.html"

OUT_ADJ_ADV_COMBINED_LINE = HERE / "adj_and_adv_count_by_rating_line.html"

OUT_STATS_XLSX = HERE / "rating_number_stats.xlsx"


def write_responsive_html(fig: go.Figure, out_path: Path, *, max_width_px: int = 1200) -> None:
    """
    Export a Plotly figure into a responsive HTML page.

    Key design choices:
    - Keep title/legend inside Plotly for consistent placement.
    - Keep footnotes outside Plotly (in HTML) so the plot area isn't squashed.
    - Force Plotly to match the container's height via relayout + ResizeObserver.
    """
    footer_html = (
        "<div class=\"footer\">"
        "<div class=\"note\"><i>Note: Reviews in the year 2021 were excluded due to missing rating entries.</i></div>"
        "<div class=\"credit\">Graphic by Claire Jong · Source: CULPA</div>"
        "</div>"
    )

    plot_div = fig.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        config={"responsive": True, "displayModeBar": False},
        default_width="100%",
        default_height="100%",
    )

    html = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{out_path.stem}</title>
    <style>
      body {{
        margin: 0;
        font-family: Roboto, -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
        background: #ffffff;
      }}
      .wrap {{
        max-width: {max_width_px}px;
        margin: 0 auto;
        padding: 12px 12px 24px;
      }}
      /* Plotly uses its own internal sizing; give it a stable height per breakpoint. */
      .plot {{
        width: 100%;
        height: 760px;
      }}
      .footer {{
        margin-top: 8px;
        font-size: 11px;
        color: rgba(0,0,0,0.7);
        line-height: 1.35;
      }}
      .footer .note {{
        font-style: italic;
      }}
      @media (max-width: 600px) {{
        .wrap {{ padding: 10px 10px 22px; }}
        .plot {{ height: 720px; }}
      }}
      @media (max-width: 320px) {{
        .wrap {{ padding: 8px 8px 20px; }}
        .plot {{ height: 660px; }}
      }}
    </style>
  </head>
  <body>
    <div class="wrap">
      <div class="plot">
        {plot_div}
      </div>
      {footer_html}
    </div>
    <script>
      function syncPlotSize() {{
        const container = document.querySelector('.plot');
        const el = document.querySelector('.plotly-graph-div');
        if (!container || !el || !window.Plotly) return;
        // Force Plotly to use the container's height (not its default ~450px).
        const h = Math.max(360, container.clientHeight);
        window.Plotly.relayout(el, {{ height: h, width: container.clientWidth }});
        window.Plotly.Plots.resize(el);
      }}

      // Run after scripts render the plot.
      window.addEventListener("load", () => {{
        syncPlotSize();
        if ('ResizeObserver' in window) {{
          const ro = new ResizeObserver(() => syncPlotSize());
          const container = document.querySelector('.plot');
          if (container) ro.observe(container);
        }}
      }});

      window.addEventListener("resize", () => syncPlotSize());
    </script>
  </body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def weighted_average_by_rating(grouped_df: pd.DataFrame, *, avg_col: str) -> pd.DataFrame:
    """
    Compute rating-level averages using per-(year,rating) averages weighted by `reviews`.
    This avoids re-running spaCy on every review.
    """

    df = grouped_df.copy()
    if "reviews" not in df.columns:
        raise ValueError("Expected `reviews` column in grouped CSV.")

    # Weighted mean: sum(avg * weight) / sum(weight).
    # Here, weights are review counts per (year, rating) so years with more reviews matter more.
    df["_weighted_sum"] = df[avg_col] * df["reviews"]
    out = (
        df.groupby("rating", as_index=False)
        .agg(weighted_avg=("_weighted_sum", "sum"), reviews=("reviews", "sum"))
    )
    out["avg"] = out["weighted_avg"] / out["reviews"]
    return out[["rating", "avg", "reviews"]].sort_values("rating")


def plot_line_and_bar(*, ratings_df: pd.DataFrame, metric_label: str, out_line: Path, out_bar: Path) -> None:
    # Builds the rating-only line chart used for `word_count_by_rating_line.html` (and similar).
    ratings_df = ratings_df.sort_values("rating")

    x = ratings_df["rating"].tolist()
    y = ratings_df["avg"].tolist()
    reviews = ratings_df["reviews"].tolist()

    # Line chart
    line_fig = go.Figure(
        data=[
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                name=metric_label,
                line=dict(color=LINE_COLOR),
                marker=dict(color=LINE_COLOR),
                customdata=[[r] for r in reviews],
                hovertemplate=(
                    "Rating %{x}<br>"
                    f"Avg {metric_label}: %{{y:.2f}}<br>"
                    "Reviews: %{customdata[0]}<extra></extra>"
                ),
            )
        ]
    )
    if out_line.name == OUT_WORD_LINE.name:
        story_title = "2-star reviews average the highest word count, while 4-star reviews are the shortest."
        line_fig.update_layout(
            title=dict(
                text=f"<b>{story_title}</b>",
                font=dict(family="Roboto", size=TITLE_SIZE),
                x=TITLE_X,
            ),
            margin=dict(t=120, b=70, l=70, r=40),
        )
    else:
        line_fig.update_layout(
            title=dict(text=f"Average {metric_label} per Review by Rating Number", font=dict(family="Roboto", size=18)),
            margin=dict(t=80, b=70),
        )

    line_fig.update_layout(
        xaxis_title="Rating (1-5)",
        yaxis_title=f"Average {metric_label}",
        hovermode="x",
        font=PLOT_FONT,
        xaxis=dict(tickmode="array", tickvals=RATING_TICKS),
    )
    if out_line.name == OUT_WORD_LINE.name:
        write_responsive_html(line_fig, out_line)
    else:
        line_fig.write_html(out_line)

    # Bar chart
    bar_fig = go.Figure(
        data=[
            go.Bar(
                x=x,
                y=y,
                customdata=[[r] for r in reviews],
                hovertemplate=(
                    "Rating %{x}<br>"
                    f"Avg {metric_label}: %{{y:.2f}}<br>"
                    "Reviews: %{customdata[0]}<extra></extra>"
                ),
            )
        ]
    )
    bar_fig.update_layout(
        title=f"Average {metric_label} per Review by Rating Number",
        xaxis_title="Rating (1-5)",
        yaxis_title=f"Average {metric_label}",
        hovermode="closest",
        font=PLOT_FONT,
        xaxis=dict(tickmode="array", tickvals=RATING_TICKS),
    )
    bar_fig.write_html(out_bar)


def plot_bar_with_trend_line(*, ratings_df: pd.DataFrame, metric_label: str, out_html: Path) -> None:
    ratings_df = ratings_df.sort_values("rating")

    x = ratings_df["rating"].tolist()
    y = ratings_df["avg"].tolist()
    reviews = ratings_df["reviews"].tolist()

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x,
            y=y,
            name=f"Average {metric_label}",
            customdata=[[r] for r in reviews],
            hovertemplate=(
                "Rating %{x}<br>"
                f"Avg {metric_label}: %{{y:.2f}}<br>"
                "Reviews: %{customdata[0]}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            name="Trend",
            line=dict(color=LINE_COLOR),
            marker=dict(color=LINE_COLOR),
            customdata=[[r] for r in reviews],
            hovertemplate=(
                "Rating %{x}<br>"
                f"Avg {metric_label}: %{{y:.2f}}<br>"
                "Reviews: %{customdata[0]}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=f"Average {metric_label} by Rating Number",
        xaxis_title="Rating (1-5)",
        yaxis_title=f"Average {metric_label}",
        hovermode="x",
        font=PLOT_FONT,
        xaxis=dict(tickmode="array", tickvals=RATING_TICKS),
    )
    fig.write_html(out_html)


def plot_adj_and_adv_combined(*, adj_df: pd.DataFrame, adv_df: pd.DataFrame) -> None:
    # Combined rating-only chart used for `adj_and_adv_count_by_rating_line.html`.
    adj_df = adj_df.sort_values("rating")
    adv_df = adv_df.sort_values("rating")

    x = RATING_TICKS
    adj_y = adj_df.set_index("rating").reindex(x)["avg"].tolist()
    adv_y = adv_df.set_index("rating").reindex(x)["avg"].tolist()

    adj_reviews = adj_df.set_index("rating").reindex(x)["reviews"].fillna(0).astype(int).tolist()
    adv_reviews = adv_df.set_index("rating").reindex(x)["reviews"].fillna(0).astype(int).tolist()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=adj_y,
            mode="lines+markers",
            name="Adjectives",
            line=dict(color="#81b7e7"),
            marker=dict(color="#81b7e7"),
            customdata=[[r] for r in adj_reviews],
            hovertemplate=(
                "Rating %{x}<br>"
                "Avg adjectives: %{y:.2f}<br>"
                "Reviews: %{customdata[0]}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=adv_y,
            mode="lines+markers",
            name="Adverbs",
            line=dict(color="#fc9332"),
            marker=dict(color="#fc9332"),
            customdata=[[r] for r in adv_reviews],
            hovertemplate=(
                "Rating %{x}<br>"
                "Avg adverbs: %{y:.2f}<br>"
                "Reviews: %{customdata[0]}<extra></extra>"
            ),
        )
    )
    story_title = "Both adjective and adverb use peak at 3 stars and drop sharply for 4-star reviews."

    fig.update_layout(
        title=dict(text=f"<b>{story_title}</b>", font=dict(family="Roboto", size=TITLE_SIZE), x=TITLE_X),
        margin=dict(t=155, b=70, l=70, r=40),
        xaxis_title="Rating (1-5)",
        yaxis_title="Average count per review",
        hovermode="x",
        font=PLOT_FONT,
        xaxis=dict(tickmode="array", tickvals=RATING_TICKS),
        legend=dict(
            x=0,
            y=LEGEND_Y,
            xanchor="left",
            yanchor="top",
            orientation="h",
            bgcolor="rgba(255,255,255,1.0)",
        ),
    )
    write_responsive_html(fig, OUT_ADJ_ADV_COMBINED_LINE)


def weighted_std(values: pd.Series, weights: pd.Series) -> float:
    weighted_mean = (values * weights).sum() / weights.sum()
    return (((weights * (values - weighted_mean) ** 2).sum()) / weights.sum()) ** 0.5


def build_group_level_metrics(pos_grouped: pd.DataFrame, words_grouped: pd.DataFrame) -> pd.DataFrame:
    grouped = pos_grouped.merge(
        words_grouped[["year", "rating", "avg_word_count", "reviews"]],
        on=["year", "rating", "reviews"],
        how="inner",
    )
    return grouped.sort_values(["year", "rating"])


def summarize_metric_stats(df: pd.DataFrame, metric_col: str, metric_label: str) -> dict:
    clean = df[["year", "rating", "reviews", metric_col]].dropna()
    x = clean["rating"]
    y = clean[metric_col]
    w = clean["reviews"]

    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_rho, spearman_p = stats.spearmanr(x, y)
    lin = stats.linregress(x, y)

    groups = [clean.loc[clean["rating"] == rating, metric_col] for rating in sorted(clean["rating"].unique())]
    anova_stat, anova_p = stats.f_oneway(*groups)

    return {
        "metric": metric_label,
        "analysis_basis": "year_rating_group_averages",
        "n_year_rating_groups": int(len(clean)),
        "total_reviews": int(w.sum()),
        "pearson_r": pearson_r,
        "pearson_p_value": pearson_p,
        "r_squared": lin.rvalue**2,
        "linear_slope": lin.slope,
        "linear_intercept": lin.intercept,
        "linear_regression_p_value": lin.pvalue,
        "spearman_rho": spearman_rho,
        "spearman_p_value": spearman_p,
        "anova_f_stat": anova_stat,
        "anova_p_value": anova_p,
        "weighted_mean_metric": (y * w).sum() / w.sum(),
        "weighted_std_metric": weighted_std(y, w),
    }


def build_stats_workbook(group_level_df: pd.DataFrame) -> None:
    summary_rows = [
        summarize_metric_stats(group_level_df, "avg_word_count", "word count"),
        summarize_metric_stats(group_level_df, "avg_adj_count", "adjective count"),
        summarize_metric_stats(group_level_df, "avg_adv_count", "adverb count"),
    ]
    summary_df = pd.DataFrame(summary_rows)

    means_by_rating = (
        group_level_df.groupby("rating", as_index=False)
        .agg(
            weighted_word_sum=("avg_word_count", lambda s: (s * group_level_df.loc[s.index, "reviews"]).sum()),
            weighted_adj_sum=("avg_adj_count", lambda s: (s * group_level_df.loc[s.index, "reviews"]).sum()),
            weighted_adv_sum=("avg_adv_count", lambda s: (s * group_level_df.loc[s.index, "reviews"]).sum()),
            reviews=("reviews", "sum"),
        )
        .sort_values("rating")
    )
    means_by_rating["avg_word_count"] = means_by_rating["weighted_word_sum"] / means_by_rating["reviews"]
    means_by_rating["avg_adj_count"] = means_by_rating["weighted_adj_sum"] / means_by_rating["reviews"]
    means_by_rating["avg_adv_count"] = means_by_rating["weighted_adv_sum"] / means_by_rating["reviews"]
    means_by_rating = means_by_rating[
        ["rating", "avg_word_count", "avg_adj_count", "avg_adv_count", "reviews"]
    ]

    with pd.ExcelWriter(OUT_STATS_XLSX, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary_statistics", index=False)
        means_by_rating.to_excel(writer, sheet_name="means_by_rating", index=False)
        group_level_df.to_excel(writer, sheet_name="year_rating_inputs", index=False)


def main() -> None:
    if not POS_BY_YEAR_RATING_CSV.exists():
        raise FileNotFoundError(f"Missing: {POS_BY_YEAR_RATING_CSV}")
    if not WORDS_BY_YEAR_RATING_CSV.exists():
        raise FileNotFoundError(f"Missing: {WORDS_BY_YEAR_RATING_CSV}")

    pos_grouped = pd.read_csv(POS_BY_YEAR_RATING_CSV)
    words_grouped = pd.read_csv(WORDS_BY_YEAR_RATING_CSV)
    # Exclude year 2000 everywhere; restrict to 2001-2025.
    pos_grouped = pos_grouped[(pos_grouped["year"] >= 2001) & (pos_grouped["year"] <= 2025)]
    words_grouped = words_grouped[(words_grouped["year"] >= 2001) & (words_grouped["year"] <= 2025)]
    # Exclude 2021 as noted in chart footers.
    pos_grouped = pos_grouped[pos_grouped["year"] != 2021]
    words_grouped = words_grouped[words_grouped["year"] != 2021]

    adj_by_rating = weighted_average_by_rating(pos_grouped, avg_col="avg_adj_count")
    adv_by_rating = weighted_average_by_rating(pos_grouped, avg_col="avg_adv_count")
    words_by_rating = weighted_average_by_rating(words_grouped, avg_col="avg_word_count")

    plot_line_and_bar(
        ratings_df=adj_by_rating,
        metric_label="adjectives",
        out_line=OUT_ADJ_LINE,
        out_bar=OUT_ADJ_BAR,
    )
    plot_line_and_bar(
        ratings_df=adv_by_rating,
        metric_label="adverbs",
        out_line=OUT_ADV_LINE,
        out_bar=OUT_ADV_BAR,
    )
    plot_line_and_bar(
        ratings_df=words_by_rating,
        metric_label="word count",
        out_line=OUT_WORD_LINE,
        out_bar=OUT_WORD_BAR,
    )
    plot_bar_with_trend_line(
        ratings_df=adj_by_rating,
        metric_label="adjectives",
        out_html=OUT_ADJ_BAR_LINE,
    )
    plot_bar_with_trend_line(
        ratings_df=adv_by_rating,
        metric_label="adverbs",
        out_html=OUT_ADV_BAR_LINE,
    )
    plot_bar_with_trend_line(
        ratings_df=words_by_rating,
        metric_label="word count",
        out_html=OUT_WORD_BAR_LINE,
    )

    plot_adj_and_adv_combined(adj_df=adj_by_rating, adv_df=adv_by_rating)

    group_level_df = build_group_level_metrics(pos_grouped, words_grouped)
    build_stats_workbook(group_level_df)

    print("Saved rating-only charts:")
    print(f"- {OUT_ADJ_LINE.name}")
    print(f"- {OUT_ADJ_BAR.name}")
    print(f"- {OUT_ADV_LINE.name}")
    print(f"- {OUT_ADV_BAR.name}")
    print(f"- {OUT_WORD_LINE.name}")
    print(f"- {OUT_WORD_BAR.name}")
    print(f"- {OUT_ADJ_BAR_LINE.name}")
    print(f"- {OUT_ADV_BAR_LINE.name}")
    print(f"- {OUT_WORD_BAR_LINE.name}")
    print(f"- {OUT_ADJ_ADV_COMBINED_LINE.name}")
    print(f"- {OUT_STATS_XLSX.name}")


if __name__ == "__main__":
    main()

