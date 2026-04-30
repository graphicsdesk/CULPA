import pandas as pd
import plotly.graph_objects as go
from pathlib import Path


DATA_PATH = Path(__file__).resolve().parent / "review.csv"
OUT_WORD_COUNTS = Path(__file__).resolve().parent / "word_count_by_year_and_rating.csv"

OUT_LINE = Path(__file__).resolve().parent / "word_count_by_year_and_rating_line.html"
OUT_BAR = Path(__file__).resolve().parent / "word_count_by_year_and_rating_bar.html"

PLOT_FONT = dict(family="Roboto")
RATING_LINE_COLORS = {
    1: "#1d2c4d",
    2: "#81b7e7",
    3: "#fc9332",
    4: "#825942",
    5: "#fcea42",
}

TITLE_SIZE = 26
SUBTITLE_SIZE = 12
FOOTNOTE_SIZE = 11
TITLE_X = 0.03
LEGEND_Y = 1.00


def write_responsive_html(fig: go.Figure, out_path: Path, *, max_width_px: int = 1200) -> None:
    footer_html = (
        "<div class=\"footer\">"
        "<div class=\"note\"><i>Note: Reviews in the year 2021 were excluded due to missing rating entries.</i></div>"
        "<div class=\"credit\">Chart: Claire Jong · Source: CULPA</div>"
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
      .plot {{
        width: 100%;
        height: 820px;
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
        .plot {{ height: 760px; }}
      }}
      @media (max-width: 320px) {{
        .wrap {{ padding: 8px 8px 20px; }}
        .plot {{ height: 700px; }}
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
        const h = Math.max(420, container.clientHeight);
        window.Plotly.relayout(el, {{ height: h, width: container.clientWidth }});
        window.Plotly.Plots.resize(el);
      }}

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


def load_reviews() -> pd.DataFrame:
    # Load, type-clean, and filter (2001–2025; exclude 2021 per note).
    df = pd.read_csv(DATA_PATH, header=None)
    df.columns = ["id", "prof_id", "review_text", "extra", "workload_text", "rating", "date"]

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["parsed_date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["rating", "parsed_date"])

    df["rating"] = df["rating"].round().astype(int)
    df["year"] = df["parsed_date"].dt.year.astype(int)
    df = df[(df["year"] >= 2001) & (df["year"] <= 2025)]
    df = df[df["year"] != 2021]

    df["full_text"] = df["review_text"].fillna("") + " " + df["workload_text"].fillna("")
    df["full_text"] = df["full_text"].astype(str)
    return df


def compute_word_count(df: pd.DataFrame) -> pd.DataFrame:
    # Simple, consistent word count based on whitespace splitting.
    df = df.copy()
    df["word_count"] = df["full_text"].apply(lambda x: len(str(x).split()))
    return df


def plot_word_count_by_year_and_rating(grouped: pd.DataFrame) -> None:
    # One line per rating; x=year, y=avg word count (hover shows year once + per-rating blocks).
    grouped = grouped.sort_values(["rating", "year"])
    ratings = sorted(grouped["rating"].unique())

    # --- Line chart ---
    line_fig = go.Figure()
    for r in ratings:
        sub = grouped[grouped["rating"] == r].sort_values("year")
        color = RATING_LINE_COLORS.get(int(r), "#1d2c4d")
        label = f"{int(r)}-star rating"
        line_fig.add_trace(
            go.Scatter(
                x=sub["year"],
                y=sub["avg_word_count"],
                mode="lines+markers",
                name=label,
                line=dict(color=color),
                marker=dict(color=color),
                text=[label] * len(sub),
                customdata=sub[["reviews"]].to_numpy(),
                hovertemplate=(
                    "%{text}<br>"
                    "Avg word count: %{y:.2f}<br>"
                    "Num of reviews: %{customdata[0]}<extra></extra>"
                ),
            )
        )

    title_text = (
        "<b>After a decade of increasingly lengthy reviews, students across all ratings have been generally<br>"
        "writing shorter reviews since 2014.</b>"
    )

    line_fig.update_layout(
        title=dict(
            text=title_text,
            font=dict(family="Roboto", size=TITLE_SIZE),
            x=TITLE_X,
        ),
        xaxis_title="Year",
        yaxis_title="Average word count",
        # Unified hover = single "Year ..." header without repeating per-trace.
        hovermode="x unified",
        xaxis=dict(
            unifiedhovertitle=dict(text="Year %{x}"),
        ),
        font=PLOT_FONT,
        margin=dict(t=230, b=70, l=70, r=40),
        # Reduce gap between legend and plot while still reserving header space.
        yaxis=dict(domain=[0.0, 0.90]),
        legend=dict(
            x=0,
            y=LEGEND_Y,
            xanchor="left",
            yanchor="top",
            orientation="h",
            bgcolor="rgba(255,255,255,1.0)",
        ),
    )
    write_responsive_html(line_fig, OUT_LINE)

    # --- Bar chart ---
    bar_fig = go.Figure()
    for r in ratings:
        sub = grouped[grouped["rating"] == r].sort_values("year")
        bar_fig.add_trace(
            go.Bar(
                x=sub["year"],
                y=sub["avg_word_count"],
                name=f"Rating {r}",
                text=[r] * len(sub),
                customdata=sub[["reviews"]].to_numpy(),
                hovertemplate=(
                    "Year %{x}<br>"
                    "Rating %{text}<br>"
                    "Avg word count: %{y:.2f}<br>"
                    "Reviews: %{customdata[0]}<extra></extra>"
                ),
            )
        )

    bar_fig.update_layout(
        title="Average Word Count by Year and Rating",
        xaxis_title="Year",
        yaxis_title="Average Word Count",
        barmode="group",
        hovermode="closest",
        font=PLOT_FONT,
    )
    bar_fig.write_html(OUT_BAR)


def main() -> None:
    df = load_reviews()
    df = compute_word_count(df)

    grouped = (
        df.groupby(["year", "rating"], as_index=False)
        .agg(avg_word_count=("word_count", "mean"), reviews=("id", "size"))
    )
    grouped.to_csv(OUT_WORD_COUNTS, index=False)
    print(f"Wrote grouped counts to: {OUT_WORD_COUNTS.name}")

    plot_word_count_by_year_and_rating(grouped)
    print(f"Saved line chart: {OUT_LINE.name}")
    print(f"Saved bar chart: {OUT_BAR.name}")


if __name__ == "__main__":
    main()