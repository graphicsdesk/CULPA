import warnings
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import spacy


warnings.filterwarnings("ignore", category=UserWarning, module="spacy")

PLOT_FONT = dict(family="Roboto")
LINE_COLOR = "#1d2c4d"

DATA_PATH = Path(__file__).resolve().parent / "review.csv"
OUT_POS_COUNTS = Path(__file__).resolve().parent / "pos_counts_by_year_and_rating.csv"

OUT_ADJ_LINE = Path(__file__).resolve().parent / "adj_count_by_year_and_rating_line.html"
OUT_ADJ_BAR = Path(__file__).resolve().parent / "adj_count_by_year_and_rating_bar.html"
OUT_ADV_LINE = Path(__file__).resolve().parent / "adv_count_by_year_and_rating_line.html"
OUT_ADV_BAR = Path(__file__).resolve().parent / "adv_count_by_year_and_rating_bar.html"


def load_reviews() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, header=None)
    df.columns = ["id", "prof_id", "review_text", "extra", "workload_text", "rating", "date"]

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["parsed_date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["rating", "parsed_date"])

    # Ensure rating is numeric but also stable for plotting/legend labels.
    df["rating"] = df["rating"].round().astype(int)
    df["year"] = df["parsed_date"].dt.year.astype(int)
    df = df[(df["year"] >= 2001) & (df["year"] <= 2025)]
    df = df[df["year"] != 2021]

    df["full_text"] = df["review_text"].fillna("") + " " + df["workload_text"].fillna("")
    df["full_text"] = df["full_text"].astype(str)
    return df


def compute_adj_adv_counts(df: pd.DataFrame) -> pd.DataFrame:
    # Use the small model for speed; disable unnecessary components.
    nlp = spacy.load(
        "en_core_web_sm",
        # `doc.count_by(spacy.attrs.POS)` works correctly with this minimal disable set.
        disable=["parser", "ner"],
    )

    adj_id = nlp.vocab.strings["ADJ"]
    adv_id = nlp.vocab.strings["ADV"]

    texts = df["full_text"].tolist()
    adj_counts = []
    adv_counts = []

    print(f"Analyzing {len(texts):,} reviews for ADJ/ADV counts...")
    for i, doc in enumerate(nlp.pipe(texts, batch_size=1000)):
        # Fast POS counting via spaCy's internal counts.
        pos_counts = doc.count_by(spacy.attrs.POS)
        adj_counts.append(pos_counts.get(adj_id, 0))
        adv_counts.append(pos_counts.get(adv_id, 0))
        if (i + 1) % 5000 == 0:
            print(f"  processed {i+1:,}/{len(texts):,} reviews...")

    df = df.copy()
    df["adj_count"] = adj_counts
    df["adv_count"] = adv_counts
    return df


def plot_metric_by_year_rating(
    grouped: pd.DataFrame,
    *,
    metric_col: str,
    metric_label: str,
    out_line: Path,
    out_bar: Path,
) -> None:
    grouped = grouped.sort_values(["rating", "year"])
    ratings = sorted(grouped["rating"].unique())

    # --- Line chart ---
    line_fig = go.Figure()
    for r in ratings:
        sub = grouped[grouped["rating"] == r].sort_values("year")
        line_fig.add_trace(
            go.Scatter(
                x=sub["year"],
                y=sub[metric_col],
                mode="lines+markers",
                name=f"Rating {r}",
                line=dict(color=LINE_COLOR),
                marker=dict(color=LINE_COLOR),
                text=[r] * len(sub),
                customdata=sub[["reviews"]].to_numpy(),
                hovertemplate=(
                    "Year %{x}<br>"
                    f"Rating %{{text}}<br>"
                    f"Avg {metric_label}: %{{y:.2f}}<br>"
                    "Reviews: %{customdata[0]}<extra></extra>"
                ),
            )
        )

    line_fig.update_layout(
        title=f"Average {metric_label} by Year and Rating",
        xaxis_title="Year",
        yaxis_title=f"Average {metric_label}",
        hovermode="x",
        font=PLOT_FONT,
    )
    line_fig.write_html(out_line)

    # --- Bar chart ---
    bar_fig = go.Figure()
    for r in ratings:
        sub = grouped[grouped["rating"] == r].sort_values("year")
        bar_fig.add_trace(
            go.Bar(
                x=sub["year"],
                y=sub[metric_col],
                name=f"Rating {r}",
                text=[r] * len(sub),
                customdata=sub[["reviews"]].to_numpy(),
                hovertemplate=(
                    "Year %{x}<br>"
                    f"Rating %{{text}}<br>"
                    f"Avg {metric_label}: %{{y:.2f}}<br>"
                    "Reviews: %{customdata[0]}<extra></extra>"
                ),
            )
        )

    bar_fig.update_layout(
        title=f"Average {metric_label} by Year and Rating",
        xaxis_title="Year",
        yaxis_title=f"Average {metric_label}",
        barmode="group",
        hovermode="closest",
        font=PLOT_FONT,
    )
    bar_fig.write_html(out_bar)


def main() -> None:
    df = load_reviews()
    df = compute_adj_adv_counts(df)

    grouped = (
        df.groupby(["year", "rating"], as_index=False)
        .agg(
            avg_adj_count=("adj_count", "mean"),
            avg_adv_count=("adv_count", "mean"),
            reviews=("id", "size"),
        )
    )

    grouped.to_csv(OUT_POS_COUNTS, index=False)
    print(f"Wrote grouped counts to: {OUT_POS_COUNTS.name}")

    plot_metric_by_year_rating(
        grouped,
        metric_col="avg_adj_count",
        metric_label="adjectives",
        out_line=OUT_ADJ_LINE,
        out_bar=OUT_ADJ_BAR,
    )
    print(f"Saved line chart: {OUT_ADJ_LINE.name}")
    print(f"Saved bar chart: {OUT_ADJ_BAR.name}")

    plot_metric_by_year_rating(
        grouped,
        metric_col="avg_adv_count",
        metric_label="adverbs",
        out_line=OUT_ADV_LINE,
        out_bar=OUT_ADV_BAR,
    )
    print(f"Saved line chart: {OUT_ADV_LINE.name}")
    print(f"Saved bar chart: {OUT_ADV_BAR.name}")


if __name__ == "__main__":
    main()

