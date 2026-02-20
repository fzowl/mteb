#!/usr/bin/env python3
"""
Prompt Evolution Visualization — Streamlit App

Usage:
    cd scripts/prompt_evolution && streamlit run app.py
"""

import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from results_tracker import FALLBACK_BASELINES, get_measured_baselines

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data
def load_all_results() -> pd.DataFrame:
    """Load and combine all results CSV files under RESULTS_DIR."""
    csvs = list(RESULTS_DIR.rglob("results.csv"))
    if not csvs:
        st.error("No results.csv files found under " + str(RESULTS_DIR))
        return pd.DataFrame()

    frames = []
    for csv_path in csvs:
        df = pd.read_csv(csv_path)
        for col in ("corpus_prompt", "corpus_prompt_id", "prompt", "error"):
            if col not in df.columns:
                df[col] = ""
            df[col] = df[col].fillna("")
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    # Drop exact duplicates (same timestamp+dataset+prompt_id+corpus_prompt_id)
    df = df.drop_duplicates(subset=["timestamp", "dataset", "prompt_id", "corpus_prompt_id"])

    # Parse generation number
    df["generation"] = df["prompt_id"].str.extract(r"gen(\d+)").astype(float).astype("Int64")

    # Parse query/corpus prompt indices
    df["q_idx"] = df["prompt_id"].str.extract(r"_q(\d+)").astype(float).astype("Int64")
    df["c_idx"] = df["corpus_prompt_id"].str.extract(r"_c(\d+)").astype(float).astype("Int64")

    return df


def get_baseline(df: pd.DataFrame, dataset: str) -> float:
    """Get baseline score for a dataset."""
    measured = get_measured_baselines(df)
    return measured.get(dataset, FALLBACK_BASELINES.get(dataset, 0.0))


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Prompt Evolution Explorer", layout="wide")
    st.title("Prompt Evolution Explorer")

    df = load_all_results()
    if df.empty:
        st.stop()

    # ---- Sidebar ----
    st.sidebar.header("Filters")
    datasets = sorted(df["dataset"].unique())
    dataset = st.sidebar.selectbox("Dataset", datasets)

    df_ds = df[df["dataset"] == dataset].copy()
    gen_min, gen_max = int(df_ds["generation"].min()), int(df_ds["generation"].max())

    gen_range = st.sidebar.slider(
        "Generation range",
        min_value=gen_min,
        max_value=gen_max,
        value=(gen_min, gen_max),
    )
    df_ds = df_ds[(df_ds["generation"] >= gen_range[0]) & (df_ds["generation"] <= gen_range[1])]

    show_errors = st.sidebar.checkbox("Include error rows", value=False)
    if not show_errors:
        df_ds = df_ds[df_ds["score"].notna()]

    baseline = get_baseline(df, dataset)

    st.sidebar.markdown("---")
    st.sidebar.metric("Total rows", len(df_ds))
    st.sidebar.metric("Valid scores", df_ds["score"].notna().sum())
    st.sidebar.metric("Baseline", f"{baseline:.4f}")

    if df_ds.empty:
        st.warning("No data for the selected filters.")
        st.stop()

    # ---- Tabs ----
    tabs = st.tabs([
        "Generation Progress",
        "Score Distribution",
        "Prompt Heatmap",
        "3D Landscape",
        "Top Prompts",
        "Multi-Dataset",
    ])

    # ==== Tab 1: Generation Progress ====
    with tabs[0]:
        _tab_generation_progress(df_ds, baseline, dataset)

    # ==== Tab 2: Score Distribution ====
    with tabs[1]:
        _tab_score_distribution(df_ds, baseline, dataset)

    # ==== Tab 3: Prompt Heatmap ====
    with tabs[2]:
        _tab_prompt_heatmap(df_ds, baseline, dataset)

    # ==== Tab 4: 3D Landscape ====
    with tabs[3]:
        _tab_3d_landscape(df_ds, baseline, dataset)

    # ==== Tab 5: Top Prompts ====
    with tabs[4]:
        _tab_top_prompts(df_ds, baseline, dataset)

    # ==== Tab 6: Multi-Dataset ====
    with tabs[5]:
        _tab_multi_dataset(df)


# ---------------------------------------------------------------------------
# Tab implementations
# ---------------------------------------------------------------------------

def _tab_generation_progress(df_ds: pd.DataFrame, baseline: float, dataset: str):
    st.subheader(f"Generation Progress — {dataset}")

    valid = df_ds[df_ds["score"].notna()]
    if valid.empty:
        st.info("No valid scores.")
        return

    stats = valid.groupby("generation")["score"].agg(["max", "mean", "median", "min", "count"]).reset_index()
    stats.columns = ["generation", "best", "mean", "median", "min", "count"]

    fig = go.Figure()

    # Min-max shaded band
    fig.add_trace(go.Scatter(
        x=pd.concat([stats["generation"], stats["generation"][::-1]]),
        y=pd.concat([stats["best"], stats["min"][::-1]]),
        fill="toself",
        fillcolor="rgba(99,110,250,0.1)",
        line=dict(color="rgba(255,255,255,0)"),
        name="Min–Max range",
        hoverinfo="skip",
    ))

    for col, color, dash in [
        ("best", "#636EFA", "solid"),
        ("mean", "#EF553B", "dash"),
        ("median", "#00CC96", "dot"),
    ]:
        fig.add_trace(go.Scatter(
            x=stats["generation"],
            y=stats[col],
            mode="lines+markers",
            name=col.capitalize(),
            line=dict(color=color, dash=dash),
            customdata=stats["count"],
            hovertemplate=f"{col.capitalize()}: %{{y:.4f}}<br>Gen %{{x}}<br>Samples: %{{customdata}}<extra></extra>",
        ))

    # Baseline
    fig.add_hline(y=baseline, line_dash="longdash", line_color="gray",
                  annotation_text=f"Baseline {baseline:.4f}", annotation_position="bottom right")

    # Annotate peak
    peak_row = stats.loc[stats["best"].idxmax()]
    fig.add_annotation(
        x=peak_row["generation"],
        y=peak_row["best"],
        text=f"Peak: {peak_row['best']:.4f}",
        showarrow=True,
        arrowhead=2,
    )

    fig.update_layout(
        xaxis_title="Generation",
        yaxis_title="NDCG@10 Score",
        hovermode="x unified",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Sample count bar
    fig_count = px.bar(stats, x="generation", y="count", labels={"count": "Samples"},
                       title="Evaluations per generation", height=250)
    st.plotly_chart(fig_count, use_container_width=True)


def _tab_score_distribution(df_ds: pd.DataFrame, baseline: float, dataset: str):
    st.subheader(f"Score Distribution — {dataset}")

    valid = df_ds[df_ds["score"].notna()].copy()
    if valid.empty:
        st.info("No valid scores.")
        return

    valid["generation_str"] = "Gen " + valid["generation"].astype(str)

    n_gens = valid["generation"].nunique()
    # Color gradient: lighter early → darker late
    colors = px.colors.sample_colorscale("Blues", [0.3 + 0.7 * i / max(n_gens - 1, 1) for i in range(n_gens)])

    plot_type = st.radio("Plot type", ["Box", "Violin"], horizontal=True, key="dist_type")

    if plot_type == "Box":
        fig = px.box(
            valid.sort_values("generation"),
            x="generation",
            y="score",
            color="generation",
            color_discrete_sequence=colors,
            labels={"score": "NDCG@10", "generation": "Generation"},
        )
    else:
        fig = px.violin(
            valid.sort_values("generation"),
            x="generation",
            y="score",
            color="generation",
            color_discrete_sequence=colors,
            labels={"score": "NDCG@10", "generation": "Generation"},
            box=True,
        )

    fig.add_hline(y=baseline, line_dash="longdash", line_color="red",
                  annotation_text=f"Baseline {baseline:.4f}")
    fig.update_layout(showlegend=False, height=550)
    st.plotly_chart(fig, use_container_width=True)


def _tab_prompt_heatmap(df_ds: pd.DataFrame, baseline: float, dataset: str):
    st.subheader(f"Prompt Heatmap — {dataset}")

    valid = df_ds[df_ds["score"].notna() & df_ds["q_idx"].notna() & df_ds["c_idx"].notna()].copy()
    if valid.empty:
        st.info("No valid data for heatmap.")
        return

    gens = sorted(valid["generation"].dropna().unique())
    sel_gen = st.select_slider("Generation", options=gens, value=gens[-1], key="hm_gen")
    gen_data = valid[valid["generation"] == sel_gen]

    if gen_data.empty:
        st.info(f"No data for generation {sel_gen}.")
        return

    pivot = gen_data.pivot_table(index="q_idx", columns="c_idx", values="score", aggfunc="first")

    # Build hover text with prompt snippets
    hover_text = []
    for qi in pivot.index:
        row_text = []
        for ci in pivot.columns:
            cell = gen_data[(gen_data["q_idx"] == qi) & (gen_data["c_idx"] == ci)]
            if not cell.empty:
                r = cell.iloc[0]
                qp = str(r["prompt"])[:80]
                cp = str(r["corpus_prompt"])[:80]
                row_text.append(f"Q: {qp}<br>C: {cp}<br>Score: {r['score']:.4f}")
            else:
                row_text.append("")
        hover_text.append(row_text)

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[f"C{int(c)}" for c in pivot.columns],
        y=[f"Q{int(q)}" for q in pivot.index],
        text=hover_text,
        hovertemplate="%{text}<extra></extra>",
        colorscale=[
            [0, "red"],
            [0.5, "white"],
            [1, "green"],
        ],
        zmid=baseline,
        colorbar=dict(title="Score"),
    ))

    fig.update_layout(
        xaxis_title="Corpus Prompt",
        yaxis_title="Query Prompt",
        height=600,
        title=f"Gen {sel_gen} — Scores (red < baseline {baseline:.4f} < green)",
    )
    st.plotly_chart(fig, use_container_width=True)


def _tab_3d_landscape(df_ds: pd.DataFrame, baseline: float, dataset: str):
    st.subheader(f"3D Prompt Landscape — {dataset}")

    valid = df_ds[df_ds["score"].notna() & df_ds["q_idx"].notna() & df_ds["c_idx"].notna()].copy()
    if valid.empty:
        st.info("No valid data for 3D view.")
        return

    gens = sorted(valid["generation"].dropna().unique())
    sel_gen = st.select_slider("Generation", options=gens, value=gens[-1], key="3d_gen")
    gen_data = valid[valid["generation"] == sel_gen]

    if gen_data.empty:
        st.info(f"No data for generation {sel_gen}.")
        return

    pivot = gen_data.pivot_table(index="q_idx", columns="c_idx", values="score", aggfunc="first")

    # Surface plot
    fig = go.Figure()

    fig.add_trace(go.Surface(
        z=pivot.values,
        x=pivot.columns.values,
        y=pivot.index.values,
        colorscale="Viridis",
        colorbar=dict(title="Score"),
        name="Scores",
    ))

    # Baseline plane
    fig.add_trace(go.Surface(
        z=[[baseline] * len(pivot.columns)] * len(pivot.index),
        x=pivot.columns.values,
        y=pivot.index.values,
        colorscale=[[0, "rgba(200,200,200,0.3)"], [1, "rgba(200,200,200,0.3)"]],
        showscale=False,
        name="Baseline",
        opacity=0.4,
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title="Corpus Prompt Idx",
            yaxis_title="Query Prompt Idx",
            zaxis_title="Score",
        ),
        title=f"Gen {sel_gen} — 3D Score Landscape",
        height=650,
    )
    st.plotly_chart(fig, use_container_width=True)


def _tab_top_prompts(df_ds: pd.DataFrame, baseline: float, dataset: str):
    st.subheader(f"Top Prompts — {dataset}")

    valid = df_ds[df_ds["score"].notna()].copy()
    if valid.empty:
        st.info("No valid scores.")
        return

    top_n = st.slider("Top N", 5, 50, 20, key="top_n")
    top = valid.nlargest(top_n, "score").copy()
    top["improvement"] = top["score"] - baseline
    top["improvement_pct"] = (top["improvement"] / baseline * 100).round(2)

    display_cols = ["generation", "prompt_id", "corpus_prompt_id", "score", "improvement", "improvement_pct"]
    st.dataframe(
        top[display_cols].rename(columns={
            "generation": "Gen",
            "prompt_id": "Query Prompt ID",
            "corpus_prompt_id": "Corpus Prompt ID",
            "score": "Score",
            "improvement": "Δ Score",
            "improvement_pct": "Δ %",
        }),
        use_container_width=True,
        height=400,
    )

    # Expandable prompt details
    st.markdown("#### Prompt Details")
    for i, (_, row) in enumerate(top.iterrows()):
        with st.expander(f"#{i+1} — Score {row['score']:.4f} ({row['improvement_pct']:+.2f}%)"):
            st.markdown(f"**Query prompt** (`{row['prompt_id']}`):")
            st.code(row["prompt"], language=None)
            st.markdown(f"**Corpus prompt** (`{row['corpus_prompt_id']}`):")
            st.code(row["corpus_prompt"], language=None)

    # CSV download
    csv_data = top.to_csv(index=False)
    st.download_button("Download CSV", csv_data, file_name=f"top_prompts_{dataset}.csv", mime="text/csv")


def _tab_multi_dataset(df_all: pd.DataFrame):
    st.subheader("Multi-Dataset Comparison")

    valid = df_all[df_all["score"].notna()]
    if valid.empty:
        st.info("No valid scores.")
        return

    baselines = get_measured_baselines(df_all)
    rows = []
    for ds in sorted(valid["dataset"].unique()):
        bl = baselines.get(ds, FALLBACK_BASELINES.get(ds, 0.0))
        best = valid[valid["dataset"] == ds]["score"].max()
        improvement = best - bl
        pct = (improvement / bl * 100) if bl > 0 else 0
        n_gens = valid[valid["dataset"] == ds]["generation"].nunique() if "generation" in valid.columns else 0
        rows.append({
            "Dataset": ds,
            "Baseline": bl,
            "Best": best,
            "Δ Score": improvement,
            "Δ %": round(pct, 2),
            "Generations": n_gens,
            "Evaluations": len(valid[valid["dataset"] == ds]),
        })

    summary = pd.DataFrame(rows)

    # Metric cards
    cols = st.columns(len(summary))
    for i, row in summary.iterrows():
        with cols[i]:
            st.metric(row["Dataset"], f"{row['Best']:.4f}", f"{row['Δ %']:+.2f}%")
            st.caption(f"Baseline: {row['Baseline']:.4f} | Gens: {row['Generations']}")

    # Grouped bar chart
    melted = summary.melt(id_vars="Dataset", value_vars=["Baseline", "Best"],
                          var_name="Type", value_name="Score")
    fig = px.bar(
        melted,
        x="Dataset",
        y="Score",
        color="Type",
        barmode="group",
        text_auto=".4f",
        color_discrete_map={"Baseline": "#AAAAAA", "Best": "#636EFA"},
        title="Baseline vs Best Score per Dataset",
        height=450,
    )

    # Add improvement % annotations
    for _, row in summary.iterrows():
        fig.add_annotation(
            x=row["Dataset"],
            y=max(row["Baseline"], row["Best"]) + 0.005,
            text=f"{row['Δ %']:+.1f}%",
            showarrow=False,
            font=dict(size=12, color="green" if row["Δ %"] > 0 else "red"),
        )

    fig.update_layout(yaxis_title="NDCG@10 Score")
    st.plotly_chart(fig, use_container_width=True)

    # Summary table
    st.dataframe(summary, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
