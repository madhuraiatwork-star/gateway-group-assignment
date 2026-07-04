"""
streamlit_app.py
================
Dashboard for the Triage Classifier batch results.

Displays results.json as a rich, interactive table with:
  - needs_human=True rows highlighted in amber
  - Priority badge colour-coding (P0 red, P1 orange, P2 blue, P3 grey)
  - Sidebar: priority distribution chart + human-review summary metric cards
  - Expandable detail panel per message (raw text + suggested action)
  - Token usage and latency sparklines in the sidebar
"""

import json
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Page config — must be the FIRST st call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Triage Classifier Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── App background ── */
.stApp {
    background: linear-gradient(135deg, #0f1117 0%, #1a1d2e 50%, #0f1117 100%);
    color: #e2e8f0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1d2e 0%, #111827 100%);
    border-right: 1px solid rgba(99,102,241,0.2);
}

/* ── Metric cards in sidebar ── */
[data-testid="metric-container"] {
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 12px;
    padding: 12px !important;
}

/* ── Table header ── */
thead th {
    background: rgba(99,102,241,0.15) !important;
    color: #a5b4fc !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    font-size: 0.72rem !important;
}

/* ── Highlighted row style used via pandas Styler ── */
.needs-human {
    background-color: rgba(251,191,36,0.12) !important;
    border-left: 3px solid #fbbf24 !important;
}

/* ── Priority badge colours ── */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.04em;
}
.p0 { background: rgba(239,68,68,0.15); color: #f87171; border: 1px solid rgba(239,68,68,0.4); }
.p1 { background: rgba(249,115,22,0.15); color: #fb923c; border: 1px solid rgba(249,115,22,0.4); }
.p2 { background: rgba(59,130,246,0.15); color: #60a5fa; border: 1px solid rgba(59,130,246,0.4); }
.p3 { background: rgba(100,116,139,0.15); color: #94a3b8; border: 1px solid rgba(100,116,139,0.3); }

/* ── Section titles ── */
h2, h3 { color: #c7d2fe !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: rgba(30, 33, 48, 0.6);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 10px;
}

/* ── Divider ── */
hr { border-color: rgba(99,102,241,0.2) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.4); border-radius: 99px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RESULTS_FILE = Path(__file__).parent / "results.json"

PRIORITY_ORDER  = ["P0", "P1", "P2", "P3"]
PRIORITY_COLORS = {
    "P0": "#f87171",
    "P1": "#fb923c",
    "P2": "#60a5fa",
    "P3": "#94a3b8",
}
CATEGORY_ICONS = {
    "technical_support": "🔧",
    "billing":           "💳",
    "feedback":          "💬",
    "security":          "🔒",
    "unclassified":      "❓",
    "general_inquiry":   "📋",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def build_dataframe(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        d = r.get("decision", {})
        m = r.get("meta", {})
        tok = m.get("tokens", {})
        rows.append({
            "message_id":    r.get("message_id", ""),
            "category":      d.get("category", ""),
            "priority":      d.get("priority", ""),
            "needs_human":   d.get("needs_human", False),
            "confidence":    d.get("confidence", 0.0),
            "summary":       d.get("summary", ""),
            "suggested_action": d.get("suggested_action", ""),
            "raw_text":      r.get("raw_text", ""),
            "latency_s":     m.get("latency_seconds", 0.0),
            "total_tokens":  tok.get("total_tokens", 0),
            "prompt_tokens": tok.get("prompt_tokens", 0),
            "completion_tokens": tok.get("completion_tokens", 0),
        })
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Helper renderers
# ---------------------------------------------------------------------------
def priority_badge(p: str) -> str:
    css = p.lower()
    return f'<span class="badge {css}">{p}</span>'

def category_icon(cat: str) -> str:
    return CATEGORY_ICONS.get(cat, "📄")

def confidence_bar(val: float) -> str:
    pct = int(val * 100)
    colour = "#f87171" if pct < 60 else "#fb923c" if pct < 80 else "#4ade80"
    return (
        f'<div style="display:flex;align-items:center;gap:6px;">'
        f'<div style="flex:1;background:rgba(255,255,255,0.08);border-radius:99px;height:6px;">'
        f'<div style="width:{pct}%;background:{colour};border-radius:99px;height:6px;"></div></div>'
        f'<span style="font-size:0.8rem;color:{colour};font-weight:600;min-width:36px">{pct}%</span>'
        f'</div>'
    )

def human_badge(val: bool) -> str:
    if val:
        return '<span style="color:#fbbf24;font-weight:700;">&#9888; Yes</span>'
    return '<span style="color:#4ade80;font-weight:600;">No</span>'

def row_style(row: pd.Series) -> list[str]:
    if row["needs_human"]:
        return ["background-color: rgba(251,191,36,0.10); border-left: 3px solid #fbbf24"] * len(row)
    return [""] * len(row)

# ---------------------------------------------------------------------------
# Plotly helpers
# ---------------------------------------------------------------------------
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#94a3b8", size=12),
    margin=dict(l=0, r=0, t=24, b=0),
)

def make_priority_donut(df: pd.DataFrame) -> go.Figure:
    counts = df["priority"].value_counts().reindex(PRIORITY_ORDER, fill_value=0)
    fig = go.Figure(go.Pie(
        labels=counts.index.tolist(),
        values=counts.values.tolist(),
        hole=0.62,
        marker=dict(
            colors=[PRIORITY_COLORS[p] for p in counts.index],
            line=dict(color="#0f1117", width=3),
        ),
        textinfo="percent+label",
        textfont=dict(size=11, family="Inter"),
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        showlegend=False,
        annotations=[dict(
            text=f"<b>{len(df)}</b><br><span style='font-size:10px'>msgs</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="#e2e8f0", family="Inter"),
        )],
        height=230,
    )
    return fig

def make_latency_bar(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=df["message_id"],
        y=df["latency_s"],
        marker=dict(
            color=df["latency_s"],
            colorscale=[[0, "#4ade80"], [0.5, "#fb923c"], [1, "#f87171"]],
            showscale=False,
        ),
        hovertemplate="<b>%{x}</b><br>Latency: %{y:.3f}s<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=160,
        xaxis=dict(showticklabels=True, tickangle=-45, tickfont=dict(size=9)),
        yaxis=dict(title="sec", gridcolor="rgba(255,255,255,0.06)"),
    )
    return fig

def make_token_bar(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Prompt",
        x=df["message_id"],
        y=df["prompt_tokens"],
        marker_color="#6366f1",
        hovertemplate="%{x}<br>Prompt: %{y}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Completion",
        x=df["message_id"],
        y=df["completion_tokens"],
        marker_color="#a78bfa",
        hovertemplate="%{x}<br>Completion: %{y}<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        barmode="stack",
        height=160,
        xaxis=dict(showticklabels=True, tickangle=-45, tickfont=dict(size=9)),
        yaxis=dict(title="tok", gridcolor="rgba(255,255,255,0.06)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=10)),
    )
    return fig

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    records = load_results(RESULTS_FILE)

    # ── Header ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="padding: 1.5rem 0 0.5rem 0;">
      <h1 style="margin:0; font-size:1.9rem; font-weight:700; color:#e2e8f0;">
        🔍 Triage Classifier Dashboard
      </h1>
      <p style="margin:4px 0 0 0; color:#64748b; font-size:0.95rem;">
        Results from <code>results.json</code> &nbsp;·&nbsp; powered by <b>llama-3.1-8b-instant</b> via Groq
      </p>
    </div>
    <hr>
    """, unsafe_allow_html=True)

    if not records:
        st.warning(
            "**No results found.** Run `python batch_process.py` first to generate `results.json`.",
            icon="⚠️",
        )
        return

    df = build_dataframe(records)

    # ── Sidebar ─────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 📊 Session Summary")

        n_total  = len(df)
        n_human  = int(df["needs_human"].sum())
        pct_human = (n_human / n_total * 100) if n_total else 0
        avg_conf  = df["confidence"].mean()
        avg_lat   = df["latency_s"].mean()
        total_tok = int(df["total_tokens"].sum())

        # Metric cards
        col1, col2 = st.columns(2)
        col1.metric("Total Messages", n_total)
        col2.metric("Needs Human", f"{n_human} ({pct_human:.0f}%)")
        col1.metric("Avg Confidence", f"{avg_conf:.1%}")
        col2.metric("Avg Latency", f"{avg_lat:.3f}s")
        st.metric("Total Tokens", f"{total_tok:,}")

        st.markdown("---")
        st.markdown("#### Priority Distribution")
        st.plotly_chart(make_priority_donut(df), use_container_width=True, config={"displayModeBar": False})

        # Priority count table
        pri_counts = (
            df.groupby("priority")
              .size()
              .reindex(PRIORITY_ORDER, fill_value=0)
              .reset_index()
        )
        pri_counts.columns = ["Priority", "Count"]
        pri_counts["Share"] = (pri_counts["Count"] / n_total * 100).map(lambda x: f"{x:.0f}%")
        st.dataframe(
            pri_counts,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Priority": st.column_config.TextColumn("Priority", width="small"),
                "Count":    st.column_config.NumberColumn("Count",  width="small"),
                "Share":    st.column_config.TextColumn("Share",   width="small"),
            },
        )

        st.markdown("---")

        # Human-review gauge
        st.markdown("#### Human Review Rate")
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pct_human,
            number={"suffix": "%", "font": {"size": 28, "color": "#fbbf24", "family": "Inter"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#475569", "tickfont": {"size": 9}},
                "bar": {"color": "#fbbf24", "thickness": 0.3},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40],  "color": "rgba(74,222,128,0.08)"},
                    {"range": [40, 70], "color": "rgba(251,191,36,0.08)"},
                    {"range": [70, 100],"color": "rgba(248,113,113,0.08)"},
                ],
                "threshold": {"line": {"color": "#f87171", "width": 3}, "thickness": 0.75, "value": 70},
            },
        ))
        gauge.update_layout(
            **PLOTLY_LAYOUT,
            height=200,
        )
        st.plotly_chart(gauge, use_container_width=True, config={"displayModeBar": False})

    # ── Filter bar ──────────────────────────────────────────────────────────
    st.markdown("### 📋 Classification Results")

    fcol1, fcol2, fcol3 = st.columns([2, 2, 2])
    with fcol1:
        sel_priorities = st.multiselect(
            "Filter by Priority",
            options=PRIORITY_ORDER,
            default=PRIORITY_ORDER,
        )
    with fcol2:
        all_cats = sorted(df["category"].unique().tolist())
        sel_cats = st.multiselect(
            "Filter by Category",
            options=all_cats,
            default=all_cats,
        )
    with fcol3:
        human_filter = st.radio(
            "Needs Human",
            options=["All", "Flagged only", "Auto-resolved only"],
            horizontal=True,
        )

    # Apply filters
    mask = df["priority"].isin(sel_priorities) & df["category"].isin(sel_cats)
    if human_filter == "Flagged only":
        mask &= df["needs_human"]
    elif human_filter == "Auto-resolved only":
        mask &= ~df["needs_human"]
    filtered = df[mask].reset_index(drop=True)

    st.markdown(
        f'<p style="color:#64748b;font-size:0.85rem;margin-bottom:8px;">'
        f'Showing <b style="color:#a5b4fc">{len(filtered)}</b> of {n_total} messages'
        f'</p>',
        unsafe_allow_html=True,
    )

    # ── Results table (rendered row-by-row for full HTML control) ───────────
    if filtered.empty:
        st.info("No messages match the current filters.")
    else:
        # Build HTML table
        header_html = (
            "<table style='width:100%;border-collapse:collapse;font-size:0.85rem;'>"
            "<thead><tr style='background:rgba(99,102,241,0.12);'>"
        )
        for col in ["ID", "Category", "Priority", "Needs Human", "Confidence", "Summary", "Tokens", "Latency"]:
            header_html += (
                f"<th style='padding:10px 12px;text-align:left;color:#a5b4fc;"
                f"font-size:0.72rem;letter-spacing:0.06em;font-weight:600;"
                f"text-transform:uppercase;white-space:nowrap;border-bottom:1px solid rgba(99,102,241,0.25);'>"
                f"{col}</th>"
            )
        header_html += "</tr></thead><tbody>"

        rows_html = ""
        for _, row in filtered.iterrows():
            bg = "rgba(251,191,36,0.07)" if row["needs_human"] else "rgba(15,17,23,0.4)"
            left_border = "border-left: 3px solid #fbbf24;" if row["needs_human"] else "border-left: 3px solid transparent;"
            icon = category_icon(row["category"])
            pri  = row["priority"]
            pri_colour = PRIORITY_COLORS.get(pri, "#94a3b8")
            conf_pct = int(row["confidence"] * 100)
            conf_colour = "#f87171" if conf_pct < 60 else "#fb923c" if conf_pct < 80 else "#4ade80"
            human_html = (
                '<span style="color:#fbbf24;font-weight:700;">&#9888; Yes</span>'
                if row["needs_human"]
                else '<span style="color:#4ade80;">No</span>'
            )
            summary_short = (row["summary"][:90] + "…") if len(row["summary"]) > 90 else row["summary"]
            tok = row["total_tokens"]
            tok_display = f"{tok:,}" if tok else '<span style="color:#475569">—</span>'
            lat_display = f"{row['latency_s']:.3f}s" if row['latency_s'] > 0.01 else '<span style="color:#475569">—</span>'

            rows_html += (
                f"<tr style='background:{bg};{left_border}transition:background 0.15s;'>"
                # ID
                f"<td style='padding:10px 12px;font-family:monospace;font-size:0.8rem;"
                f"color:#94a3b8;white-space:nowrap;border-bottom:1px solid rgba(255,255,255,0.04);'>{row['message_id']}</td>"
                # Category
                f"<td style='padding:10px 12px;border-bottom:1px solid rgba(255,255,255,0.04);'>"
                f"<span style='display:flex;align-items:center;gap:6px;'>"
                f"<span>{icon}</span>"
                f"<span style='color:#cbd5e1;font-size:0.82rem;'>{row['category']}</span>"
                f"</span></td>"
                # Priority
                f"<td style='padding:10px 12px;border-bottom:1px solid rgba(255,255,255,0.04);'>"
                f"<span style='background:rgba({','.join(str(int(pri_colour.lstrip('#')[i:i+2],16)) for i in (0,2,4))},0.15);"
                f"color:{pri_colour};border:1px solid {pri_colour}44;border-radius:999px;"
                f"padding:2px 10px;font-size:0.75rem;font-weight:700;'>{pri}</span></td>"
                # Needs Human
                f"<td style='padding:10px 12px;border-bottom:1px solid rgba(255,255,255,0.04);'>{human_html}</td>"
                # Confidence
                f"<td style='padding:10px 12px;border-bottom:1px solid rgba(255,255,255,0.04);min-width:120px;'>"
                f"<div style='display:flex;align-items:center;gap:6px;'>"
                f"<div style='flex:1;background:rgba(255,255,255,0.08);border-radius:99px;height:5px;'>"
                f"<div style='width:{conf_pct}%;background:{conf_colour};border-radius:99px;height:5px;'></div></div>"
                f"<span style='font-size:0.8rem;color:{conf_colour};font-weight:600;min-width:30px'>{conf_pct}%</span>"
                f"</div></td>"
                # Summary
                f"<td style='padding:10px 12px;color:#94a3b8;font-size:0.82rem;max-width:320px;"
                f"border-bottom:1px solid rgba(255,255,255,0.04);'>{summary_short}</td>"
                # Tokens
                f"<td style='padding:10px 12px;text-align:right;font-size:0.8rem;color:#64748b;"
                f"font-family:monospace;border-bottom:1px solid rgba(255,255,255,0.04);white-space:nowrap;'>{tok_display}</td>"
                # Latency
                f"<td style='padding:10px 12px;text-align:right;font-size:0.8rem;color:#64748b;"
                f"font-family:monospace;border-bottom:1px solid rgba(255,255,255,0.04);white-space:nowrap;'>{lat_display}</td>"
                "</tr>"
            )

        table_html = header_html + rows_html + "</tbody></table>"

        st.markdown(
            f'<div style="overflow-x:auto;border:1px solid rgba(99,102,241,0.2);border-radius:12px;">'
            f'{table_html}</div>',
            unsafe_allow_html=True,
        )

    # ── Detail expanders ─────────────────────────────────────────────────────
    if not filtered.empty:
        st.markdown("### 🔎 Message Details")
        for _, row in filtered.iterrows():
            icon = category_icon(row["category"])
            label_colour = "#fbbf24" if row["needs_human"] else "#4ade80"
            label_text = "Needs Human Review" if row["needs_human"] else "Auto-resolved"
            with st.expander(
                f"{icon} **{row['message_id']}** — {row['category']} · {row['priority']} "
                f"· :{('orange' if row['needs_human'] else 'green')}[{label_text}]"
            ):
                dcol1, dcol2 = st.columns([3, 2])
                with dcol1:
                    st.markdown("**Raw Message**")
                    raw = row["raw_text"] or "_\\<empty\\>_"
                    st.markdown(
                        f"<div style='background:rgba(15,17,23,0.8);border:1px solid rgba(99,102,241,0.2);"
                        f"border-radius:8px;padding:12px 16px;color:#cbd5e1;font-size:0.88rem;"
                        f"line-height:1.6;'>{raw}</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown("**Summary**")
                    st.markdown(
                        f"<div style='background:rgba(99,102,241,0.06);border:1px solid rgba(99,102,241,0.2);"
                        f"border-radius:8px;padding:12px 16px;color:#a5b4fc;font-size:0.88rem;"
                        f"line-height:1.6;'>{row['summary']}</div>",
                        unsafe_allow_html=True,
                    )
                with dcol2:
                    st.markdown("**Suggested Action**")
                    st.markdown(
                        f"<div style='background:rgba(74,222,128,0.04);border:1px solid rgba(74,222,128,0.15);"
                        f"border-radius:8px;padding:12px 16px;color:#86efac;font-size:0.88rem;"
                        f"line-height:1.6;'>{row['suggested_action']}</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown("**Token Usage**")
                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("Prompt",     f"{row['prompt_tokens']:,}")
                    mc2.metric("Completion", f"{row['completion_tokens']:,}")
                    mc3.metric("Total",      f"{row['total_tokens']:,}")
                    st.metric("Latency", f"{row['latency_s']:.3f}s")

    # ── Performance charts ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📈 Performance Overview")
    ch1, ch2 = st.columns(2)
    with ch1:
        st.markdown("**Latency per Message (seconds)**")
        st.plotly_chart(make_latency_bar(df), use_container_width=True, config={"displayModeBar": False})
    with ch2:
        st.markdown("**Token Usage per Message**")
        st.plotly_chart(make_token_bar(df), use_container_width=True, config={"displayModeBar": False})

    # ── Footer ───────────────────────────────────────────────────────────────
    st.markdown("""
    <hr>
    <p style="text-align:center;color:#334155;font-size:0.78rem;padding-bottom:12px;">
      Triage Classifier &nbsp;·&nbsp; Groq &amp; LangGraph &nbsp;·&nbsp; llama-3.1-8b-instant
    </p>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
