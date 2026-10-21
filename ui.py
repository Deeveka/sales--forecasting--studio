"""Shared look-and-feel: CSS, KPI cards and small display helpers."""
import pandas as pd
import streamlit as st

CSS = """
<style>
.block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1300px;}
#MainMenu, footer {visibility: hidden;}

[data-testid="stSidebar"] {background: #0F172A;}
[data-testid="stSidebar"] * {color: #E2E8F0;}
[data-testid="stSidebar"] hr {border-color: #1E293B;}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {background: #1E293B;}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {color: #CBD5E1;}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {color: #0F172A;}

.page-title {font-size: 1.9rem; font-weight: 700; color: #0F172A; margin-bottom: 0;}
.page-sub {color: #64748B; margin-top: 0.1rem; margin-bottom: 1.4rem;}

.kpi-card {background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 14px;
           padding: 1rem 1.2rem; box-shadow: 0 1px 3px rgba(15,23,42,0.05); height: 100%;}
.kpi-label {color: #64748B; font-size: 0.8rem; font-weight: 600;
            text-transform: uppercase; letter-spacing: 0.04em;}
.kpi-value {color: #0F172A; font-size: 1.7rem; font-weight: 700; margin-top: 0.15rem;}
.kpi-delta {font-size: 0.82rem; font-weight: 600; margin-top: 0.2rem;}
.kpi-up {color: #059669;}
.kpi-down {color: #DC2626;}
.kpi-neutral {color: #64748B;}

.login-title {text-align: center; font-size: 2rem; font-weight: 800; color: #0F172A;}
.login-sub {text-align: center; color: #64748B; margin-bottom: 1.2rem;}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = ""):
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="page-sub">{subtitle}</div>', unsafe_allow_html=True)


def fmt_money(value: float) -> str:
    """Format a number with thousands separators and the chosen currency symbol."""
    symbol = st.session_state.get("currency", "")
    if pd.isna(value):
        return "n/a"
    return f"{symbol}{value:,.0f}"


def kpi_card(label: str, value: str, delta: str = "", tone: str = "neutral"):
    """tone: 'up' (green), 'down' (red) or 'neutral' (grey)."""
    delta_html = f'<div class="kpi-delta kpi-{tone}">{delta}</div>' if delta else ""
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>{delta_html}</div>',
        unsafe_allow_html=True,
    )


def kpi_row(items):
    """items: list of dicts with keys label, value, and optional delta / tone."""
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        with col:
            kpi_card(item["label"], item["value"], item.get("delta", ""), item.get("tone", "neutral"))
    st.write("")


def show_chart(fig):
    """Display a Plotly chart (works across Streamlit versions)."""
    try:
        st.plotly_chart(fig, width="stretch")
    except Exception:
        st.plotly_chart(fig, use_container_width=True)


def show_table(df: pd.DataFrame, **kwargs):
    """Display a DataFrame (works across Streamlit versions)."""
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kwargs)
    except Exception:
        st.dataframe(df, use_container_width=True, hide_index=True, **kwargs)
