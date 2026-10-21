"""All Plotly charts live here so the pages stay short and readable."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

PRIMARY = "#4F46E5"
CYAN = "#06B6D4"
AMBER = "#F59E0B"
GREEN = "#10B981"
RED = "#EF4444"
GREY = "#94A3B8"
GRID = "#E2E8F0"

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _style(fig, title: str, height: int = 380):
    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, x=0.01, font=dict(size=16, color="#0F172A")),
        height=height,
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def sales_trend(daily: pd.DataFrame, ma_days: int = 7):
    fig = go.Figure()
    fig.add_scatter(x=daily["Date"], y=daily["Sales"], name="Daily sales",
                    mode="lines", line=dict(color="#C7D2FE", width=1.2))
    ma = daily["Sales"].rolling(ma_days, min_periods=1).mean()
    fig.add_scatter(x=daily["Date"], y=ma, name=f"{ma_days}-day average",
                    mode="lines", line=dict(color=PRIMARY, width=2.8))
    fig.update_xaxes(rangeslider_visible=True)
    return _style(fig, "Sales trend", 440)


def monthly_sales(daily: pd.DataFrame):
    monthly = daily.set_index("Date")["Sales"].resample("MS").sum().reset_index()
    fig = go.Figure(go.Bar(x=monthly["Date"], y=monthly["Sales"], marker_color=PRIMARY, name="Sales"))
    return _style(fig, "Monthly sales", 360)


def weekday_sales(daily: pd.DataFrame):
    avg = daily.groupby("DayOfWeek")["Sales"].mean().reindex(range(7))
    fig = go.Figure(go.Bar(x=WEEKDAYS, y=avg.values, marker_color=CYAN, name="Average sales"))
    fig.update_layout(hovermode="closest")
    return _style(fig, "Average sales by weekday", 360)


def actual_vs_predicted(df: pd.DataFrame):
    fig = go.Figure()
    fig.add_scatter(x=df["Date"], y=df["Sales"], name="Actual",
                    mode="lines", line=dict(color=PRIMARY, width=2))
    fig.add_scatter(x=df["Date"], y=df["Predicted"], name="Predicted",
                    mode="lines", line=dict(color=AMBER, width=2, dash="dot"))
    fig.update_xaxes(rangeslider_visible=True)
    return _style(fig, "Actual vs predicted sales", 440)


def actual_vs_predicted_scatter(df: pd.DataFrame):
    lo = float(min(df["Sales"].min(), df["Predicted"].min()))
    hi = float(max(df["Sales"].max(), df["Predicted"].max()))
    fig = go.Figure()
    fig.add_scatter(x=df["Sales"], y=df["Predicted"], mode="markers", name="Days",
                    marker=dict(color=PRIMARY, opacity=0.45, size=6))
    fig.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect prediction",
                    line=dict(color=GREY, dash="dash"))
    fig.update_xaxes(title="Actual", showgrid=True, gridcolor=GRID)
    fig.update_yaxes(title="Predicted")
    fig.update_layout(hovermode="closest")
    return _style(fig, "Predicted vs actual (closer to the line = better)", 380)


def residual_histogram(df: pd.DataFrame):
    errors = df["Sales"] - df["Predicted"]
    fig = go.Figure(go.Histogram(x=errors, nbinsx=40, marker_color=PRIMARY, opacity=0.85))
    fig.update_xaxes(title="Error (actual - predicted)")
    fig.update_layout(hovermode="closest")
    return _style(fig, "Prediction error distribution", 380)


def feature_importance_chart(imp: pd.DataFrame):
    imp = imp.sort_values("Importance")
    fig = go.Figure(go.Bar(x=imp["Importance"], y=imp["Feature"], orientation="h", marker_color=GREEN))
    fig.update_layout(hovermode="closest")
    return _style(fig, "What the model relies on (feature importance)", 340)


def forecast_chart(history: pd.DataFrame, forecast: pd.DataFrame, show_band: bool = True):
    fig = go.Figure()
    fig.add_scatter(x=history["Date"], y=history["Sales"], name="Actual sales",
                    mode="lines", line=dict(color=PRIMARY, width=2.2))
    if show_band and forecast["Lower"].notna().all():
        fig.add_scatter(x=forecast["Date"], y=forecast["Upper"], mode="lines",
                        line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=forecast["Date"], y=forecast["Lower"], mode="lines",
                        line=dict(width=0), fill="tonexty", fillcolor="rgba(245,158,11,0.18)",
                        name="Likely range (10th-90th pct of trees)", hoverinfo="skip")
    fig.add_scatter(x=forecast["Date"], y=forecast["Forecast"], name="Forecast",
                    mode="lines", line=dict(color=AMBER, width=2.8))
    start = forecast["Date"].iloc[0]
    fig.add_shape(type="line", x0=start, x1=start, y0=0, y1=1, yref="paper",
                  line=dict(color=GREY, dash="dash", width=1))
    fig.add_annotation(x=start, y=1.0, yref="paper", text="Forecast starts", showarrow=False,
                       xanchor="left", yanchor="bottom", font=dict(color=GREY, size=11))
    return _style(fig, "Sales forecast", 470)


def histogram(daily: pd.DataFrame):
    fig = go.Figure(go.Histogram(x=daily["Sales"], nbinsx=40, marker_color=CYAN, opacity=0.9))
    fig.update_xaxes(title="Daily sales")
    fig.update_layout(hovermode="closest")
    return _style(fig, "Distribution of daily sales", 360)


def weekday_box(daily: pd.DataFrame):
    fig = go.Figure()
    for i, name in enumerate(WEEKDAYS):
        values = daily.loc[daily["DayOfWeek"] == i, "Sales"]
        if len(values):
            fig.add_trace(go.Box(y=values, name=name, marker_color=PRIMARY, showlegend=False))
    fig.update_layout(hovermode="closest")
    return _style(fig, "Sales spread by weekday", 360)


def month_weekday_heatmap(daily: pd.DataFrame):
    pivot = daily.pivot_table(index="Month", columns="DayOfWeek", values="Sales", aggfunc="mean")
    pivot = pivot.reindex(index=range(1, 13), columns=range(7))
    fig = go.Figure(go.Heatmap(z=pivot.values, x=WEEKDAYS, y=MONTHS, colorscale="Purples",
                               colorbar=dict(title="Avg sales")))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(hovermode="closest")
    return _style(fig, "Average sales: month x weekday", 420)


def yearly_comparison(daily: pd.DataFrame):
    pivot = daily.pivot_table(index="Month", columns="Year", values="Sales", aggfunc="mean")
    pivot = pivot.reindex(range(1, 13))
    fig = go.Figure()
    palette = [PRIMARY, AMBER, GREEN, CYAN, RED, GREY]
    for i, year in enumerate(pivot.columns):
        fig.add_scatter(x=MONTHS, y=pivot[year], name=str(year), mode="lines+markers",
                        line=dict(color=palette[i % len(palette)], width=2.5))
    return _style(fig, "Average daily sales by month, per year", 360)
