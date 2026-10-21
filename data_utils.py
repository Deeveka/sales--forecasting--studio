"""Loading, cleaning and feature engineering for the sales CSV."""
import io

import numpy as np
import pandas as pd

DATE_HINTS = ["date", "order_date", "orderdate", "timestamp", "day", "time"]
SALES_HINTS = ["sales", "sale", "revenue", "amount", "total", "turnover"]


def read_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    """Read an uploaded CSV (tries UTF-8 first, then Latin-1)."""
    try:
        return pd.read_csv(io.BytesIO(file_bytes))
    except UnicodeDecodeError:
        return pd.read_csv(io.BytesIO(file_bytes), encoding="latin-1")


def guess_column(columns, hints, fallback_index=0):
    """Return the index of the column whose name best matches the hints."""
    lowered = [str(c).lower().strip() for c in columns]
    for hint in hints:
        for i, name in enumerate(lowered):
            if name == hint:
                return i
    for hint in hints:
        for i, name in enumerate(lowered):
            if hint in name:
                return i
    return min(fallback_index, len(columns) - 1)


def add_calendar_features(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """Add the four columns the model expects, built from the date.

    Day       -> day of the month (1-31)
    Month     -> month number (1-12)
    Year      -> 4-digit year
    DayOfWeek -> Monday=0 ... Sunday=6  (pandas' dt.dayofweek)
    """
    df = df.copy()
    df["Day"] = df[date_col].dt.day
    df["Month"] = df[date_col].dt.month
    df["Year"] = df[date_col].dt.year
    df["DayOfWeek"] = df[date_col].dt.dayofweek
    return df


def prepare_daily(raw: pd.DataFrame, date_col: str, sales_col: str, dayfirst: bool = False):
    """Turn a raw CSV into one row per day: Date, Sales + model features.

    Returns (daily_df, report_dict). Raises ValueError if nothing usable is left.
    """
    work = pd.DataFrame(
        {
            "Date": pd.to_datetime(raw[date_col], errors="coerce", dayfirst=dayfirst),
            "Sales": raw[sales_col],
        }
    )
    rows_in = len(work)

    # Clean the sales column ("$1,200.50" -> 1200.5)
    if not pd.api.types.is_numeric_dtype(work["Sales"]):
        cleaned = work["Sales"].astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
        work["Sales"] = pd.to_numeric(cleaned, errors="coerce")

    bad_dates = int(work["Date"].isna().sum())
    bad_sales = int(work["Sales"].isna().sum())
    work = work.dropna(subset=["Date", "Sales"])
    if work.empty:
        raise ValueError(
            "No usable rows found. Check that the Date and Sales columns are chosen correctly."
        )

    work["Date"] = work["Date"].dt.normalize()
    rows_used = len(work)

    # If there are several rows per date (e.g. one per transaction), add them up.
    daily = work.groupby("Date", as_index=False)["Sales"].sum().sort_values("Date")
    daily = add_calendar_features(daily).reset_index(drop=True)

    expected_days = (daily["Date"].max() - daily["Date"].min()).days + 1
    report = {
        "rows_in": rows_in,
        "rows_used": rows_used,
        "bad_dates": bad_dates,
        "bad_sales": bad_sales,
        "days": len(daily),
        "missing_days": int(expected_days - len(daily)),
        "merged_rows": int(rows_used - len(daily)),
        "date_min": daily["Date"].min(),
        "date_max": daily["Date"].max(),
    }
    return daily, report


def generate_sample_data(start="2022-01-01", end="2023-12-31", seed=42) -> pd.DataFrame:
    """Synthetic demo data so the app works before you upload anything."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    month_effect = {1: -600, 2: -500, 3: -200, 4: 0, 5: 200, 6: 300,
                    7: 100, 8: 200, 9: 400, 10: 800, 11: 1400, 12: 1800}
    weekday_effect = [0, 100, 150, 200, 400, 900, 700]
    sales = [
        8000
        + month_effect[d.month]
        + weekday_effect[d.dayofweek]
        + (d.year - 2022) * 500
        + rng.normal(0, 450)
        for d in dates
    ]
    return pd.DataFrame({"Date": dates.strftime("%Y-%m-%d"), "Sales": np.round(sales, 2)})
