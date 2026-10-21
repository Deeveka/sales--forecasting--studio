# Sales Forecasting Studio

A Streamlit dashboard built around your trained `sales_forecasting_model.pkl`
(a scikit-learn `RandomForestRegressor`).

## What the model expects

| Column      | Meaning                              |
|-------------|--------------------------------------|
| `Day`       | Day of the month (1-31)              |
| `Month`     | Month number (1-12)                  |
| `Year`      | 4-digit year                         |
| `DayOfWeek` | Monday = 0 ... Sunday = 6            |

It returns the predicted **sales for that one day**. Your CSV does not need these
columns: the app builds them from your Date column.

## Your CSV needs

* a **date column** (e.g. `2024-03-15`)
* a **sales column** (numbers; `$` and `,` are cleaned automatically)

Extra columns are ignored. Several rows per date are added into one daily total.
Try `sample_sales_data.csv` (synthetic demo data) to see the format.

## Run it

1. Install Python 3.10 or newer.
2. Open a terminal in this folder and (recommended) create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   ```
3. Install the packages:

   ```bash
   pip install -r requirements.txt
   ```
4. Start the app:

   ```bash
   streamlit run app.py
   ```
5. Your browser opens at http://localhost:8501. Click **Create account**, then log in.

## Project layout

```
app.py              main file: login page, sidebar, navigation
auth.py             sign up / log in (SQLite file users.db, hashed passwords)
data_utils.py       CSV cleaning + building Day/Month/Year/DayOfWeek
model_utils.py      loads the model, predicts, forecasts, metrics
charts.py           all Plotly charts
ui.py               CSS styling and KPI cards
views/              one file per page (overview, forecasting, performance, analysis)
.streamlit/config.toml   theme
```

## Good to know

* Keep `scikit-learn==1.6.1`: the model was saved with that version.
* Only load `.pkl` files you trust (pickle files can run code when opened).
* `users.db` is created automatically. Delete it to reset all accounts.
* Login is meant for a personal/demo app. For a public deployment, use a managed
  auth service and HTTPS.
