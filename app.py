"""Sales Forecasting Studio - main entry point.

Run with:  streamlit run app.py
"""
import streamlit as st

# set_page_config must be the first Streamlit command.
st.set_page_config(page_title="Sales Forecasting Studio", page_icon="📈", layout="wide")

import auth
import data_utils
import model_utils
import ui
from views import analysis, forecasting, overview, performance

PAGES = {
    "📊  Sales Overview": overview.render,
    "🔮  Sales Forecasting": forecasting.render,
    "🎯  Model Performance": performance.render,
    "🔎  Data Analysis": analysis.render,
}


# ---------------------------------------------------------------- login page
def login_page():
    _, middle, _ = st.columns([1, 1.3, 1])
    with middle:
        st.write("")
        st.markdown('<div class="login-title">📈 Sales Forecasting Studio</div>', unsafe_allow_html=True)
        st.markdown('<div class="login-sub">Log in or create an account to continue</div>',
                    unsafe_allow_html=True)
        tab_login, tab_signup = st.tabs(["Log in", "Create account"])

        with tab_login:
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Log in")
            if submitted:
                ok, result = auth.verify_user(username, password)
                if ok:
                    st.session_state["user"] = result
                    st.rerun()
                else:
                    st.error(result)

        with tab_signup:
            with st.form("signup_form"):
                full_name = st.text_input("Full name")
                new_username = st.text_input("Choose a username")
                email = st.text_input("Email (optional)")
                new_password = st.text_input("Password (min. 8 characters)", type="password")
                confirm = st.text_input("Confirm password", type="password")
                created = st.form_submit_button("Create account")
            if created:
                if new_password != confirm:
                    st.error("Passwords do not match.")
                else:
                    ok, message = auth.create_user(new_username, full_name, email, new_password)
                    (st.success if ok else st.error)(message)


# ------------------------------------------------------------ data selection
@st.cache_data(show_spinner=False)
def _parse_upload(file_bytes: bytes):
    return data_utils.read_csv_bytes(file_bytes)


def sidebar_data():
    """Sidebar controls for uploading a CSV. Returns (daily_df, report, source_label)."""
    st.sidebar.markdown("### Data")
    st.session_state["currency"] = st.sidebar.text_input("Currency symbol (optional)", value="$", max_chars=3)
    uploaded = st.sidebar.file_uploader("Upload sales CSV", type=["csv"])

    if uploaded is None:
        raw = data_utils.generate_sample_data()
        date_col, sales_col, dayfirst, label = "Date", "Sales", False, "Sample data (synthetic demo)"
        st.sidebar.caption("Showing built-in sample data. Upload your CSV to use your own.")
    else:
        try:
            raw = _parse_upload(uploaded.getvalue())
        except Exception as exc:
            st.sidebar.error(f"Could not read that file: {exc}")
            return None, None, None
        cols = list(raw.columns)
        date_col = st.sidebar.selectbox("Date column", cols, index=data_utils.guess_column(cols, data_utils.DATE_HINTS))
        sales_col = st.sidebar.selectbox("Sales column", cols, index=data_utils.guess_column(cols, data_utils.SALES_HINTS, 1))
        dayfirst = st.sidebar.checkbox("Dates are day-first (DD/MM/YYYY)")
        label = uploaded.name
        st.sidebar.caption("If a date appears on several rows, those rows are added into one daily total.")

    try:
        daily, report = data_utils.prepare_daily(raw, date_col, sales_col, dayfirst)
    except Exception as exc:
        st.sidebar.error(str(exc))
        return None, None, None
    return daily, report, label


# ----------------------------------------------------------------- main app
def main_app():
    ui.inject_css()
    user = st.session_state["user"]

    st.sidebar.markdown("## 📈 Sales Forecasting")
    st.sidebar.caption(f"Signed in as **{user['full_name']}**")
    if st.sidebar.button("Log out"):
        st.session_state["user"] = None
        st.rerun()
    st.sidebar.divider()
    page = st.sidebar.radio("Navigation", list(PAGES), label_visibility="collapsed")
    st.sidebar.divider()

    # Load the model and make sure we can build every column it needs.
    try:
        model = model_utils.load_model()
    except Exception as exc:
        st.error(f"Could not load sales_forecasting_model.pkl: {exc}")
        st.stop()
    unsupported = model_utils.check_features(model)
    if unsupported:
        st.error(f"The model expects columns this app cannot build from a date: {unsupported}")
        st.stop()

    daily, report, label = sidebar_data()
    if daily is None:
        st.warning("Fix the data settings in the sidebar to continue.")
        st.stop()

    st.caption(f"Data source: **{label}**")
    PAGES[page](daily, model, report)


# ---------------------------------------------------------------- entry point
auth.init_db()
st.session_state.setdefault("user", None)

if st.session_state["user"] is None:
    ui.inject_css()
    login_page()
else:
    main_app()
