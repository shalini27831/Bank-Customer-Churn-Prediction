import os
import joblib
import streamlit as st
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="wide"
)


# ==========================================================
# CUSTOM CSS  (design only)
# ==========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --ink: #0E2A33;
    --ink-soft: #4C6670;
    --paper: #F2F6F7;
    --card: #FFFFFF;
    --line: #DCE6E9;
    --brand: #0F5C63;
    --brand-deep: #0A3F45;
    --risk: #C23B2E;
    --risk-bg: #FCEDEB;
    --safe: #1D7A55;
    --safe-bg: #E8F5EF;
}

html, body, [class*="css"], .stApp {
    font-family: 'Manrope', sans-serif;
    color: var(--ink);
}

.stApp {
    background: var(--paper);
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header[data-testid="stHeader"] {
    visibility: hidden;
    height: 0;
}

.block-container {
    padding-top: 1.8rem;
    padding-bottom: 3rem;
    max-width: 1150px;
}

/* ---------- Header band ---------- */
.hero {
    background: linear-gradient(120deg, var(--brand-deep) 0%, var(--brand) 100%);
    border-radius: 18px;
    padding: 34px 40px;
    color: #fff;
    margin-bottom: 26px;
}

.hero .main-title {
    font-size: 34px;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin: 0 0 6px 0;
    line-height: 1.15;
}

.hero .subtitle {
    font-size: 16px;
    font-weight: 500;
    color: #BFE0E3;
    margin: 0;
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: var(--card);
    border-right: 1px solid var(--line);
}

section[data-testid="stSidebar"] h2 {
    font-size: 20px;
    font-weight: 800;
    color: var(--ink);
}

section[data-testid="stSidebar"] label p {
    font-weight: 600;
    font-size: 14px;
    color: var(--ink-soft);
}

section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
section[data-testid="stSidebar"] div[data-baseweb="input"],
section[data-testid="stSidebar"] div[data-baseweb="base-input"] {
    border-radius: 10px;
}

/* ---------- Sidebar no longer used ---------- */
section[data-testid="stSidebar"],
div[data-testid="stSidebarCollapsedControl"],
button[data-testid="stExpandSidebarButton"] {
    display: none;
}

/* ---------- Form card ---------- */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 12px 18px;
}

div[data-testid="stVerticalBlockBorderWrapper"] label p {
    font-weight: 600;
    font-size: 14px;
    color: var(--ink-soft);
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"],
div[data-baseweb="base-input"] {
    border-radius: 10px;
}

/* ---------- Font fallback ---------- */
html, body, .stApp, .stMarkdown, p, label, input, textarea, button,
div[data-baseweb="select"], div[data-baseweb="input"] {
    font-family: 'Manrope', 'Segoe UI', system-ui, -apple-system, sans-serif !important;
}

/* ---------- Input cards ---------- */
div[data-testid="stVerticalBlockBorderWrapper"] {
    padding: 18px 22px 20px 22px;
    border-top: 4px solid var(--brand);
    box-shadow: 0 6px 18px rgba(14, 42, 51, 0.06);
}

.card-head {
    display: flex;
    align-items: center;
    gap: 12px;
    padding-bottom: 14px;
    margin-bottom: 8px;
    border-bottom: 1px solid var(--line);
}

.card-icon {
    width: 42px;
    height: 42px;
    border-radius: 12px;
    background: #E3F1F2;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    flex-shrink: 0;
}

.card-title {
    font-size: 17px;
    font-weight: 800;
    line-height: 1.2;
}

.card-note {
    font-size: 13px;
    color: var(--ink-soft);
}

/* Labels */
div[data-testid="stVerticalBlockBorderWrapper"] label p {
    font-size: 15px;
    font-weight: 700;
    color: var(--ink);
}

/* Inputs and dropdowns */
div[data-testid="stNumberInput"] div[data-baseweb="input"],
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background: #F7FAFB;
    border: 1.5px solid var(--line);
    border-radius: 12px;
    min-height: 48px;
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
}

div[data-testid="stNumberInput"] div[data-baseweb="input"] > div,
div[data-testid="stNumberInput"] div[data-baseweb="base-input"] {
    background: transparent;
    border: none;
}

div[data-testid="stNumberInput"] input {
    font-size: 16px;
    font-weight: 600;
    color: var(--ink);
}

div[data-testid="stSelectbox"] div[data-baseweb="select"] div {
    font-size: 16px;
    font-weight: 600;
    color: var(--ink);
}

div[data-testid="stNumberInput"] div[data-baseweb="input"]:hover,
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:hover {
    border-color: #9CC6CA;
}

div[data-testid="stNumberInput"] div[data-baseweb="input"]:focus-within,
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:focus-within {
    border-color: var(--brand);
    background: #fff;
    box-shadow: 0 0 0 3px rgba(15, 92, 99, 0.15);
}

div[data-testid="stNumberInput"] button {
    border-radius: 8px;
    color: var(--brand);
}

/* ============ Premium form look ============ */

/* Outer box */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(180deg, #FFFFFF 0%, #F9FCFC 100%);
    border: 1px solid #D5E3E6;
    border-top: 5px solid var(--brand);
    border-radius: 20px;
    padding: 22px 24px 24px 24px;
    box-shadow: 0 12px 32px rgba(15, 92, 99, 0.10);
}

/* Panel headers */
.card-head {
    border-bottom: none;
    padding-bottom: 6px;
    margin-bottom: 10px;
}

.card-icon {
    width: 46px;
    height: 46px;
    border-radius: 14px;
    font-size: 22px;
    box-shadow: 0 4px 10px rgba(14, 42, 51, 0.10);
}

.card-title { font-size: 18px; font-weight: 800; }
.card-note  { font-size: 13px; font-weight: 500; }

.card-head.tone-teal   .card-icon { background: #BFE3E6; }
.card-head.tone-teal   .card-title { color: #0A3F45; }
.card-head.tone-indigo .card-icon { background: #CBD5F7; }
.card-head.tone-indigo .card-title { color: #2B3A8C; }
.card-head.tone-amber  .card-icon { background: #FADFA5; }
.card-head.tone-amber  .card-title { color: #8A5200; }

/* Labels: darker and clearer */
div[data-testid="stVerticalBlockBorderWrapper"] label p {
    font-size: 15px;
    font-weight: 700;
    color: #0E2A33;
    letter-spacing: 0.1px;
}

/* White field boxes */
div[data-testid="stNumberInputContainer"],
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background: #FFFFFF;
    border: 1.5px solid #CBDCE0;
    border-radius: 12px;
    min-height: 50px;
    box-shadow: 0 1px 2px rgba(14, 42, 51, 0.05);
    overflow: hidden;
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
}

/* Number input: one unified box, +/- inside it */
div[data-testid="stNumberInput"] div[data-baseweb="input"],
div[data-testid="stNumberInput"] div[data-baseweb="base-input"] {
    background: transparent;
    border: none;
    box-shadow: none;
    min-height: 0;
}

div[data-testid="stNumberInput"] input {
    font-size: 17px;
    font-weight: 700;
    color: #0A3F45;
}

div[data-testid="stNumberInputContainer"] button {
    background: #EEF5F6;
    color: var(--brand);
    border: none;
    border-left: 1px solid #DDE9EB;
    border-radius: 0;
    height: 50px;
    width: 40px;
}

div[data-testid="stNumberInputContainer"] button:hover {
    background: var(--brand);
    color: #FFFFFF;
}

/* Dropdown text */
div[data-testid="stSelectbox"] div[data-baseweb="select"] div {
    font-size: 17px;
    font-weight: 700;
    color: #0A3F45;
}

div[data-testid="stSelectbox"] svg { color: var(--brand); }

/* Hover and focus */
div[data-testid="stNumberInputContainer"]:hover,
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:hover {
    border-color: #7DB9BE;
}

div[data-testid="stNumberInputContainer"]:focus-within,
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:focus-within {
    border-color: var(--brand);
    box-shadow: 0 0 0 4px rgba(15, 92, 99, 0.16);
}

/* ---------- Top accent bar (works on any Streamlit version) ---------- */
.form-topbar {
    height: 7px;
    border-radius: 999px;
    background: linear-gradient(90deg, #0F5C63 0%, #3F51B5 50%, #E0A026 100%);
    margin: 2px 0 18px 0;
}

/* ---------- Colored panels (match the inner block that holds one header) ---------- */
div[data-testid="stVerticalBlock"]:has(.tone-teal):not(:has(.tone-indigo)):not(:has(.tone-amber)) {
    background: #EAF5F6;
    border: 1px solid #CFE6E8;
    border-radius: 16px;
    padding: 16px 16px 12px 16px;
}

div[data-testid="stVerticalBlock"]:has(.tone-indigo):not(:has(.tone-teal)):not(:has(.tone-amber)) {
    background: #EEF1FC;
    border: 1px solid #D6DDF6;
    border-radius: 16px;
    padding: 16px 16px 12px 16px;
}

div[data-testid="stVerticalBlock"]:has(.tone-amber):not(:has(.tone-teal)):not(:has(.tone-indigo)) {
    background: #FFF6E4;
    border: 1px solid #F5E1B5;
    border-radius: 16px;
    padding: 16px 16px 12px 16px;
}

/* ---------- Header banners (tinted even if panels don't match) ---------- */
.card-head {
    padding: 12px 14px;
    border-radius: 14px;
    margin-bottom: 12px;
}

.card-head.tone-teal   { background: #D3EBED; }
.card-head.tone-indigo { background: #DCE2FA; }
.card-head.tone-amber  { background: #FBE8BC; }

.card-head .card-icon {
    background: #FFFFFF;
}

/* ---------- Form box: own teal outline (no native grey border) ---------- */
div[data-testid="stVerticalBlock"]:has(.form-topbar):not(:has(.hero)) {
    background: linear-gradient(180deg, #FFFFFF 0%, #F7FBFB 100%);
    border: 1.5px solid #9CC6CA;
    border-radius: 20px;
    padding: 22px 24px 24px 24px;
    box-shadow: 0 12px 32px rgba(15, 92, 99, 0.12);
}

/* ---------- Equal-height panels ---------- */
div[data-testid="stHorizontalBlock"]:has(.tone-teal):has(.tone-indigo):has(.tone-amber) {
    align-items: stretch !important;
}

div[data-testid="stHorizontalBlock"]:has(.tone-teal):has(.tone-indigo):has(.tone-amber) > div {
    display: flex;
    flex-direction: column;
}

div[data-testid="stHorizontalBlock"]:has(.tone-teal):has(.tone-indigo):has(.tone-amber) > div > div[data-testid="stVerticalBlock"],
div[data-testid="stHorizontalBlock"]:has(.tone-teal):has(.tone-indigo):has(.tone-amber) > div > div > div[data-testid="stVerticalBlock"] {
    flex: 1 1 auto;
    height: 100%;
}

/* ---------- "Press Enter to apply" hint: neat pill on the label row ---------- */
div[data-testid="stNumberInput"] {
    position: relative;
}

div[data-testid="stNumberInputContainer"],
div[data-testid="stNumberInput"] div[data-baseweb="input"],
div[data-testid="stNumberInput"] div[data-baseweb="base-input"] {
    position: static !important;
    overflow: visible;
}

div[data-testid="stNumberInputContainer"] button:last-of-type {
    border-radius: 0 10px 10px 0;
}

div[data-testid="InputInstructions"] {
    position: absolute !important;
    top: 0 !important;
    right: 0 !important;
    bottom: auto !important;
    left: auto !important;
    z-index: 5;
    pointer-events: none;
    font-size: 0 !important;
    line-height: 1;
}

div[data-testid="InputInstructions"] * {
    display: none;
}

div[data-testid="InputInstructions"]::after {
    content: "↵  Enter to apply";
    display: inline-block;
    font-size: 11px;
    font-weight: 700;
    color: var(--brand);
    background: #E3F1F2;
    border: 1px solid #BBDDE0;
    border-radius: 999px;
    padding: 3px 10px;
    box-shadow: 0 2px 6px rgba(15, 92, 99, 0.12);
}

/* ---------- Section heading ---------- */
.section-title {
    font-size: 20px;
    font-weight: 800;
    margin: 6px 0 4px 0;
}

.section-note {
    color: var(--ink-soft);
    font-size: 15px;
    margin-bottom: 14px;
}

/* ---------- Button ---------- */
div.stButton > button {
    background: var(--brand);
    color: #fff;
    border: none;
    border-radius: 12px;
    padding: 0.85rem 1rem;
    font-family: 'Manrope', sans-serif;
    font-size: 17px;
    font-weight: 700;
    transition: background 0.15s ease;
}

div.stButton > button:hover {
    background: var(--brand-deep);
    color: #fff;
    border: none;
}

div.stButton > button:focus-visible {
    outline: 3px solid #7CC4C9;
    outline-offset: 2px;
}

/* ---------- Result card ---------- */
.result-box {
    display: flex;
    align-items: center;
    gap: 34px;
    padding: 30px 34px;
    border-radius: 18px;
    background: var(--card);
    border: 1px solid var(--line);
    border-left: 8px solid var(--accent);
    margin: 18px 0 22px 0;
}

.result-box.churn { --accent: var(--risk); }
.result-box.stay  { --accent: var(--safe); }

.gauge {
    position: relative;
    width: 150px;
    height: 150px;
    flex-shrink: 0;
}

.gauge svg { transform: rotate(-90deg); }

.gauge .gauge-value {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 30px;
    font-weight: 800;
    color: var(--accent);
}

.result-text .verdict {
    font-size: 26px;
    font-weight: 800;
    line-height: 1.2;
    margin-bottom: 6px;
    color: var(--accent);
}

.result-text .detail {
    font-size: 16px;
    color: var(--ink-soft);
    margin-bottom: 14px;
}

.pill {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 999px;
    font-size: 14px;
    font-weight: 700;
}

.pill.churn { background: var(--risk-bg); color: var(--risk); }
.pill.stay  { background: var(--safe-bg); color: var(--safe); }

/* ---------- Probability cards ---------- */
.prob-card {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 20px 24px;
}

.prob-card .prob-label {
    font-size: 15px;
    font-weight: 600;
    color: var(--ink-soft);
}

.prob-card .prob-value {
    font-size: 32px;
    font-weight: 800;
    margin: 2px 0 12px 0;
}

.bar-track {
    height: 10px;
    background: var(--line);
    border-radius: 999px;
    overflow: hidden;
}

.bar-fill {
    height: 100%;
    border-radius: 999px;
}

.bar-fill.stay  { background: var(--safe); }
.bar-fill.churn { background: var(--risk); }

@media (max-width: 700px) {
    .hero { padding: 24px 22px; }
    .hero .main-title { font-size: 26px; }
    .result-box { flex-direction: column; text-align: center; }
}

@media (prefers-reduced-motion: reduce) {
    * { transition: none !important; }
}
</style>
""", unsafe_allow_html=True)


# ==========================================================
# TITLE
# ==========================================================

st.markdown(
    """
    <div class="hero">
        <div class="main-title">🏦 Bank Customer Churn Prediction</div>
        <p class="subtitle">Machine Learning Based Customer Churn Prediction System</p>
    </div>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# LOAD DATASET
# ==========================================================

@st.cache_data
def load_data():

    df = pd.read_csv("Bank_Churn_Classification_Dataset.csv")

    return df


df = load_data()


# ==========================================================
# TRAIN / LOAD MODEL
# ==========================================================

@st.cache_resource
def get_model(df):

    model_file = "bank_churn_model.joblib"

    # Load existing model
    if os.path.exists(model_file):

        saved_data = joblib.load(model_file)

        return (
            saved_data["model"],
            saved_data["encoders"],
            saved_data["feature_columns"]
        )

    # Copy dataset
    df1 = df.copy()

    # Separate features and target
    X = df1.drop("Churn", axis=1)

    y = df1["Churn"]

    # Encode categorical columns
    categorical_columns = X.select_dtypes(
        include="object"
    ).columns

    encoders = {}

    for col in categorical_columns:

        le = LabelEncoder()

        X[col] = le.fit_transform(X[col])

        encoders[col] = le

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Random Forest model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Save model
    saved_data = {
        "model": model,
        "encoders": encoders,
        "feature_columns": X.columns.tolist()
    }

    joblib.dump(
        saved_data,
        model_file
    )

    return (
        model,
        encoders,
        X.columns.tolist()
    )


model, encoders, feature_columns = get_model(df)


# ==========================================================
# CUSTOMER INFORMATION (centered form)
# ==========================================================

st.markdown(
    """
    <div class="section-title">👤 Customer Information</div>
    <div class="section-note">Fill in the details below, then run the prediction.</div>
    """,
    unsafe_allow_html=True
)

def card_header(icon, title, note, tone):
    st.markdown(
        f"""
        <div class="card-head {tone}">
            <div class="card-icon">{icon}</div>
            <div>
                <div class="card-title">{title}</div>
                <div class="card-note">{note}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with st.container():

    st.markdown('<div class="form-topbar"></div>', unsafe_allow_html=True)

    card1, card2, card3 = st.columns(3, gap="large")

    with card1:

        card_header("👤", "Customer Profile", "Who the customer is", "tone-teal")

        customer_id = st.number_input(
            "Customer ID",
            min_value=0,
            value=0,
            step=1
        )

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

    with card2:

        card_header("📈", "Account Activity", "How long and how much", "tone-indigo")

        tenure = st.number_input(
            "Tenure (months)",
            min_value=0,
            max_value=100,
            value=12,
            step=1
        )

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=70.0,
            step=0.01
        )

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=1000.0,
            step=0.01
        )

    with card3:

        card_header("📄", "Plan & Payment", "Contract and billing", "tone-amber")

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer",
                "Credit card"
            ]
        )


# ==========================================================
# PREDICTION
# ==========================================================

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

predict_button = st.button(
    "Predict Customer Churn",
    use_container_width=True
)


if predict_button:

    # Create input data
    input_data = pd.DataFrame({
        "Unnamed: 0": [customer_id],
        "CustomerID": [customer_id],
        "Gender": [gender],
        "SeniorCitizen": [senior_citizen],
        "Tenure": [tenure],
        "MonthlyCharges": [monthly_charges],
        "Contract": [contract],
        "PaymentMethod": [payment_method],
        "TotalCharges": [total_charges]
    })

    # Encode categorical values
    for col in encoders:

        input_data[col] = encoders[col].transform(
            input_data[col]
        )

    # Arrange columns exactly like training data
    input_data = input_data[feature_columns]

    # Make prediction
    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0]

    stay_probability = probability[0] * 100

    churn_probability = probability[1] * 100


    # ======================================================
    # RESULT
    # ======================================================

    circumference = 2 * 3.14159265 * 60

    if prediction == 1:

        css_class = "churn"
        shown_probability = churn_probability
        verdict = "⚠️ Customer is likely to CHURN"
        detail = "This customer may be at risk of leaving the bank."
        pill_text = "High attrition risk"

    else:

        css_class = "stay"
        shown_probability = stay_probability
        verdict = "✅ Customer is likely to STAY"
        detail = "This customer is predicted to continue with the bank."
        pill_text = "Likely to be retained"

    gauge_color = "#C23B2E" if prediction == 1 else "#1D7A55"
    dash_offset = circumference * (1 - shown_probability / 100)
    shown_label = "Churn Probability" if prediction == 1 else "Stay Probability"

    st.markdown(
        f"""
        <div class="result-box {css_class}">
            <div class="gauge">
                <svg width="150" height="150" viewBox="0 0 150 150">
                    <circle cx="75" cy="75" r="60" fill="none"
                            stroke="#DCE6E9" stroke-width="14"/>
                    <circle cx="75" cy="75" r="60" fill="none"
                            stroke="{gauge_color}" stroke-width="14"
                            stroke-linecap="round"
                            stroke-dasharray="{circumference:.2f}"
                            stroke-dashoffset="{dash_offset:.2f}"/>
                </svg>
                <div class="gauge-value">{shown_probability:.0f}%</div>
            </div>
            <div class="result-text">
                <div class="verdict">{verdict}</div>
                <div class="detail">{detail}</div>
                <span class="pill {css_class}">{shown_label}: {shown_probability:.2f}% &nbsp;·&nbsp; {pill_text}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


    # ======================================================
    # PROBABILITY
    # ======================================================

    st.markdown(
        '<div class="section-title">📊 Prediction Probability</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            f"""
            <div class="prob-card">
                <div class="prob-label">Stay Probability</div>
                <div class="prob-value" style="color: var(--safe);">{stay_probability:.2f}%</div>
                <div class="bar-track">
                    <div class="bar-fill stay" style="width: {int(stay_probability)}%;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            f"""
            <div class="prob-card">
                <div class="prob-label">Churn Probability</div>
                <div class="prob-value" style="color: var(--risk);">{churn_probability:.2f}%</div>
                <div class="bar-track">
                    <div class="bar-fill churn" style="width: {int(churn_probability)}%;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )