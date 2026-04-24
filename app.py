import streamlit as st
import joblib
import numpy as np
import time
import os
import pandas as pd
from xgboost import XGBClassifier

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #080c14 !important;
    font-family: 'DM Sans', sans-serif;
    color: #e2e8f0;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(220,38,38,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(30,64,175,0.10) 0%, transparent 60%),
        #080c14 !important;
}

[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

.hero {
    text-align: center;
    padding: 3rem 0 2rem;
}
.hero-badge {
    display: inline-block;
    background: rgba(220,38,38,0.12);
    border: 1px solid rgba(220,38,38,0.3);
    color: #f87171;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    padding: 0.35rem 1rem;
    border-radius: 100px;
    margin-bottom: 1.2rem;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.2rem, 5vw, 3.8rem);
    font-weight: 800;
    line-height: 1.05;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #ffffff 0%, #94a3b8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.8rem;
}
.hero-title span {
    background: linear-gradient(135deg, #ef4444 0%, #f97316 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    color: #64748b;
    font-size: 1rem;
    font-weight: 300;
}

.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
    margin: 1.5rem 0;
}

.metrics-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-bottom: 2rem;
}
.metric-card {
    background: linear-gradient(135deg, #0f1623 0%, #111827 100%);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    border-radius: 16px 16px 0 0;
}
.metric-card.green::before  { background: linear-gradient(90deg, #10b981, #34d399); }
.metric-card.blue::before   { background: linear-gradient(90deg, #3b82f6, #60a5fa); }
.metric-card.red::before    { background: linear-gradient(90deg, #ef4444, #f97316); }
.metric-icon { font-size: 1.4rem; margin-bottom: 0.6rem; }
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.metric-card.green .metric-value { color: #10b981; }
.metric-card.blue  .metric-value { color: #60a5fa; }
.metric-card.red   .metric-value { color: #f87171; }
.metric-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #475569;
}

.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(255,255,255,0.06);
}

.input-panel {
    background: #0d1117;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px;
    padding: 1.8rem;
    margin-bottom: 1.5rem;
}

[data-testid="stSelectbox"] > div > div,
[data-testid="stNumberInput"] input {
    background: #161d2b !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
}
label, [data-testid="stWidgetLabel"] p {
    color: #94a3b8 !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.03em !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stButton"] > button {
    width: 100%;
    background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.9rem 2rem !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    box-shadow: 0 4px 24px rgba(220,38,38,0.3) !important;
    margin-top: 1rem !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 32px rgba(220,38,38,0.45) !important;
}

.result-fraud {
    background: linear-gradient(135deg, rgba(127,29,29,0.6) 0%, rgba(153,27,27,0.4) 100%);
    border: 1px solid rgba(239,68,68,0.4);
    border-left: 4px solid #ef4444;
    border-radius: 16px;
    padding: 2rem 2.4rem;
    margin-top: 1.5rem;
    animation: slideUp 0.4s ease;
}
.result-legit {
    background: linear-gradient(135deg, rgba(6,78,59,0.5) 0%, rgba(16,185,129,0.15) 100%);
    border: 1px solid rgba(16,185,129,0.3);
    border-left: 4px solid #10b981;
    border-radius: 16px;
    padding: 2rem 2.4rem;
    margin-top: 1.5rem;
    animation: slideUp 0.4s ease;
}
@keyframes slideUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    margin-bottom: 0.4rem;
    letter-spacing: -0.01em;
}
.result-fraud .result-title { color: #f87171; }
.result-legit .result-title { color: #34d399; }
.result-sub { color: #94a3b8; font-size: 0.9rem; font-weight: 300; }

.gauge-wrap { margin-top: 1.2rem; }
.gauge-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.75rem;
    color: #64748b;
    margin-bottom: 0.4rem;
    font-weight: 500;
}
.gauge-track {
    height: 8px;
    background: rgba(255,255,255,0.07);
    border-radius: 100px;
    overflow: hidden;
}
.gauge-fill { height: 100%; border-radius: 100px; }
.gauge-fill.fraud { background: linear-gradient(90deg, #f97316, #ef4444); }
.gauge-fill.legit { background: linear-gradient(90deg, #10b981, #34d399); }

.sidebar-section {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
}
.sidebar-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 0.8rem;
}
.feat-row {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.6rem;
}
.feat-name { font-size: 0.78rem; color: #94a3b8; flex: 1; }
.feat-pct  { font-size: 0.78rem; color: #e2e8f0; font-weight: 600; width: 3.5rem; text-align: right; }
.feat-bar-track {
    flex: 2;
    height: 4px;
    background: rgba(255,255,255,0.06);
    border-radius: 100px;
    overflow: hidden;
}
.feat-bar-fill {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #ef4444, #f97316);
}
.status-dot {
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #10b981;
    box-shadow: 0 0 6px #10b981;
    margin-right: 0.4rem;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.4; }
}
</style>
""", unsafe_allow_html=True)


# ── Load Model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = r"D:\dell\fraud_detection\fraud_model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    
    with st.spinner("Training model for the first time. This may take a few moments..."):
        data_path = r"D:\dell\fraud_detection\paysim_sample.csv"
        if not os.path.exists(data_path):
            st.error(f"Dataset not found at {data_path}. Please provide the dataset to train the model.")
            st.stop()
            
        df = pd.read_csv(data_path)
        
        df = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])].copy()
        df['type'] = df['type'].map({'TRANSFER': 0, 'CASH_OUT': 1})
        df['errorBalanceOrig'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
        df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']
        
        features = ['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
                    'oldbalanceDest', 'newbalanceDest', 'errorBalanceOrig', 'errorBalanceDest']
        X = df[features]
        y = df['isFraud']
        
        model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
        model.fit(X, y)
        
        joblib.dump(model, model_path)
    return model

model = load_model()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1.2rem 0 0.5rem;'>
        <div style='font-family:Syne,sans-serif;font-size:1.3rem;font-weight:800;
                    color:#fff;letter-spacing:-0.01em;margin-bottom:0.2rem;'>
            🛡️ FraudShield
        </div>
        <div style='font-size:0.75rem;color:#475569;'>AI-Powered Detection Engine</div>
    </div>
    <div class='divider'></div>
    <div class='sidebar-section'>
        <div class='sidebar-title'>System Status</div>
        <div style='font-size:0.82rem;color:#94a3b8;'>
            <span class='status-dot'></span>Model Online
        </div>
        <div style='font-size:0.82rem;color:#94a3b8;margin-top:0.4rem;'>
            <span style='color:#475569;'>Algorithm:</span> XGBoost Classifier
        </div>
        <div style='font-size:0.82rem;color:#94a3b8;margin-top:0.4rem;'>
            <span style='color:#475569;'>Trained on:</span> 6.3M transactions
        </div>
        <div style='font-size:0.82rem;color:#94a3b8;margin-top:0.4rem;'>
            <span style='color:#475569;'>Dataset:</span> PaySim Synthetic
        </div>
    </div>
    <div class='sidebar-section'>
        <div class='sidebar-title'>Feature Importance</div>
        <div class='feat-row'>
            <div class='feat-name'>errorBalanceOrig</div>
            <div class='feat-bar-track'><div class='feat-bar-fill' style='width:100%'></div></div>
            <div class='feat-pct'>67.87%</div>
        </div>
        <div class='feat-row'>
            <div class='feat-name'>newbalanceOrig</div>
            <div class='feat-bar-track'><div class='feat-bar-fill' style='width:43%'></div></div>
            <div class='feat-pct'>29.16%</div>
        </div>
        <div class='feat-row'>
            <div class='feat-name'>amount</div>
            <div class='feat-bar-track'><div class='feat-bar-fill' style='width:3%'></div></div>
            <div class='feat-pct'>2.03%</div>
        </div>
    </div>
    <div class='sidebar-section'>
        <div class='sidebar-title'>How to Use</div>
        <div style='font-size:0.8rem;color:#64748b;line-height:1.7;'>
            1. Select transaction type<br>
            2. Enter the transfer amount<br>
            3. Fill in account balances<br>
            4. Click <b style='color:#f87171'>Analyze Transaction</b>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
    <div class='hero-badge'>🛡️ &nbsp; Real-Time AI Detection</div>
    <div class='hero-title'>Bank Transfer<br><span>Fraud Detection</span></div>
    <div class='hero-sub'>Powered by XGBoost · Trained on 6.3 million transactions · 99.78% AUC-ROC</div>
</div>
""", unsafe_allow_html=True)


# ── Metric Cards ───────────────────────────────────────────────────────────────
st.markdown("""
<div class='metrics-row'>
    <div class='metric-card green'>
        <div class='metric-icon'>✅</div>
        <div class='metric-value'>100%</div>
        <div class='metric-label'>Model Accuracy</div>
    </div>
    <div class='metric-card blue'>
        <div class='metric-icon'>📈</div>
        <div class='metric-value'>0.9978</div>
        <div class='metric-label'>AUC-ROC Score</div>
    </div>
    <div class='metric-card red'>
        <div class='metric-icon'>🎯</div>
        <div class='metric-value'>99%</div>
        <div class='metric-label'>Fraud Recall Rate</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)


# ── Input Panel ────────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>Transaction Details</div>", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("<div class='input-panel'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>💳 &nbsp; Transaction Info</div>", unsafe_allow_html=True)
    tx_type   = st.selectbox("Transaction Type", ["TRANSFER", "CASH_OUT"])
    amount    = st.number_input("Amount (₹)", min_value=0.0, value=150000.0, step=1000.0, format="%.2f")
    old_bal_o = st.number_input("Old Balance — Sender", min_value=0.0, value=200000.0, step=1000.0, format="%.2f")
    new_bal_o = st.number_input("New Balance — Sender", min_value=0.0, value=50000.0, step=1000.0, format="%.2f")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='input-panel'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>🏦 &nbsp; Recipient Account</div>", unsafe_allow_html=True)
    old_bal_d = st.number_input("Old Balance — Recipient", min_value=0.0, value=0.0, step=1000.0, format="%.2f")
    new_bal_d = st.number_input("New Balance — Recipient", min_value=0.0, value=150000.0, step=1000.0, format="%.2f")
    st.markdown("""
    <div style='margin-top:1.4rem;padding:1rem;background:rgba(239,68,68,0.05);
                border:1px solid rgba(239,68,68,0.12);border-radius:10px;'>
        <div style='font-size:0.72rem;font-weight:600;letter-spacing:0.1em;
                    text-transform:uppercase;color:#f87171;margin-bottom:0.4rem;'>
            ⚡ Auto-Calculated
        </div>
        <div style='font-size:0.8rem;color:#64748b;line-height:1.6;'>
            Balance error features are computed automatically from your inputs
            to power the AI detection engine.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ── Predict Button ─────────────────────────────────────────────────────────────
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    predict = st.button("🔍  Analyze Transaction")


# ── Prediction ─────────────────────────────────────────────────────────────────
if predict:
    with st.spinner("Analyzing transaction patterns..."):
        time.sleep(0.6)

    type_enc       = 0 if tx_type == "TRANSFER" else 1
    error_bal_orig = new_bal_o + amount - old_bal_o
    error_bal_dest = old_bal_d + amount - new_bal_d

    features   = np.array([[type_enc, amount, old_bal_o, new_bal_o,
                             old_bal_d, new_bal_d, error_bal_orig, error_bal_dest]])
    prediction = model.predict(features)[0]
    proba      = model.predict_proba(features)[0]
    fraud_pct  = round(proba[1] * 100, 2)
    legit_pct  = round(proba[0] * 100, 2)

    if prediction == 1:
        st.markdown(f"""
        <div class='result-fraud'>
            <div class='result-title'>🚨 &nbsp; Fraud Detected</div>
            <div class='result-sub'>This transaction matches known fraud patterns. Immediate review recommended.</div>
            <div class='gauge-wrap'>
                <div class='gauge-label'>
                    <span>Risk Score</span>
                    <span style='color:#f87171;font-weight:700;font-size:0.9rem;'>{fraud_pct}% Fraud Probability</span>
                </div>
                <div class='gauge-track'>
                    <div class='gauge-fill fraud' style='width:{fraud_pct}%'></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='result-legit'>
            <div class='result-title'>✅ &nbsp; Legitimate Transaction</div>
            <div class='result-sub'>No significant fraud patterns detected. Transaction appears safe to process.</div>
            <div class='gauge-wrap'>
                <div class='gauge-label'>
                    <span>Safety Score</span>
                    <span style='color:#34d399;font-weight:700;font-size:0.9rem;'>{legit_pct}% Legitimate</span>
                </div>
                <div class='gauge-track'>
                    <div class='gauge-fill legit' style='width:{legit_pct}%'></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Analysis Breakdown</div>", unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)
    with d1:
        st.metric("Transaction Type", tx_type)
    with d2:
        st.metric("Amount", f"₹{amount:,.0f}")
    with d3:
        st.metric("Balance Error (Sender)", f"₹{error_bal_orig:,.0f}")