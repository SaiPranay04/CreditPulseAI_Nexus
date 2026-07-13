import streamlit as st
import pandas as pd
import json
import sys
from pathlib import Path

# Add project root to path for imports
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import config

# Set page layout and title
st.set_page_config(
    page_title="CreditPulse AI — Credit Risk Decisions",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply global dark mode premium styling
st.markdown("""
    <style>
        /* Import Outfit font from Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
        
        /* Main background and fonts */
        html, body, [data-testid="stAppViewContainer"] {
            background-color: #0e1117;
            font-family: 'Outfit', sans-serif;
            color: #e6e9ef;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #0b0d13;
            border-right: 1px solid #1a1d29;
        }
        
        /* Card-like divs */
        div.stMetric {
            background-color: #1a1d29;
            padding: 1.2rem;
            border-radius: 12px;
            border: 1px solid #282a36;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            transition: transform 0.2s, border-color 0.2s;
        }
        div.stMetric:hover {
            transform: translateY(-2px);
            border-color: #00c2ff;
        }
        
        /* Customize tabs */
        button[data-baseweb="tab"] {
            font-size: 1.1rem;
            font-weight: 500;
            color: #8a8d9a;
            background-color: transparent;
            padding: 10px 20px;
            transition: color 0.3s, border-color 0.3s;
        }
        button[data-baseweb="tab"]:hover {
            color: #00c2ff;
        }
        button[aria-selected="true"] {
            color: #00c2ff !important;
            border-bottom-color: #00c2ff !important;
        }
        
        /* Header titles */
        h1, h2, h3 {
            font-weight: 700 !important;
            color: #ffffff;
            letter-spacing: -0.02em;
        }
        
        /* Info/warning banners */
        div.stAlert {
            border-radius: 8px;
            background-color: #1a1d29;
            border: 1px solid #282a36;
        }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# Cache Data Loading (Fast O(1) reads)
# ----------------------------------------------------
@st.cache_data(show_spinner="Loading portfolio intelligence...")
def load_scored_data():
    df = pd.read_parquet(config.SCORED)
    # Set SK_ID_CURR as index for instant O(1) lookups
    return df.set_index("SK_ID_CURR", drop=False)

@st.cache_data(show_spinner="Loading local explanation matrix...")
def load_shap_top10():
    return pd.read_parquet(config.SHAP_TOP10)

@st.cache_data(show_spinner="Loading global feature drivers...")
def load_shap_global():
    return pd.read_parquet(config.SHAP_GLOBAL)

@st.cache_data(show_spinner="Loading model evaluation metrics...")
def load_metrics():
    with open(config.METRICS, 'r') as f:
        return json.load(f)

# Load resources
try:
    scored_df = load_scored_data()
    shap_top10_df = load_shap_top10()
    shap_global_df = load_shap_global()
    metrics = load_metrics()
except Exception as e:
    st.error(f"Error loading scored assets: {e}")
    st.info("Ensure you have run the scoring pipeline: `python -m pipeline.04_score`")
    st.stop()

# Import component render functions
from app.components import portfolio, lookup, ews, performance

# ----------------------------------------------------
# Sidebar & Navigation
# ----------------------------------------------------
st.sidebar.markdown("""
    <div style='text-align: center; padding: 1.5rem 0;'>
        <h1 style='font-size: 2.2rem; margin-bottom: 0; background: linear-gradient(135deg, #00c2ff, #2ecc71); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>CreditPulse AI</h1>
        <p style='color: #8a8d9a; font-size: 0.9rem; margin-top: 0.2rem;'>Decision Support Dashboard</p>
    </div>
    <hr style='border: 0; border-top: 1px solid #1a1d29; margin-bottom: 1.5rem;' />
""", unsafe_allow_html=True)

# Sidebar metadata / indicators
st.sidebar.subheader("System Metadata")
st.sidebar.markdown(f"""
*   **Active Portfolio Size**: `{len(scored_df):,}` customers
*   **Target Imbalance (11.4:1)**: Calibrated
*   **Stack Models**: `LGBM + XGBoost + CatBoost`
*   **Meta-Learner**: `Logistic Regression`
*   **SHAP Status**: `Precomputed (O(1) latency)`
""")

st.sidebar.markdown("""
    <hr style='border: 0; border-top: 1px solid #1a1d29; margin-top: 2rem;' />
    <div style='text-align: center; color: #8a8d9a; font-size: 0.8rem;'>
        IDBI Innovate Hackathon 2026<br/>
        <b>Engineered by Antigravity</b>
    </div>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# Main Layout Tabs
# ----------------------------------------------------
tabs = st.tabs([
    "📊 Portfolio Overview",
    "🔎 Customer Lookup",
    "⚠️ Early Warning System",
    "⚙️ Model Performance"
])

with tabs[0]:
    portfolio.render(scored_df)

with tabs[1]:
    lookup.render(scored_df, shap_top10_df)

with tabs[2]:
    ews.render(scored_df)

with tabs[3]:
    performance.render(metrics, shap_global_df)
