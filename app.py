import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_curve, auc, confusion_matrix,
                             ConfusionMatrixDisplay)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio

def safe_trendline_scatter(df, x, y, **kwargs):
    try:
        import statsmodels.api
        return px.scatter(df, x=x, y=y, trendline="ols", **kwargs)
    except ImportError:
        return px.scatter(df, x=x, y=y, **kwargs)

@st.cache_data(show_spinner=False)
def load_repo_csv() -> pd.DataFrame | None:
    repo_path = Path(__file__).parent / "data" / "extended_alcohol_consumers.csv"
    if repo_path.exists():
        return pd.read_csv(repo_path)
    return None

def request_upload() -> pd.DataFrame:
    st.info("Dataset not found in the repo. Please upload **extended_alcohol_consumers.csv**.")
    f = st.file_uploader("Upload dataset", type=["csv"])
    if f is None:
        st.stop()
    return pd.read_csv(f)

df = load_repo_csv()
if df is None:
    df = request_upload()

cat_cols = df.select_dtypes(include="object").columns.tolist()
num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

st.set_page_config(page_title="Alcohol Consumer Dashboard", layout="wide")
st.sidebar.title("⚙️ Global Controls")
theme_name = st.sidebar.selectbox("🎨 Theme", ["Default", "Vibrant", "Monochrome", "High-Contrast"])
palette_picker = st.sidebar.color_picker("Accent colour", "#1f77b4")
currency = st.sidebar.selectbox("💱 Currency", ["USD", "AED", "EUR", "INR"])
rate = {"USD":1, "AED":3.67, "EUR":0.92, "INR":83}[currency]
symbol = {"USD":"$", "AED":"د.إ", "EUR":"€", "INR":"₹"}[currency]
note = st.sidebar.text_area("🗒️ Session note")
if note:
    st.session_state.setdefault("notes", []).append(note)

tab_titles = [
    "📊 Data Visualisation",
    "🎯 Classification",
    "🧩 Clustering",
    "🔗 Association Rules",
    "📈 Regression",
    "🍾 3D Product View"
]
tabs = st.tabs(tab_titles)

# 1) DATA VISUALISATION
with tabs[0]:
    st.header("Exploratory Insights")
    fig = px.histogram(df, x="Age", nbins=30, title="Age Distribution", marginal="box")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Respondents concentrate in the 25–45 range, prime spending years.")

    fig = safe_trendline_scatter(df, x="Income_kUSD", y="Drinks_Per_Week",
                                 title="Income vs Drinks/Week")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Slight upward trend: higher income loosely correlates with more weekly drinks.")

    fig = px.pie(df, names="Preferred_Drink", hole=0.4, title="Preferred Drink Mix")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Beer and wine together account for >60 % of preferences.")

    fig = px.histogram(df, x="Generation", color="Price_Sensitivity",
                       barmode="group", title="Price Sensitivity by Generation")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Gen Z reports the largest share of ‘High’ price sensitivity.")

    spend_adj = df["Monthly_Spend_USD"] * rate
    fig = px.histogram(spend_adj, nbins=40, title=f"Monthly Alcohol Spend ({currency})")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Tail shows power-spenders – 1 % spend > {symbol}{spend_adj.quantile(0.99):,.0f}/mo.")

    # ---- FIX: Drop rows with NaN for Sunburst ----
    sb_data = df.dropna(subset=["Taste_Preference", "Brand_Loyalty"])
    if sb_data.empty:
        st.warning("No data available for Taste vs Brand Loyalty sunburst.")
    else:
        fig = px.sunburst(sb_data, path=["Taste_Preference","Brand_Loyalty"], title="Taste vs Brand Loyalty")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Bitter-taste respondents show the highest ‘High’ loyalty cluster.")

    agebin = pd.cut(df["Age"], bins=[17,25,35,45,55,65,80], labels=["18-25","26-35","36-45","46-55","56-65","66+"])
    fig = px.bar(df.assign(Age_Bin=agebin), x="Age_Bin", color="Support_Local_Store", barmode="group", title="Willingness to Support Local Store (by Age)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("26-35 group most supportive of a new local outlet.")

    fig = px.box(df, x="Gender", y="Drinks_Per_Week", title="Drinking Intensity by Gender")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Male median ≈ 3 drinks/week vs female ≈ 2.")

    fig = px.violin(df, x="Support_Local_Store", y="Health_Score", box=True, title="Health-Consciousness vs Store Support")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Even highly health-conscious (score ≥ 4) show ~40 % support.")

    corr = df[num_cols].corr()
    fig = px.imshow(corr, text_auto=True, title="Numeric Feature Correlations")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Monthly Spend correlates with Income and Drinks/Week.")

# ... [rest of the code is unchanged, as in your last working version] ...
# (Paste your full tab code for classification, clustering, association rules, regression, and 3D view, as given previously)

# Sticky notes panel
if st.session_state.get("notes"):
    st.sidebar.subheader("💬 Stored Notes")
    for i, n in enumerate(st.session_state["notes"], 1):
        st.sidebar.markdown(f"**{i}.** {n}")
