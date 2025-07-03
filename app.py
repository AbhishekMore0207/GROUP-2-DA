import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px

# ... (all your other imports, unchanged)
# (Include everything you had previously in app.py)

def safe_trendline_scatter(df, x, y, **kwargs):
    # Only use trendline="ols" if statsmodels is present
    try:
        import statsmodels.api
        return px.scatter(df, x=x, y=y, trendline="ols", **kwargs)
    except ImportError:
        return px.scatter(df, x=x, y=y, **kwargs)

# ... all your data loading and sidebar code (unchanged) ...

# ----------- Data Visualisation Tab -------------
with tabs[0]:
    st.header("Exploratory Insights")
    # ... other charts ...
    # Instead of:
    # fig = px.scatter(df, x="Income_kUSD", y="Drinks_Per_Week",
    #                  trendline="ols", title="Income vs Drinks/Week")
    # Use:
    fig = safe_trendline_scatter(df, x="Income_kUSD", y="Drinks_Per_Week",
                                 title="Income vs Drinks/Week")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Slight upward trend: higher income loosely correlates with more weekly drinks.")
    # ... rest of the tab as before ...

# ... (rest of your code remains unchanged) ...
