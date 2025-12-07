# ======================================================
#   FINAL PROJECT — STREAMLIT APP (uses daily_mean.csv)
# ======================================================

import streamlit as st
import pandas as pd

# import numpy as np
import matplotlib.pyplot as plt

# import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
# import matplotlib.dates as mdates
# from sklearn.linear_model import LinearRegression

# -------- Streamlit page config ----------
st.set_page_config(
    page_title="Ocean–Atmosphere Explorer 🌊",
    layout="wide",
)


# -------- Cached Data Loader -------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/daily_mean.csv")

    # date processing
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    return df


df = load_data()

# =====================================================
# Sidebar Navigation
# =====================================================
menu = [
    "Overview",
    "Missingness",
    "Temporal Coverage",
    "Correlation Study",
    "Temperature Profiles",
    "ENSO Anomalies",
    "Conclusion",
]

choice = st.sidebar.radio("Navigate:", menu)

# =====================================================
# Overview
# =====================================================

if choice == "Overview":
    st.title("🌊 Final Project: Ocean–Atmosphere Dynamics Explorer")

    st.markdown("""
    This dashboard summarizes the relationships between:
    - **Sea Surface Temperature (SST)**  
    - **Subsurface temperatures at multiple depths**  
    - **Winds, humidity, and air temperature**  
    - **ENSO (El Niño / La Niña)**  

    Your dataset is already fully processed into daily averages from all buoys.
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(df))
    col2.metric("Total Features", df.shape[1])
    col3.metric("Missing Cells", df.isna().sum().sum())

    st.subheader("📋 Column Summary")
    st.dataframe(
        pd.DataFrame(
            {
                "Column": df.columns,
                "Type": df.dtypes.astype(str),
                "Missing": df.isna().sum(),
            }
        )
    )

    st.subheader("📈 Summary Statistics")
    st.write(df.describe())

# =====================================================
# Missingness
# =====================================================

elif choice == "Missingness":
    st.title("🚧 Missingness Analysis")

    missing_table = pd.DataFrame(
        {
            "Missing Values": df.isna().sum(),
            "Missing %": (df.isna().mean() * 100).round(2),
        }
    ).sort_values("Missing Values", ascending=False)

    st.subheader("📌 Missingness Summary")
    st.dataframe(missing_table)

    st.subheader("🔍 Missingness Heatmap")
    nan_array = df.isna().astype(int).to_numpy()

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(nan_array.T, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(df.columns)))
    ax.set_yticklabels(df.columns)
    ax.set_title("Missing Values Heatmap")
    plt.colorbar(im)
    st.pyplot(fig)

# =====================================================
# Temporal Coverage
# =====================================================

elif choice == "Temporal Coverage":
    st.title("📆 Temporal Coverage of Observations")

    df["year_month"] = (
        df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2)
    )
    ym_counts = df["year_month"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(15, 5))
    ym_counts.plot(ax=ax)
    plt.xticks(rotation=90)
    ax.set_title("Number of Observations per Year-Month")
    st.pyplot(fig)

    st.subheader("📅 SST Over Time (Colored by ENSO Index)")
    if "ANOM" in df.columns:
        anom_abs = max(abs(df["ANOM"].min()), abs(df["ANOM"].max()))
        fig_scatter = px.scatter(
            df,
            x="date",
            y="T_25",
            color="ANOM",
            color_continuous_scale="RdBu_r",
            opacity=0.5,
            title="Sea Surface Temperature Over Time",
        )
        fig_scatter.update_layout(coloraxis=dict(cmin=-anom_abs, cmax=anom_abs, cmid=0))
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.warning("ENSO index not found in dataset.")

# =====================================================
# Correlation Study
# =====================================================

elif choice == "Correlation Study":
    st.header("📊 Correlation Between Variables")

    selected = [
        "T_25",
        "AT_21",
        "RH_910",
        "WU_422",
        "WV_423",
        "temp_10m",
        "temp_50m",
        "temp_100m",
        "temp_150m",
    ]
    subset = df[selected].dropna()
    corr = subset.corr()

    st.subheader("🔸 Correlation Heatmap")
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔸 Scatter Matrix")
    scatter_df = subset.sample(min(len(subset), 1500))
    fig2 = px.scatter_matrix(scatter_df, dimensions=scatter_df.columns)
    fig2.update_traces(diagonal_visible=False)
    st.plotly_chart(fig2, use_container_width=True)

# =====================================================
# Temperature Profiles
# =====================================================

elif choice == "Temperature Profiles":
    st.header("🌡 Temperature at Multiple Depths")

    depth_cols = [c for c in df.columns if "temp_" in c]

    st.subheader("📈 Temperature Over Time by Depth")
    fig = go.Figure()
    for col in depth_cols:
        fig.add_trace(go.Scatter(x=df["date"], y=df[col], mode="lines", name=col))
    fig.update_layout(template="plotly_white", height=500)
    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# ENSO Anomalies
# =====================================================

elif choice == "ENSO Anomalies":
    st.header("🌡 ENSO & Temperature Anomalies")

    if "ANOM" not in df.columns:
        st.error("ENSO index (ANOM) missing from dataset!")
    else:
        df["sst_monthly_clim"] = df.groupby("month")["T_25"].transform("mean")
        df["sst_anomaly"] = df["T_25"] - df["sst_monthly_clim"]

        fig = px.line(df, x="date", y="sst_anomaly", title="SST Anomaly Over Time")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Scatter of SST Anomalies vs Air Temperature")
        fig2 = px.scatter(df, x="sst_anomaly", y="AT_21", opacity=0.5, trendline="ols")
        st.plotly_chart(fig2, use_container_width=True)

# =====================================================
# Conclusion
# =====================================================

elif choice == "Conclusion":
    st.title("📖 Conclusion")

    st.markdown("""
    ### Key Takeaways

    - Ocean and atmosphere variables show strong coupling.  
    - Air temperature and SST are highly correlated.  
    - ENSO (El Niño/La Niña) patterns clearly influence SST anomalies.  
    - Subsurface temperature profiles reveal long-term shifts and stratification.  

    This dashboard allows interactive exploration of these relationships.
    """)
