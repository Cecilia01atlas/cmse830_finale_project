# ======================================================
#   FINAL PROJECT — STREAMLIT APP (uses daily_mean.csv)
# ======================================================

import streamlit as st
import pandas as pd

import numpy as np
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

with st.sidebar:
    st.title("🌐 ENSO Explorer")
    st.markdown(
        """
Explore the **impact of El Niño & La Niña** on ocean–atmosphere variables  
through interactive visualizations and imputation tools.
"""
    )
    st.markdown("---")
    choice = st.radio("Navigate to:", menu)


# =====================================================
# Overview
# =====================================================

# =========================
# Tab 1: Overview
# =========================
if choice == "Overview":
    st.title("🌊 Dataset Overview")

    st.markdown("""
    Welcome to the **ENSO Explorer App**, an interactive platform that visualizes
    ocean–atmosphere interactions using long-term TAO mooring data.
    
    This dataset includes measurements of:
    - 🌡️ Sea Surface Temperature (SST)
    - 🌡️ Air Temperature
    - 💨 Zonal & Meridional Winds
    - 💧 Relative Humidity
    - 📉 Subsurface Temperatures at multiple depths
    - 🌍 ENSO Niño 3.4 Index (ANOM)
    
    The goal of this app is to help investigate ocean–atmosphere coupling
    and ENSO dynamics over several decades.
    """)

    # -------------------------
    # Key metrics
    # -------------------------
    total_rows = len(df)
    total_cols = len(df.columns)
    total_missing = df.isna().sum().sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", total_rows)
    col2.metric("Total Features", total_cols)
    col3.metric("Missing Values", total_missing)

    # -------------------------
    # Column information
    # --------------------------

    with st.expander("📋 Column Information"):
        descriptions = {
            "date": "Date of observation",
            "year": "Year extracted from date",
            "month": "Month extracted from date",
            "T_25": "Sea Surface Temperature (°C)",
            "AT_21": "Air Temperature (°C)",
            "RH_910": "Relative Humidity (%)",
            "WU_422": "Zonal Wind (m/s, west-east)",
            "WV_423": "Meridional Wind (m/s, south-north)",
            "ANOM": "ENSO Niño 3.4 Index",
            # Depth temperatures (automatically handled below)
        }

        # Automatically describe depth variables
        for col in df.columns:
            if col.startswith("temp_"):
                depth = col.replace("temp_", "").replace("m", "")
                descriptions[col] = f"Temperature at {depth} m depth (°C)"

        col_info = pd.DataFrame(
            {
                "Column": df.columns,
                "Type": df.dtypes.values,
                "Missing Values": df.isna().sum().values,
                "Description": [descriptions.get(col, "") for col in df.columns],
            }
        )

        st.dataframe(col_info.astype(str))

    # -------------------------
    # Summary statistics
    # -------------------------
    with st.expander("📈 Summary Statistics"):
        numeric_df = df.select_dtypes(include=["float64", "int64"])
        st.write(numeric_df.describe())

    # -------------------------
    # Temporal coverage plot
    # -------------------------
    with st.expander("🕒 Temporal Coverage"):
        st.markdown("""
        This graph shows the **number of recorded measurements per month**.
        It helps identify gaps in observation coverage and denser periods.
        """)

        df["year_month"] = (
            df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2)
        )
        ym_counts = df["year_month"].value_counts().sort_index()

        fig, ax = plt.subplots(figsize=(14, 4))
        ym_counts.plot(ax=ax)

        ax.set_title("Temporal Coverage of Observations")
        ax.set_xlabel("Year-Month")
        ax.set_ylabel("Number of Measurements")
        ax.tick_params(axis="x", rotation=90)

        st.pyplot(fig)

    # -------------------------
    # Duplicate detection
    # -------------------------
    with st.expander("🔁 Duplicate Records"):
        duplicate_count = df.duplicated().sum()

        if duplicate_count > 0:
            st.warning(f"⚠️ {duplicate_count} duplicate rows found.")
            st.dataframe(df[df.duplicated()].head())
        else:
            st.success("✅ No duplicate rows found.")

    # -------------------------
    # Outlier detection
    # -------------------------
    with st.expander("🚨 Outlier Detection (|Z-score| > 3)"):
        st.markdown("""
        Outliers may represent measurement anomalies or extreme
        environmental events (especially during ENSO conditions).
        """)

        # Columns we want to check
        numeric_cols = ["T_25", "AT_21", "RH_910", "WU_422", "WV_423", "ANOM"] + [
            col for col in df.columns if col.startswith("temp_")
        ]

        outlier_results = {}

        for col in numeric_cols:
            if col in df.columns:
                z = (df[col] - df[col].mean()) / df[col].std()
                outlier_results[col] = (z.abs() > 3).sum()

        st.write(
            pd.DataFrame.from_dict(
                outlier_results, orient="index", columns=["Outliers"]
            )
        )


# ================================================
# Tab 2: Missingness
# ================================================
elif choice == "Missingness":
    st.title("🚧 Missingness Analysis")
    st.markdown("""
    Long-term environmental datasets often contain **missing values** due to instrument failure,
    weather disruptions, or transmission gaps.  
    Before performing modeling or anomaly detection, it's crucial to explore what is missing
    and apply a robust imputation method.
    """)

    df = st.session_state.get("df_original", df)  # keep original stored once

    # -----------------------------
    # Missingness Table
    # -----------------------------
    with st.expander("📋 Missingness Summary Table"):
        summary_table = pd.DataFrame(
            {
                "Missing Values": df.isna().sum(),
                "Missing %": (df.isna().mean() * 100).round(2),
            }
        ).sort_values("Missing Values", ascending=False)

        st.dataframe(summary_table)

    # -----------------------------
    # Missingness Heatmap
    # -----------------------------
    with st.expander("🌡 Missingness Heatmap"):

    st.markdown("""
    The heatmap below shows missing values (yellow = missing),
    aligned over time. Each row corresponds to a feature.
    """)

    cols_for_heatmap = [
        col for col in df.columns 
        if col not in ["year", "month"]
    ]

    nan_array = df[cols_for_heatmap].isna().astype(int).to_numpy().T  # transpose here

    # HIGH DPI + SHARP EDGES
    fig, ax = plt.subplots(figsize=(30, 15), dpi=150)

    # pcolormesh gives cleaner boundaries for big grids
    mesh = ax.pcolormesh(
        nan_array,
        cmap="cividis",
        shading="nearest"  # <-- SHARP
    )

    ax.set_title("Missing Values Heatmap (1 = Missing)", fontsize=24, pad=20)
    ax.set_xlabel("Time Index", fontsize=20)
    ax.set_ylabel("Features", fontsize=20)

    # Y-axis labels (features)
    ax.set_yticks(np.arange(len(cols_for_heatmap)) + 0.5)
    ax.set_yticklabels(cols_for_heatmap, fontsize=12)

    # X-axis ticks (time)
    n_rows = nan_array.shape[1]
    n_ticks = 10  # fewer, clearer ticks
    tick_positions = np.linspace(0, n_rows-1, n_ticks).astype(int)
    tick_labels = df.loc[tick_positions, "date"].dt.strftime("%Y-%m-%d")

    ax.set_xticks(tick_positions + 0.5)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=12)

    # Add colorbar
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Missingness", fontsize=18)
    cbar.ax.tick_params(labelsize=12)

    st.pyplot(fig)


    # -----------------------------
    # RF-MICE IMPUTATION
    # -----------------------------
    st.subheader("🤖 Random Forest MICE Imputation")

    st.markdown("""
    This imputation method uses **Iterative Imputation** with **Random Forest regressors**,  
    allowing each variable to be predicted from all others.  
    This method handles nonlinear relationships and works well for environmental datasets.
    """)

    if st.button("Run Imputation"):
        # from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        from sklearn.ensemble import RandomForestRegressor

        # Variables to impute
        columns_to_impute = [
            "WU_422",
            "WV_423",
            "RH_910",
            "AT_21",
            "temp_1m",
            "temp_10m",
            "temp_20m",
            "temp_50m",
            "temp_75m",
            "temp_100m",
            "temp_150m",
            "temp_175m",
            "temp_200m",
            "temp_250m",
            "T_25",
        ]

        missing_mask = df[columns_to_impute].isna()

        # Remove non-numeric columns
        non_numeric_cols = df.select_dtypes(exclude=["float", "int"]).columns
        numeric_df = df.drop(columns=non_numeric_cols)

        rf = RandomForestRegressor(
            n_estimators=50, max_depth=10, random_state=42, n_jobs=-1
        )

        imputer = IterativeImputer(estimator=rf, max_iter=5, random_state=42)

        df_imputed = pd.DataFrame(
            imputer.fit_transform(numeric_df), columns=numeric_df.columns
        )

        # Re-add date/year/month columns
        for col in non_numeric_cols:
            df_imputed[col] = df[col]

        # Store imputed version
        st.session_state["df"] = df_imputed.copy()

        st.success("Imputation completed successfully!")

        # -----------------------------
        # Plot Original vs Imputed
        # -----------------------------
        st.subheader("📊 Original vs Imputed Values")

        pretty_names = {
            "WU_422": "Zonal Wind (m/s)",
            "WV_423": "Meridional Wind (m/s)",
            "RH_910": "Relative Humidity (%)",
            "AT_21": "Air Temperature (°C)",
            "T_25": "Sea Surface Temperature (°C)",
            "temp_1m": "Temperature at 1m (°C)",
            "temp_10m": "Temperature at 10m (°C)",
            "temp_20m": "Temperature at 20m (°C)",
            "temp_50m": "Temperature at 50m (°C)",
            "temp_75m": "Temperature at 75m (°C)",
            "temp_100m": "Temperature at 100m (°C)",
            "temp_150m": "Temperature at 150m (°C)",
            "temp_175m": "Temperature at 175m (°C)",
            "temp_200m": "Temperature at 200m (°C)",
            "temp_250m": "Temperature at 250m (°C)",
        }

        for col in columns_to_impute:
            fig, ax = plt.subplots(figsize=(14, 4))

            ax.plot(df["date"], df[col], alpha=0.4, label="Original")
            ax.scatter(
                df.loc[missing_mask[col], "date"],
                df_imputed.loc[missing_mask[col], col],
                s=10,
                color="orange",
                label="Imputed",
            )

            ax.set_title(f"{pretty_names.get(col, col)} — Original vs Imputed")
            ax.set_xlabel("Date")
            ax.set_ylabel(pretty_names.get(col, col))
            ax.legend()
            ax.grid(alpha=0.3)

            st.pyplot(fig)

        st.write("Missing AFTER imputation:")
        st.write(df_imputed[columns_to_impute].isna().sum())

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
