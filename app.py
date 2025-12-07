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
    Long climate records often contain **missing measurements**, especially at certain depths or during sensor outages.
    Understanding these gaps is essential before doing imputation or trend analysis.
    """)

    # =====================================================
    # 1) Missingness Summary Table
    # =====================================================
    with st.expander("📋 Missingness Summary Table"):
        summary_table = pd.DataFrame(
            {
                "Missing Values": df.isna().sum(),
                "Missing %": (df.isna().mean() * 100).round(2),
            }
        ).sort_values("Missing Values", ascending=False)

        st.dataframe(summary_table)

        # ---- Detect variables with too many missing values ----
        missing_pct = df.isna().mean()

        too_missing = missing_pct[missing_pct > 0.30]  # >30% missing

        if len(too_missing) > 0:
            st.warning(
                "⚠️ **Variables with > 30% missing values** — recommended to exclude:\n\n"
                + "\n".join(
                    [
                        f"- **{col}** ({pct:.1%} missing)"
                        for col, pct in too_missing.items()
                    ]
                )
            )
            st.markdown("""
            Depths **15 m** and **175 m** are especially sparse in the TAO dataset.
            These variables can distort imputation and correlation results and should be removed from analysis.
            """)

        # ---- ENSO Coverage Warning ----
        if "ANOM" in df.columns:
            enso_valid_dates = df[df["ANOM"].notna()]["date"]

            if len(enso_valid_dates) > 0:
                start_enso = enso_valid_dates.min().year
                end_enso = enso_valid_dates.max().year

                st.info(
                    f"📌 **ENSO Index Available:** {start_enso} → {end_enso}\n\n"
                    f"ENSO (`ANOM`) values only exist until **{end_enso}**. "
                    "ENSO-colored plots and anomaly analysis will automatically stop at this year."
                )
            else:
                st.error(
                    "❌ The ENSO index has no valid values — ENSO analysis will be skipped."
                )

    # =====================================================
    # 2) Missingness Heatmap
    # =====================================================
    with st.expander("🌡 Missingness Heatmap"):
        st.markdown("""
        The heatmap below shows missingness over time  
        (**yellow = missing**, **dark = present**).
        """)

        cols_for_heatmap = [c for c in df.columns if c not in ["year", "month"]]

        nan_array = df[cols_for_heatmap].isna().astype(int).to_numpy().T

        # High-DPI sharp rendering
        fig, ax = plt.subplots(figsize=(30, 15), dpi=150)

        mesh = ax.pcolormesh(
            nan_array,
            cmap="cividis",
            shading="nearest",  # Sharp pixels, no blur
        )

        ax.set_title("Missing Values Heatmap (1 = Missing)", fontsize=22, pad=20)
        ax.set_ylabel("Features", fontsize=18)
        ax.set_xlabel("Date", fontsize=18)

        # Y-axis feature labels
        ax.set_yticks(np.arange(len(cols_for_heatmap)) + 0.5)
        ax.set_yticklabels(cols_for_heatmap, fontsize=12)

        # Time ticks along X-axis
        n_rows = nan_array.shape[1]
        n_ticks = 12  # nice readable number
        tick_pos = np.linspace(0, n_rows - 1, n_ticks).astype(int)
        tick_labels = df.loc[tick_pos, "date"].dt.strftime("%Y-%m-%d")

        ax.set_xticks(tick_pos + 0.5)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=12)

        # Colorbar
        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label("Missingness", fontsize=18)
        cbar.ax.tick_params(labelsize=12)

        st.pyplot(fig)

    # =====================================================
    # Tab 2b: RF-MICE Imputation
    # =====================================================
    st.title("🤖 Random Forest MICE Imputation")

    st.markdown("""
    This section uses **Iterative Imputation** with a **Random Forest model**.
    Each variable with missing data is predicted using all other available variables.
    
    This method handles nonlinear relationships and is well-suited for climate data.
    """)

    # ---------------------------------------------------------
    # Step 1 — Variables to impute (EXCLUDING high-missing ones)
    # ---------------------------------------------------------
    variables_to_exclude = ["temp_15m", "temp_175m"]  # remove problematic depths

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
        "temp_200m",
        "temp_250m",
        "T_25",
    ]

    # Remove columns that shouldn't be imputed
    columns_to_impute = [c for c in columns_to_impute if c not in variables_to_exclude]
    # ENSO index should NEVER be imputed
    if "ANOM" in columns_to_impute:
        columns_to_impute.remove("ANOM")

    st.write("### Variables selected for imputation:")
    st.write(columns_to_impute)

    # ---------------------------------------------------------
    # Step 2 — Run Imputation
    # ---------------------------------------------------------
    if st.button("Run Imputation"):
        st.info("⏳ Imputation in progress... This may take **10–30 seconds**.")

        with st.spinner("Running Random Forest MICE imputation..."):
            # Enable IterativeImputer
            from sklearn.experimental import enable_iterative_imputer
            from sklearn.impute import IterativeImputer
            from sklearn.ensemble import RandomForestRegressor

            # Save original reference
            df_original = df.copy()

            # Mask missing values for each column
            missing_mask = df_original[columns_to_impute].isna()

            # Remove non-numeric columns (date, year, month)
            non_numeric_cols = df_original.select_dtypes(
                exclude=["float", "int"]
            ).columns
            numeric_df = df_original.drop(columns=non_numeric_cols)

            # Random Forest model
            rf = RandomForestRegressor(
                n_estimators=50, max_depth=10, n_jobs=-1, random_state=42
            )

            # Build the imputer
            imputer = IterativeImputer(estimator=rf, max_iter=5, random_state=42)

            # Fit and transform
            df_imp_numeric = pd.DataFrame(
                imputer.fit_transform(numeric_df), columns=numeric_df.columns
            )

            # Reattach non-numeric columns
            df_imputed = pd.concat(
                [df_imp_numeric, df_original[non_numeric_cols]], axis=1
            )

            # Save to Streamlit session
            st.session_state["df"] = df_imputed.copy()

        st.success("✅ Imputation completed successfully!")

        # ---------------------------------------------------------
        # Step 3 — Plot Original vs Imputed Values
        # ---------------------------------------------------------

        st.subheader("📊 Original vs Imputed Values")

        pretty_names = {
            "WU_422": "Zonal Wind (m/s)",
            "WV_423": "Meridional Wind (m/s)",
            "RH_910": "Relative Humidity (%)",
            "AT_21": "Air Temperature (°C)",
            "T_25": "Sea Surface Temperature (°C)",
            "temp_1m": "Temp @1m",
            "temp_10m": "Temp @10m",
            "temp_20m": "Temp @20m",
            "temp_50m": "Temp @50m",
            "temp_75m": "Temp @75m",
            "temp_100m": "Temp @100m",
            "temp_150m": "Temp @150m",
            "temp_200m": "Temp @200m",
            "temp_250m": "Temp @250m",
        }

        for col in columns_to_impute:
            fig, ax = plt.subplots(figsize=(14, 4))

            ax.plot(df_original["date"], df_original[col], alpha=0.4, label="Original")

            ax.scatter(
                df_original.loc[missing_mask[col], "date"],
                df_imputed.loc[missing_mask[col], col],
                s=12,
                color="orange",
                label="Imputed",
            )

            ax.set_title(f"{pretty_names.get(col, col)} — Original vs Imputed")
            ax.set_xlabel("Date")
            ax.set_ylabel(pretty_names.get(col, col))
            ax.legend()
            ax.grid(alpha=0.3)

            st.pyplot(fig)

        # ---------------------------------------------------------
        # Step 4 — Missingness Report
        # ---------------------------------------------------------
        st.subheader("🧮 Missing Values After Imputation")

        st.write("**Before:**")
        st.write(df_original[columns_to_impute].isna().sum())

        st.write("**After:**")
        st.write(df_imputed[columns_to_impute].isna().sum())

        st.success("🎉 All selected variables successfully imputed!")


# =====================================================
# Tab 3: Temporal Coverage
# =====================================================
elif choice == "Temporal Coverage":
    st.header("📆 Temporal Coverage & ENSO Influence")

    df = st.session_state["df"].copy()

    # Ensure datetime
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    st.markdown("""
This tab explores **how climate variables evolve over time**, especially in relation to  
**ENSO events (El Niño & La Niña)**.

- 🔴 **El Niño** → positive ENSO index, warming of SST  
- 🔵 **La Niña** → negative ENSO index, cooling of SST  
- ⚪ Neutral → weak anomalies  

Select a variable below to view how it behaves over time.
    """)

    # Available climate variables
    numeric_cols = ["T_25", "AT_21", "RH_910", "WU_422", "WV_423"]
    feature = st.selectbox(
        "Select variable to visualize:",
        numeric_cols,
        index=numeric_cols.index("T_25"),
    )

    # =============================
    # 1. ENSO-colored scatter plot
    # =============================
    st.subheader(f"📅 {feature} Over Time (ENSO-Colored Scatter)")

    df_plot = df.dropna(subset=[feature, "ANOM", "date"])
    anom_abs = max(abs(df_plot["ANOM"].min()), abs(df_plot["ANOM"].max()))

    fig_scatter = px.scatter(
        df_plot,
        x="date",
        y=feature,
        color="ANOM",
        color_continuous_scale="RdBu_r",
        opacity=0.5,
        title=f"{feature} Over Time (Colored by ENSO Index)",
    )

    fig_scatter.update_layout(
        coloraxis=dict(
            cmin=-anom_abs,
            cmax=anom_abs,
            cmid=0,
            colorscale="RdBu_r",
            colorbar=dict(title="ENSO Index"),
        ),
        template="plotly_white",
        title_x=0.5,
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

    # =============================
    # 2. ENSO-shaded line plot
    # =============================
    st.subheader(f"📈 Smoothed {feature} with ENSO Shading")

    show_shading = st.checkbox("Show ENSO shading", value=True)

    df_daily = df.groupby("date")[numeric_cols + ["ANOM"]].mean().reset_index()

    el_thresh, la_thresh = 1.0, -1.0
    df_daily["event"] = np.where(
        df_daily["ANOM"] > el_thresh,
        "El Niño",
        np.where(df_daily["ANOM"] < la_thresh, "La Niña", None),
    )

    # Build shading periods
    shading_periods = []
    current_event, start_date = None, None

    for _, row in df_daily.iterrows():
        event, date = row["event"], row["date"]

        if event != current_event:
            if current_event is not None:
                shading_periods.append(
                    {"event": current_event, "start": start_date, "end": date}
                )
            current_event, start_date = event, date

    if current_event is not None:
        shading_periods.append(
            {
                "event": current_event,
                "start": start_date,
                "end": df_daily["date"].iloc[-1],
            }
        )

    fig_line = go.Figure()

    if show_shading:
        for period in shading_periods:
            if period["event"] is None:
                continue
            color = (
                "rgba(255, 0, 0, 0.15)"
                if period["event"] == "El Niño"
                else "rgba(0, 0, 255, 0.15)"
            )
            fig_line.add_vrect(
                x0=period["start"],
                x1=period["end"],
                fillcolor=color,
                opacity=0.3,
                layer="below",
                line_width=0,
            )

    fig_line.add_trace(
        go.Scatter(
            x=df_daily["date"],
            y=df_daily[feature],
            mode="lines",
            line=dict(color="royalblue", width=1.4),
            name=feature,
        )
    )

    fig_line.update_layout(
        title=f"{feature} with ENSO Shading",
        template="plotly_white",
        title_x=0.5,
    )

    st.plotly_chart(fig_line, use_container_width=True)

    # ===================================================================
    # 3. TEMPERATURE AT MULTIPLE DEPTHS OVER TIME
    # ===================================================================
    st.subheader("🌡 Sea Temperature Over Time at Multiple Depths")

    # Keep only relevant depth columns
    depth_cols = [c for c in df.columns if c.startswith("temp_")]

    # Remove the high-missing ones
    depth_cols = [c for c in depth_cols if c not in ["temp_15m", "temp_175m"]]

    # Convert names → numeric depth
    depths = [int(col.split("_")[1].replace("m", "")) for col in depth_cols]
    colors = px.colors.sequential.Viridis[: len(depth_cols)]

    fig_depth = go.Figure()

    for i, (col, depth) in enumerate(zip(depth_cols, depths)):
        fig_depth.add_trace(
            go.Scatter(
                x=df["date"],
                y=df[col],
                mode="lines",
                name=f"{depth} m",
                line=dict(color=colors[i], width=1.7),
            )
        )

    fig_depth.update_layout(
        title="Sea Temperature Over Time at Multiple Depths",
        xaxis_title="Date",
        yaxis_title="Temperature (°C)",
        template="plotly_white",
        legend_title="Depth",
        height=500,
    )

    st.plotly_chart(fig_depth, use_container_width=True)

    # ===================================================================
    # 4. INTERACTIVE VERTICAL TEMPERATURE PROFILE
    # ===================================================================
    st.subheader("📉 Interactive Vertical Temperature Profile")

    df_profile = df.iloc[::7]  # every 7th day for speed
    depth_cols = [
        c
        for c in df.columns
        if c.startswith("temp_") and c not in ["temp_15m", "temp_175m"]
    ]
    depths = [int(col.split("_")[1].replace("m", "")) for col in depth_cols]

    fig_prof = go.Figure()

    # frames for animation
    frames = []
    for i in range(len(df_profile)):
        frame_vals = df_profile.iloc[i][depth_cols].values
        frames.append(
            go.Frame(
                data=[
                    go.Scatter(
                        x=frame_vals,
                        y=depths,
                        mode="lines+markers",
                    )
                ],
                name=str(i),
            )
        )

    # initial trace
    fig_prof.add_trace(
        go.Scatter(
            x=df_profile.iloc[0][depth_cols].values, y=depths, mode="lines+markers"
        )
    )

    fig_prof.update_layout(
        title="Vertical Temperature Profile (Animation)",
        xaxis_title="Temperature (°C)",
        yaxis_title="Depth (m)",
        yaxis_autorange="reversed",
        template="plotly_white",
        height=600,
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 40, "redraw": False}}],
                    }
                ],
            }
        ],
        sliders=[
            {
                "steps": [
                    {
                        "label": str(df_profile["date"].iloc[i].date()),
                        "method": "animate",
                        "args": [[str(i)], {"frame": {"duration": 0, "redraw": False}}],
                    }
                    for i in range(len(df_profile))
                ]
            }
        ],
    )

    fig_prof.frames = frames

    st.plotly_chart(fig_prof, use_container_width=True)


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
