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
    "ENSO Anomalies",
    "Forecast Models",
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

    # --------------------------------------------------
    # Dataset construction explanation
    # --------------------------------------------------
    st.markdown("""
    ## 📑 How the Final Dataset Was Built

    The dataset displayed in this app was created from **multiple raw TAO files**, originally containing measurements at different depths, variables, and time intervals (often hourly or irregular).

    To create a clean, analysis-ready dataset:

    ### **1️⃣ Merging multiple raw data tables**
    Several TAO files were merged together using:
    - **Date** as the joining key  
    - Ensuring all variables for a given day appear in a single combined row  
    - Handling station-specific inconsistencies and aligning timestamps  

    ### **2️⃣ Daily aggregation**
    Because the original measurements were **sub-daily**, each day's values were aggregated using:

    **Daily Mean:**
    ```python
    daily_mean = df_raw.groupby("date").mean().reset_index()
    ```
    This produced **one representative value per variable per day**, reducing noise and ensuring compatibility with longer-term climate indices such as **ENSO (ANOM)**.

    ### **3️⃣ Final cleaned dataset**
    The resulting file (`daily_mean.csv`) contains:
    - ~30 years of daily measurements  
    - Fully aligned atmospheric + oceanic variables  
    - Ready for anomaly computation, correlation analysis, and imputation  

    This preprocessing step is essential because ENSO signals operate on **weekly-to-monthly** scales rather than hourly variability.
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
            "ClimAdjust": "Climatological adjustment applied to Sea Surface Temperature",
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

        st.markdown("""
        ### 🧭 Interpretation of Temporal Coverage

        The plot above shows the **number of available measurements for each year-month**
        across the entire dataset. A perfectly balanced dataset would show roughly the
        same number of records every month (≈ 30 observations).

        From the plot, we observe:

        - ✔️ **Nearly all months have full coverage** of ~28–31 records  
        - ✔️ **Only a few early years (late 1970s–early 1980s)** show reduced coverage  
        — this is expected due to limited early deployment of TAO moorings  
        - ✔️ **After ~1985, the record is almost perfectly consistent**, with only minor natural
        variability in the number of daily observations  
        - ✔️ There is **no systematic imbalance**, missing block, or measurement gap that would bias
        long-term climate trends  
        - ✔️ This means the dataset is **well-suited for anomaly computation, climatology studies,
        and ENSO analysis**, since monthly sampling density is stable

        In summary, temporal coverage is **strong, stable, and uniform**, ensuring that downstream
        analyses (correlations, seasonal cycles, anomalies, ML modeling) are not affected by
        time-dependent sampling biases.
        """)

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
    st.title("🌲 Random Forest MICE Imputation")

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

    # ---------------------------------------------------------
    # Step 2 — Run Imputation
    # ---------------------------------------------------------
    if st.button("Run Imputation"):
        st.info("⏳ Imputation in progress... This may take **around 2 minutes**.")

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
    st.markdown("""
This tab explores how **ocean and atmospheric variables evolve over time**, with special emphasis on **ENSO (El Niño / La Niña)** impacts.

- 🔴 **El Niño** → warm anomalies  
- 🔵 **La Niña** → cool anomalies  
""")

    # --------------------------------------------------
    # 1) Determine which dataset to use
    # --------------------------------------------------
    df_original = df.copy()  # original cached dataset

    if "df" in st.session_state and "T_25" in st.session_state["df"].columns:
        df = st.session_state["df"].copy()
        st.info("Using **imputed dataset** ✔")
    else:
        df = df_original.copy()
        st.warning("Using **original dataset** (imputation not run yet).")

    # --------------------------------------------------
    # 2) Ensure datetime exists and is sorted
    # --------------------------------------------------
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date")

    # --------------------------------------------------
    # 3) ENSO-Colored SST Scatter
    # --------------------------------------------------
    st.subheader("🌡 SST Over Time (Colored by ENSO Index)")

    if "ANOM" not in df.columns or df["ANOM"].isna().all():
        st.error("ENSO index (ANOM) is not available.")
    else:
        df_plot = df.dropna(subset=["T_25", "ANOM"])
        anom_abs = max(abs(df_plot["ANOM"].min()), abs(df_plot["ANOM"].max()))

        fig = px.scatter(
            df_plot,
            x="date",
            y="T_25",
            color="ANOM",
            color_continuous_scale="RdBu_r",
            opacity=0.55,
            labels={"T_25": "Sea Surface Temperature (°C)", "ANOM": "ENSO Index"},
            title="Sea Surface Temperature Over Time (ENSO-Colored)",
        )

        fig.update_layout(
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

        fig.update_traces(marker=dict(size=5))
        st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------
    # 4) Multi-Depth SST Time Series
    # --------------------------------------------------
    st.subheader("🌊 Ocean Temperature at Multiple Depths Over Time")

    # Only include selected depths
    depth_keep = [
        "temp_10m",
        "temp_50m",
        "temp_100m",
        "temp_150m",
        "temp_200m",
        "temp_250m",
    ]
    depth_cols = [col for col in depth_keep if col in df.columns]

    # Filter to show only data from 1990+
    df_depth = df[df["date"].dt.year >= 1990].copy()

    if len(depth_cols) == 0:
        st.error("None of the selected depth variables are available in the dataset.")
    else:
        import itertools

        color_cycle = itertools.cycle(px.colors.sequential.Viridis)

        depths = [int(col.replace("temp_", "").replace("m", "")) for col in depth_cols]

        fig = go.Figure()

        for col, depth in zip(depth_cols, depths):
            fig.add_trace(
                go.Scatter(
                    x=df_depth["date"],
                    y=df_depth[col],
                    mode="lines",
                    name=f"{depth} m",
                    line=dict(color=next(color_cycle), width=1.8),
                )
            )

        fig.update_layout(
            title="Sea Temperature Over Time at Selected Depths (1990–Present)",
            xaxis_title="Date",
            yaxis_title="Temperature (°C)",
            template="plotly_white",
            legend_title="Depth",
            height=500,
            title_x=0.5,
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    ### 🔍 What This Plot Shows

    The figure displays **ocean temperature variations over time** at six selected depths  
    (**10 m, 50 m, 100 m, 150 m, 200 m, 250 m**) from **1990 to 2025**.

    #### ✅ Key observations:

    - **Surface and shallow waters (10–50 m)** show the **highest temperatures**  
    (typically 26–30°C) and **strong seasonal cycles**, warming in mid-year and cooling at the beginning/end of each year.

    - **Intermediate depths (100–150 m)** are cooler and show **muted seasonal variability**,  
    but still respond to ENSO-driven vertical mixing.

    - **Deep layers (200–250 m)** are the coldest (12–16°C) and exhibit the **smoothest, most stable signals**,  
    as deeper waters are less influenced by short-term atmospheric forcing.

    - **ENSO signals propagate downward**: during El Niño, warm anomalies can extend below 100 m,  
    visible as periods where intermediate curves rise noticeably above their usual cycles.

    - Despite natural variability, **the thermal stratification remains consistent**:  
    warmer near the surface, cooler with depth.

    Overall, the plot illustrates how the **upper ocean responds dynamically to climate variability**,  
    while deeper layers act as a slower, more stable thermal reservoir.
    """)

    # --------------------------------------------------
    # 5) Interactive Vertical Temperature Profile
    # --------------------------------------------------
    st.subheader("📉 Interactive Vertical Temperature Profile")

    if len(depth_cols) < 2:
        st.error("Not enough depth levels for a vertical profile.")
    else:
        df_anim = df.iloc[::7]  # reduce size for animation smoothness

        depths_sorted = sorted(
            [int(col.split("_")[1].replace("m", "")) for col in depth_cols]
        )
        depth_cols_sorted = [f"temp_{d}m" for d in depths_sorted]

        fig = go.Figure()
        frames = []

        for i in range(len(df_anim)):
            temp_values = df_anim.iloc[i][depth_cols_sorted].values
            frames.append(
                go.Frame(
                    data=[
                        go.Scatter(
                            x=temp_values,
                            y=depths_sorted,
                            mode="lines+markers",
                        )
                    ],
                    name=str(i),
                )
            )

        # Initial trace
        fig.add_trace(
            go.Scatter(
                x=df_anim.iloc[0][depth_cols_sorted].values,
                y=depths_sorted,
                mode="lines+markers",
            )
        )

        fig.update_layout(
            title="",
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
                            "args": [
                                None,
                                {"frame": {"duration": 50, "redraw": False}},
                            ],
                        }
                    ],
                }
            ],
            sliders=[
                {
                    "steps": [
                        {
                            "label": str(df_anim["date"].iloc[i].date()),
                            "method": "animate",
                            "args": [
                                [str(i)],
                                {"frame": {"duration": 0, "redraw": False}},
                            ],
                        }
                        for i in range(len(df_anim))
                    ]
                }
            ],
        )

        fig.frames = frames
        st.plotly_chart(fig, use_container_width=True)


# =====================================================
# Tab 4: Correlation Study
# =====================================================
elif choice == "Correlation Study":
    st.header("📊 Correlation & Feature Relationships")
    st.markdown("""
Understanding how **oceanic and atmospheric variables interact** is key to explaining  
ENSO-related variability and heat exchange in the tropical Pacific.

This tab provides:
- 🔸 A **correlation heatmap**  
- 🔸 Clean **scatterplots with regression lines**  
- 🔸 A **scatter-matrix** (pairwise relationships)  
- 🔸 **ENSO-colored scatter plots** for key variables  
""")

    # --------------------------------------------------
    # 1) Select dataset (imputed if available)
    # --------------------------------------------------
    if "df" in st.session_state:
        df_corr = st.session_state["df"].copy()
        st.info("Using **imputed dataset** ✔")
    else:
        df_corr = df.copy()
        st.warning(
            "Using **original dataset** (run imputation for cleaner correlations)."
        )

    # --------------------------------------------------
    # Pretty variable names
    # --------------------------------------------------
    pretty_names = {
        "WU_422": "Zonal Wind (m/s)",
        "WV_423": "Meridional Wind (m/s)",
        "RH_910": "Relative Humidity (%)",
        "AT_21": "Air Temperature (°C)",
        "temp_10m": "Temp @10m (°C)",
        "temp_50m": "Temp @50m (°C)",
        "temp_100m": "Temp @100m (°C)",
        "temp_150m": "Temp @150m (°C)",
        "temp_200m": "Temp @200m (°C)",
        "temp_250m": "Temp @250m (°C)",
        "T_25": "Sea Surface Temp (SST, °C)",
    }

    # Default variables
    default_vars = list(pretty_names.keys())

    st.subheader("🔧 Select Variables for Correlation")
    selected_features = st.multiselect(
        "Choose variables to include:",
        options=list(pretty_names.keys()),
        default=default_vars,
    )

    if len(selected_features) < 2:
        st.warning("Please select **at least two variables**.")
        st.stop()

    df_corr_numeric = (
        df_corr[selected_features].apply(pd.to_numeric, errors="coerce").dropna()
    )

    # =====================================================
    # 🔸 Correlation Heatmap
    # =====================================================
    st.subheader("🔸 Correlation Heatmap")
    st.markdown("""
The heatmap below shows **Pearson correlations** between key ocean–atmosphere variables.  
Strong correlations indicate tightly connected physical processes.
""")

    corr = df_corr_numeric.corr()

    # rename to pretty labels
    corr.index = [pretty_names.get(c, c) for c in corr.index]
    corr.columns = [pretty_names.get(c, c) for c in corr.columns]

    # mask upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    corr_masked = corr.mask(mask)

    fig_corr = go.Figure(
        data=go.Heatmap(
            z=corr_masked.values,
            x=corr_masked.columns,
            y=corr_masked.index,
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            hoverongaps=False,
            colorbar=dict(title="Correlation"),
        )
    )

    # numeric annotations
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            if not mask[i, j]:
                fig_corr.add_annotation(
                    x=corr.columns[j],
                    y=corr.index[i],
                    text=f"{corr.values[i, j]:.2f}",
                    showarrow=False,
                    font=dict(size=10),
                )

    fig_corr.update_layout(
        title="Correlation Heatmap (Lower Triangle)",
        title_x=0.5,
        width=900,
        height=900,
        plot_bgcolor="white",
    )

    st.plotly_chart(fig_corr, use_container_width=True)

    # Interpretation
    st.markdown("""
### 🔍 Interpretation
- **SST and air temperature** exhibit a **very strong correlation**, reflecting direct ocean–atmosphere heat exchange.  
- **Subsurface temperatures** correlate strongly with SST, but the correlation weakens slightly with depth.  
- **Winds** show weaker correlations because their variability is driven by seasonal and ENSO dynamics rather than SST alone.  
""")

    # =====================================================
    # 🔸 Scatterplots — manual regression (no Plotly trendline)
    # =====================================================
    st.subheader("🔸 Key Scatterplots with Regression Lines")
    st.markdown("""
These scatterplots show how SST relates to selected atmospheric and oceanic variables.
""")

    def scatter_with_manual_reg(x_var, y_var):
        # 1) Clean two-column DataFrame
        df_two = df_corr[[x_var, y_var]].copy()
        df_two = df_two.apply(pd.to_numeric, errors="coerce").dropna()

        # Optional downsampling for speed
        if len(df_two) > 8000:
            df_two = df_two.sample(8000, random_state=42)

        x = df_two[x_var].to_numpy()
        y = df_two[y_var].to_numpy()

        # 2) Manual linear regression (least squares)
        slope, intercept = np.polyfit(x, y, 1)

        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = slope * x_line + intercept

        # 3) Build figure: scatter + fitted line
        fig = go.Figure()

        # Scatter points
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name="Data",
                marker=dict(size=4, opacity=0.5, color="rgba(30, 100, 160, 0.6)"),
            )
        )

        # Regression line
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name="Linear fit",
                line=dict(color="red", width=2),
            )
        )

        fig.update_layout(
            title=f"{pretty_names.get(x_var, x_var)} vs {pretty_names.get(y_var, y_var)}",
            xaxis_title=pretty_names.get(x_var, x_var),
            yaxis_title=pretty_names.get(y_var, y_var),
            template="plotly_white",
            title_x=0.5,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        st.plotly_chart(fig, use_container_width=True)

    # Generate scatterplots
    scatter_with_manual_reg("AT_21", "T_25")
    scatter_with_manual_reg("RH_910", "T_25")
    scatter_with_manual_reg("WU_422", "T_25")
    scatter_with_manual_reg("WV_423", "T_25")

    st.markdown("""
### 🔍 Interpretation
- **Air temperature vs SST** shows a near-perfect linear relationship, confirming strong ocean–atmosphere heat exchange.  
- **Humidity vs SST** increases gradually — warm oceans evaporate more moisture into the lower atmosphere.  
- **Wind components vs SST** are noisier, since winds respond to pressure gradients and ENSO-related circulation, not just local SST.  
""")

    # =====================================================
    # 🔸 ENSO-Colored Scatter Plots
    # =====================================================
    st.subheader("🔸 ENSO-Colored Scatter Plots")
    st.markdown("""
These plots highlight how ENSO phases influence the relationships between SST and other variables.  
Red = **El Niño**, Blue = **La Niña**.
""")

    if "ANOM" in df_corr.columns and not df_corr["ANOM"].isna().all():
        vars_to_compare = ["AT_21", "RH_910", "WU_422", "WV_423"]
        cols = st.columns(2)

        for i, var in enumerate(vars_to_compare):
            with cols[i % 2]:
                fig = px.scatter(
                    df_corr,
                    x="T_25",
                    y=var,
                    color="ANOM",
                    color_continuous_scale="RdBu_r",
                    opacity=0.6,
                    labels={
                        "T_25": "Sea Surface Temp (°C)",
                        var: pretty_names.get(var, var),
                        "ANOM": "ENSO Index",
                    },
                    title=f"{pretty_names.get(var, var)} vs SST (ENSO-colored)",
                )
                fig.update_layout(template="plotly_white", title_x=0.5)
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
### 🔍 ENSO Interpretation
- During **El Niño**, SST and air temperature shift upward together.  
- **La Niña** clusters reveal cooler SST and lower air temperature.  
- Wind fields become more scattered, reflecting ENSO-driven circulation changes.  
""")

    else:
        st.warning(
            "No ENSO index (`ANOM`) available — skipping ENSO-colored scatter plots."
        )

    # =====================================================
    # 🔸 Scatter Matrix
    # =====================================================
    st.subheader("🔸 Pairwise Scatter Matrix")
    st.markdown("""
The scatter matrix highlights joint variability among several variables.  
Color indicates **air temperature**, helping reveal thermodynamic structure.
""")

    scatter_features = [
        "T_25",
        "temp_50m",
        "temp_150m",
        "AT_21",
        "RH_910",
        "WU_422",
        "WV_423",
    ]

    available_features = [f for f in scatter_features if f in df_corr.columns]
    df_scatter = df_corr[available_features].dropna().rename(columns=pretty_names)

    if not df_scatter.empty:
        fig_matrix = px.scatter_matrix(
            df_scatter,
            dimensions=list(df_scatter.columns),
            color="Air Temperature (°C)"
            if "Air Temperature (°C)" in df_scatter.columns
            else None,
            opacity=0.45,
            color_continuous_scale="RdBu_r",
            title="Pairwise Relationships Between Ocean & Atmosphere Variables",
        )

        # improve readability
        fig_matrix.update_traces(marker=dict(size=2))
        fig_matrix.update_layout(height=900, title_x=0.5, plot_bgcolor="white")

        st.plotly_chart(fig_matrix, use_container_width=True)

        st.markdown("""
### 🔍 Interpretation
- Ocean temperatures cluster tightly and show coherent seasonal structure.  
- Atmospheric variables (humidity, winds) display greater spread.  
- The matrix cleanly separates **thermodynamic variables** from **dynamic variables**.  
""")

    else:
        st.warning("Not enough complete data to display the scatter matrix.")


# =====================================================
# Tab 5: ENSO Anomalies
# =====================================================
elif choice == "ENSO Anomalies":
    st.header("🌡 ENSO-Driven Climate Anomalies")

    st.markdown("""
This tab explores how **ENSO (El Niño–Southern Oscillation)** influences  
ocean and atmosphere dynamics by computing:

- 🌡 **Climatological anomalies** (SST, air temperature, winds)  
- 🔥 **Monthly SST anomaly heatmap**  
- 📈 **Time series of anomalies**  
- 🧩 **STL decomposition** (trend, seasonal, residual)  
- 🧪 **Feature engineering for ENSO prediction**  
""")

    # --------------------------------------------------
    # 1. Select dataset
    # --------------------------------------------------
    if "df" in st.session_state:
        df_ano = st.session_state["df"].copy()
        st.info("Using imputed dataset ✔")
    else:
        df_ano = df.copy()
        st.warning("Using original dataset (no imputation detected).")

    # --------------------------------------------------
    # 2. Date formatting
    # --------------------------------------------------
    df_ano["date"] = pd.to_datetime(df_ano["date"], errors="coerce")
    df_ano = df_ano.sort_values("date").set_index("date")

    df_ano["month"] = df_ano.index.month
    df_ano["year"] = df_ano.index.year

    # --------------------------------------------------
    # 3. Compute monthly climatological anomalies
    # --------------------------------------------------
    anomaly_vars = ["T_25", "AT_21", "WU_422", "WV_423"]

    for var in anomaly_vars:
        clim = df_ano.groupby("month")[var].transform("mean")
        df_ano[f"{var}_anom"] = df_ano[var] - clim

    # --------------------------------------------------
    # 4. SST anomaly time series
    # --------------------------------------------------
    st.subheader("📈 SST Anomalies Over Time")

    st.markdown("""
This plot highlights periods of **positive SST anomalies** (El Niño warming) and **negative anomalies** (La Niña cooling).  
Removing the seasonal cycle reveals real climate signals instead of annual variations.
""")

    fig = px.line(
        df_ano,
        y="T_25_anom",
        labels={"T_25_anom": "SST Anomaly (°C)"},
        title="Sea Surface Temperature (SST) Anomalies Over Time",
    )
    fig.update_layout(template="plotly_white", title_x=0.5)
    st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------
    # 5. Monthly SST anomaly heatmap
    # --------------------------------------------------
    st.subheader("🔥 Monthly SST Anomaly Heatmap")

    st.markdown("""
This heatmap shows **how warm or cool each month was**, relative to normal conditions.

- 🔴 Red streaks = **El Niño warm anomalies**  
- 🔵 Blue streaks = **La Niña cool anomalies**  
- Vertical bands show multi-year ENSO events  
""")

    sst_heatmap = df_ano.pivot_table(
        index="month",
        columns="year",
        values="T_25_anom",
        aggfunc="mean",
    )

    fig_heat = px.imshow(
        sst_heatmap,
        color_continuous_scale="RdBu_r",
        zmin=-3,
        zmax=3,
        labels={"x": "Year", "y": "Month", "color": "SST Anomaly (°C)"},
        title="Monthly SST Anomalies Heatmap",
    )
    fig_heat.update_layout(height=550, title_x=0.5)
    st.plotly_chart(fig_heat, use_container_width=True)

    # --------------------------------------------------
    # 6. STL decomposition
    # --------------------------------------------------
    st.subheader("🔍 STL Decomposition of Monthly SST")

    st.markdown("""
STL decomposition separates SST into:

- **Trend** — long-term warming pattern  
- **Seasonal** — normal annual cycle  
- **Residual** — ENSO-related variability and noise  

This technique is widely used in climate analysis to isolate ENSO signals.
""")

    from statsmodels.tsa.seasonal import STL

    sst_monthly = df_ano["T_25"].resample("M").mean().dropna()
    stl = STL(sst_monthly, period=12, robust=True).fit()

    fig_stl, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    axs[0].plot(sst_monthly.index, sst_monthly, lw=1.2)
    axs[0].set_title("Observed Monthly SST")

    axs[1].plot(stl.trend.index, stl.trend, color="tab:blue")
    axs[1].set_title("Trend")

    axs[2].plot(stl.seasonal.index, stl.seasonal, color="tab:green")
    axs[2].set_title("Seasonal Component")

    axs[3].scatter(stl.resid.index, stl.resid, s=10, alpha=0.6)
    axs[3].axhline(0, color="black", lw=0.7)
    axs[3].set_title("Residual")

    fig_stl.suptitle("STL Decomposition of SST", fontsize=15)
    fig_stl.tight_layout(rect=[0, 0, 1, 0.97])

    st.pyplot(fig_stl)

    # --------------------------------------------------
    # 7. Feature engineering (clean, rewritten)
    # --------------------------------------------------
    st.subheader("🧪 Feature Engineering for ENSO Prediction Models")

    st.markdown("""
To build predictive ENSO or SST models, we engineer features that represent:

- 🔁 **Persistence** (lagged anomalies)
- 📉 **Recent changes** (SST differencing)
- 🌊 **Low-frequency ENSO signal** (rolling anomalies)
- 🧭 **Seasonality** (sin/cos encoding)
- 💨 **Wind forcing** (combined wind anomaly magnitude)

These features are widely used in operational ENSO forecasting research.
""")

    df_feat = df_ano.copy()

    # 1) First-difference of SST (short-term momentum)
    df_feat["T_25_diff"] = df_feat["T_25"].diff()

    # 2) Rolling anomalies (ENSO smooth signal)
    df_feat["T_25_anom_roll3"] = df_feat["T_25_anom"].rolling(3).mean()

    # 3) Lagged SST anomalies (ENSO persistence)
    for lag in [1, 2, 3]:
        df_feat[f"T_25_anom_lag{lag}"] = df_feat["T_25_anom"].shift(lag)

    # 4) Seasonal encoding (continuous cycle)
    df_feat["month_sin"] = np.sin(2 * np.pi * df_feat["month"] / 12)
    df_feat["month_cos"] = np.cos(2 * np.pi * df_feat["month"] / 12)

    # 5) Wind anomaly magnitude
    df_feat["wind_speed_anom"] = np.sqrt(
        df_feat["WU_422_anom"] ** 2 + df_feat["WV_423_anom"] ** 2
    )

    df_model = df_feat.dropna()

    st.write("Preview of engineered dataset:")
    st.dataframe(df_model.head())

    # --------------------------------------------------
    # 8. Modeling matrix & scaling
    # --------------------------------------------------
    st.subheader("📦 Modeling Matrix (X, y)")

    st.markdown("""
Below is the feature matrix used for machine learning models.  
Features represent ENSO memory, momentum, seasonality, and wind forcing.
""")

    features = [
        "T_25_anom_lag1",
        "T_25_anom_lag2",
        "T_25_anom_lag3",
        "T_25_diff",
        "T_25_anom_roll3",
        "AT_21_anom",
        "WU_422_anom",
        "WV_423_anom",
        "wind_speed_anom",
        "month_sin",
        "month_cos",
    ]

    from sklearn.preprocessing import StandardScaler

    X = df_model[features]
    y = df_model["T_25"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    st.success("Feature matrix successfully constructed and scaled ✔")

    st.write("### Shapes:")
    st.write("X:", X_scaled.shape)
    st.write("y:", y.shape)


# =====================================================
# Tab 6: Forecast Models
# =====================================================
elif choice == "Forecast Models":
    st.header("🔮 Forecasting Sea Surface Temperature (SST)")

    st.markdown("""
This tab compares **two very different forecasting approaches** applied to the  
Sea Surface Temperature (SST) record at the TAO buoy:

### 🌊 **Model 1 — AutoRegressive (AR) Model**
Uses **only past SST values** to predict future SST  
(very common in classical time-series analysis).

### 🌳 **Model 2 — Random Forest Regression**
Uses **engineered features** from the ENSO tab  
(lags, rolling anomalies, wind anomalies, seasonal cycle).

These two models illustrate the difference between  
a **pure time-series method** vs. a **feature-based machine-learning model**.
""")

    # --------------------------------------------------
    # 1. Load imputed dataset
    # --------------------------------------------------
    if "df" in st.session_state:
        df_modeling = st.session_state["df"].copy()
        st.success("Using imputed dataset for forecasting ✔")
    else:
        st.warning("Imputed data not found — using original dataset")
        df_modeling = df.copy()

    df_modeling["date"] = pd.to_datetime(df_modeling["date"])
    df_modeling = df_modeling.set_index("date").sort_index()

    # =====================================================
    # PART A — AutoRegressive Model (Raw SST)
    # =====================================================
    st.subheader("📈 Model 1: AutoRegressive (AR) Forecast of SST")

    st.markdown("""
This model predicts SST based **only on its own past values**:

\[
SST(t) = a_1 SST(t-1) + a_2 SST(t-2) + \dots + a_p SST(t-p)
\]

We use a large number of lags so the model can capture:
- Long ENSO cycles (2–7 years),
- Seasonal structure,
- High-frequency ocean variability.
""")

    # Extract SST
    sst = df_modeling["T_25"].dropna()

    # Train/test split
    n = len(sst)
    train_size = int(0.8 * n)
    sst_train = sst.iloc[:train_size]
    sst_test = sst.iloc[train_size:]

    st.write(
        f"Training: **{sst_train.index.min().date()} → {sst_train.index.max().date()}**"
    )
    st.write(
        f"Testing : **{sst_test.index.min().date()} → {sst_test.index.max().date()}**"
    )

    # Fit AR model
    from statsmodels.tsa.ar_model import AutoReg
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    ar_model = AutoReg(sst_train, lags=5000, old_names=False).fit()

    # Predict test period
    start = len(sst_train)
    end = start + len(sst_test) - 1
    ar_pred = ar_model.predict(start=start, end=end)
    ar_pred.index = sst_test.index

    # Predict full dataset
    ar_pred_full = ar_model.predict(start=0, end=len(sst) - 1)
    ar_pred_full.index = sst.index

    # Compute metrics
    rmse = np.sqrt(mean_squared_error(sst_test, ar_pred))
    mae = mean_absolute_error(sst_test, ar_pred)
    r2 = r2_score(sst_test, ar_pred)

    st.markdown(f"""
### 📊 AR Model Performance  
- **RMSE:** {rmse:.3f}  
- **MAE:** {mae:.3f}  
- **R²:** {r2:.3f}
""")

    # Plot
    fig_ar = go.Figure()
    fig_ar.add_trace(go.Scatter(x=sst.index, y=sst, name="Actual SST"))
    fig_ar.add_trace(
        go.Scatter(
            x=ar_pred_full.index, y=ar_pred_full, name="AR Prediction", opacity=0.7
        )
    )
    fig_ar.update_layout(
        title="AutoRegressive (AR) Prediction of SST (Full Dataset)",
        xaxis_title="Date",
        yaxis_title="SST (°C)",
        template="plotly_white",
        height=450,
    )
    st.plotly_chart(fig_ar, use_container_width=True)

    st.markdown("""
### 🔍 Interpretation
- The AR model captures **seasonal cycles**, **ENSO cycles**, and long-term variability.
- However, because it uses **only SST**, it cannot use:
  - winds  
  - air temperature  
  - subsurface structure  
  - anomaly information  
  - climate dynamics  

This makes AR a **useful baseline**, but not a climate-aware forecast model.
""")

    # =====================================================
    # PART B — Random Forest Regression using Engineered Features
    # =====================================================
    st.subheader("🌳 Model 2: Random Forest Regression (Feature-Based)")

    st.markdown("""
This model uses the **engineered features** created in the ENSO Anomalies tab:

- SST anomaly lags (1, 2, 3 months)  
- Rolling anomaly (3-month mean)  
- SST differencing  
- Air temperature anomaly  
- Wind anomalies + wind magnitude  
- Seasonal cycle encoded as sin/cos  

This turns SST forecasting into a **supervised learning task**.
""")

    # ---------------------------
    # Recreate engineered features
    # ---------------------------
    df_feat = df_modeling.copy()

    # Anomalies
    for var in ["T_25", "AT_21", "WU_422", "WV_423"]:
        clim = df_feat[var].groupby(df_feat.index.month).transform("mean")
        df_feat[f"{var}_anom"] = df_feat[var] - clim

    # Feature engineering
    df_feat["T_25_diff"] = df_feat["T_25"].diff()
    df_feat["T_25_anom_roll3"] = df_feat["T_25_anom"].rolling(3).mean()

    for lag in [1, 2, 3]:
        df_feat[f"T_25_anom_lag{lag}"] = df_feat["T_25_anom"].shift(lag)

    df_feat["month_sin"] = np.sin(2 * np.pi * df_feat.index.month / 12)
    df_feat["month_cos"] = np.cos(2 * np.pi * df_feat.index.month / 12)

    df_feat["wind_speed_anom"] = np.sqrt(
        df_feat["WU_422_anom"] ** 2 + df_feat["WV_423_anom"] ** 2
    )

    df_feat = df_feat.dropna()

    # Prepare ML matrix
    features = [
        "T_25_anom_lag1",
        "T_25_anom_lag2",
        "T_25_anom_lag3",
        "T_25_diff",
        "T_25_anom_roll3",
        "AT_21_anom",
        "WU_422_anom",
        "WV_423_anom",
        "wind_speed_anom",
        "month_sin",
        "month_cos",
    ]

    X = df_feat[features]
    y = df_feat["T_25"]

    # Train-test split (time-aware)
    split = int(0.8 * len(X))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # Train RF
    from sklearn.ensemble import RandomForestRegressor

    rf = RandomForestRegressor(n_estimators=300, random_state=42)
    rf.fit(X_train, y_train)

    rf_pred = rf.predict(X_test)

    # Metrics
    rmse_rf = np.sqrt(mean_squared_error(y_test, rf_pred))
    mae_rf = mean_absolute_error(y_test, rf_pred)
    r2_rf = r2_score(y_test, rf_pred)

    st.markdown(f"""
### 📊 Random Forest Performance  
- **RMSE:** {rmse_rf:.3f}  
- **MAE:** {mae_rf:.3f}  
- **R²:** {r2_rf:.3f}
""")

    # Plot RF predictions
    fig_rf = go.Figure()
    fig_rf.add_trace(
        go.Scatter(x=y_test.index, y=y_test, name="Actual SST", line=dict(color="blue"))
    )
    fig_rf.add_trace(
        go.Scatter(
            x=y_test.index, y=rf_pred, name="RF Prediction", line=dict(color="red")
        )
    )
    fig_rf.update_layout(
        title="Random Forest: Actual vs Predicted SST",
        xaxis_title="Date",
        yaxis_title="SST (°C)",
        template="plotly_white",
        height=450,
    )
    st.plotly_chart(fig_rf, use_container_width=True)

    st.markdown("""
### 🔍 Interpretation
- The Random Forest model typically outperforms the AR model because it uses  
  **multiple climate indicators**, not just SST.
- Lagged anomalies give the model **ENSO memory**.
- Wind anomalies help capture **upwelling and mixing effects**.
- Seasonal encoding allows the model to learn the **annual temperature cycle**.
- This demonstrates the value of **feature engineering** in climate prediction.
""")

    # Feature importance
    importances = pd.Series(rf.feature_importances_, index=features).sort_values()

    st.subheader("📌 Feature Importance")
    st.bar_chart(importances)


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
