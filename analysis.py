import pandas as pd
import streamlit as st
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.power import TTestIndPower


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="Just Luck - Data Analysis",
    layout="wide"
)

st.title("Just Luck – Experimental Data Analysis")


# =========================================================
# LOAD RAW DATA
# =========================================================

FILE = "results-survey642194.csv"

# Variables needed for balance table / controls
cols_balance = [5, 20, 21, 22, 23, 24, 25, 26]

df_balance = pd.read_csv(
    FILE,
    usecols=cols_balance
)

df_balance.columns = [
    "version",
    "political_orientation",
    "financial_risk_willingness",
    "success_determinants",
    "education",
    "age",
    "gender",
    "salary"
]

# Participant ID
df_balance["id"] = df_balance.index

# Make sure version is numeric
df_balance["version"] = pd.to_numeric(
    df_balance["version"],
    errors="coerce"
)


# =========================================================
# EXPERIMENTAL DATA
# =========================================================

cols_experiment = [
    5, 7, 8, 9, 10, 11, 12,
    14, 15, 16, 17, 18, 19
]

df_exp = pd.read_csv(
    FILE,
    usecols=cols_experiment
)

df_exp.columns = [
    "version",
    "v1_90_noninv",
    "v1_90_inv",
    "v1_30_noninv",
    "v1_30_inv",
    "v1_60_unlucky",
    "v1_60_lucky",
    "v2_90_inv",
    "v2_90_noninv",
    "v2_30_inv",
    "v2_30_noninv",
    "v2_60_unlucky",
    "v2_60_lucky"
]

# Same participant ID
df_exp["id"] = df_exp.index

df_exp["version"] = pd.to_numeric(
    df_exp["version"],
    errors="coerce"
)


# =========================================================
# CONVERT EXPERIMENTAL DATA TO LONG FORMAT
# =========================================================

v1_cols = [
    "id"
] + [
    c for c in df_exp.columns
    if c.startswith("v1_")
]

v2_cols = [
    "id"
] + [
    c for c in df_exp.columns
    if c.startswith("v2_")
]

df_v1 = df_exp[v1_cols].copy()
df_v2 = df_exp[v2_cols].copy()


# ---------------------------------------------------------
# MELT
# ---------------------------------------------------------

df_v1_long = df_v1.melt(
    id_vars="id",
    var_name="scenario",
    value_name="value"
)

df_v2_long = df_v2.melt(
    id_vars="id",
    var_name="scenario",
    value_name="value"
)


# =========================================================
# EXTRACT SCENARIO INFORMATION
# =========================================================

for d in [df_v1_long, df_v2_long]:

    d[["v", "amount", "condition"]] = (
        d["scenario"]
        .str.split("_", expand=True)
    )

    d["amount"] = pd.to_numeric(
        d["amount"],
        errors="coerce"
    )

    d["value"] = pd.to_numeric(
        d["value"],
        errors="coerce"
    )


# =========================================================
# REMOVE MISSING OBSERVATIONS
# =========================================================

df_v1_long = df_v1_long.dropna(
    subset=["value"]
)

df_v2_long = df_v2_long.dropna(
    subset=["value"]
)


# =========================================================
# PIVOT
# ONE ROW = ONE PARTICIPANT × ONE AMOUNT
# =========================================================

df_v1_clean = (
    df_v1_long
    .pivot_table(
        index=["id", "amount"],
        columns="condition",
        values="value",
        aggfunc="first"
    )
    .reset_index()
)

df_v2_clean = (
    df_v2_long
    .pivot_table(
        index=["id", "amount"],
        columns="condition",
        values="value",
        aggfunc="first"
    )
    .reset_index()
)


# Remove possible column-index name
df_v1_clean.columns.name = None
df_v2_clean.columns.name = None


# =========================================================
# DIFFERENCE SCORE
# =========================================================

# Difference between allocation to investor and non-investor

df_v1_clean["diff"] = (
    df_v1_clean["inv"] -
    df_v1_clean["noninv"]
)

df_v2_clean["diff"] = (
    df_v2_clean["inv"] -
    df_v2_clean["noninv"]
)


# =========================================================
# ADD VERSION VARIABLE
# =========================================================

df_v1_clean["version"] = 1
df_v2_clean["version"] = 2

df_analysis = pd.concat(
    [
        df_v1_clean,
        df_v2_clean
    ],
    ignore_index=True
)


# =========================================================
# MERGE PARTICIPANT CHARACTERISTICS
# =========================================================

control_variables = [
    "political_orientation",
    "financial_risk_willingness",
    "success_determinants",
    "education",
    "age",
    "gender",
    "salary"
]

df_analysis = df_analysis.merge(
    df_balance[
        ["id", "version"] + control_variables
    ],
    on=["id", "version"],
    how="left"
)


# =========================================================
# CONVERT CONTROL VARIABLES
# =========================================================

# Likert variables
likert_variables = [
    "political_orientation",
    "financial_risk_willingness",
    "success_determinants"
]

for var in likert_variables:

    df_analysis[var] = pd.to_numeric(
        df_analysis[var],
        errors="coerce"
    )


# Gender / education / age / salary remain categorical
categorical_variables = [
    "education",
    "age",
    "gender",
    "salary"
]


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("Analysis")

show_data = st.sidebar.checkbox(
    "Show cleaned experimental data"
)

show_descriptives = st.sidebar.checkbox(
    "Show descriptive statistics"
)

run_regressions = st.sidebar.checkbox(
    "Run regression analysis",
    value=True
)


# =========================================================
# CLEAN DATA DISPLAY
# =========================================================

if show_data:

    st.header("Clean Experimental Data")

    st.subheader("Version 1")

    st.dataframe(
        df_v1_clean,
        use_container_width=True
    )

    st.subheader("Version 2")

    st.dataframe(
        df_v2_clean,
        use_container_width=True
    )


# =========================================================
# SAMPLE SIZES
# =========================================================

st.header("Sample")

n_total = df_analysis["id"].nunique()

n_v1 = df_analysis.loc[
    df_analysis["version"] == 1,
    "id"
].nunique()

n_v2 = df_analysis.loc[
    df_analysis["version"] == 2,
    "id"
].nunique()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total participants",
        n_total
    )

with col2:
    st.metric(
        "Version 1",
        n_v1
    )

with col3:
    st.metric(
        "Version 2",
        n_v2
    )


# =========================================================
# DESCRIPTIVE STATISTICS
# =========================================================

if show_descriptives:

    st.header("Descriptive Statistics")

    descriptive = (
        df_analysis
        .groupby(
            ["version", "amount"]
        )["diff"]
        .agg(
            [
                "count",
                "mean",
                "std",
                "median"
            ]
        )
        .reset_index()
    )

    descriptive["se"] = (
        descriptive["std"] /
        np.sqrt(descriptive["count"])
    )

    st.dataframe(
        descriptive,
        use_container_width=True
    )


# =========================================================
# REGRESSION DATA
# =========================================================

# Only 30 and 90 contain investor/non-investor
# observations and therefore enter the main regression.

df_reg = df_analysis[
    df_analysis["amount"].isin([30, 90])
].copy()


# Treatment indicator
# Version 1 = 0
# Version 2 = 1

df_reg["treatment"] = (
    df_reg["version"] == 2
).astype(int)


# Amount indicator
# 90 = 1
# 30 = 0

df_reg["amount90"] = (
    df_reg["amount"] == 90
).astype(int)


# =========================================================
# MAIN REGRESSION ANALYSIS
# =========================================================

if run_regressions:

    st.header("Regression Analysis")

    st.markdown(
        """
        The dependent variable is the difference between the amount
        allocated to the investor and the amount allocated to the
        non-investor. The main explanatory variable is the experimental
        treatment (Version 2). Standard errors are clustered at the
        participant level because participants provide multiple
        observations.
        """
    )


    # =====================================================
    # MODEL 1
    # SIMPLE TREATMENT EFFECT
    # =====================================================

    model_1 = smf.ols(
        formula="diff ~ treatment",
        data=df_reg
    ).fit(
        cov_type="cluster",
        cov_kwds={
            "groups": df_reg["id"]
        }
    )


    # =====================================================
    # MODEL 2
    # ADD AMOUNT
    # =====================================================

    model_2 = smf.ols(
        formula="diff ~ treatment + amount90",
        data=df_reg
    ).fit(
        cov_type="cluster",
        cov_kwds={
            "groups": df_reg["id"]
        }
    )


    # =====================================================
    # MODEL 3
    # TREATMENT × AMOUNT
    # =====================================================

    model_3 = smf.ols(
        formula="""
        diff ~ treatment
             + amount90
             + treatment:amount90
        """,
        data=df_reg
    ).fit(
        cov_type="cluster",
        cov_kwds={
            "groups": df_reg["id"]
        }
    )


    # =====================================================
    # MODEL 4
    # CONTROLS
    # =====================================================

    model_4 = smf.ols(
        formula="""
        diff ~ treatment
             + amount90
             + treatment:amount90
             + political_orientation
             + financial_risk_willingness
             + success_determinants
        """,
        data=df_reg
    ).fit(
        cov_type="cluster",
        cov_kwds={
            "groups": df_reg["id"]
        }
    )


    # =====================================================
    # MODEL 5
    # FULL CONTROLS
    # =====================================================

    model_5 = smf.ols(
        formula="""
        diff ~ treatment
             + amount90
             + treatment:amount90
             + political_orientation
             + financial_risk_willingness
             + success_determinants
             + C(education)
             + C(age)
             + C(gender)
             + C(salary)
        """,
        data=df_reg
    ).fit(
        cov_type="cluster",
        cov_kwds={
            "groups": df_reg["id"]
        }
    )


    # =====================================================
    # REGRESSION TABLE
    # =====================================================

    models = [
        model_1,
        model_2,
        model_3,
        model_4,
        model_5
    ]

    model_names = [
        "Model 1",
        "Model 2",
        "Model 3",
        "Model 4",
        "Model 5"
    ]


    # Variables we want to display
    display_variables = [
        "treatment",
        "amount90",
        "treatment:amount90",
        "political_orientation",
        "financial_risk_willingness",
        "success_determinants"
    ]


    def significance_stars(p):

        if p < 0.01:
            return "***"

        elif p < 0.05:
            return "**"

        elif p < 0.10:
            return "*"

        return ""


    regression_table = []


    for variable in display_variables:

        row = {
            "Variable": variable
        }

        for name, model in zip(
            model_names,
            models
        ):

            if variable in model.params.index:

                coef = model.params[variable]
                se = model.bse[variable]
                p = model.pvalues[variable]

                row[name] = (
                    f"{coef:.3f}"
                    f"{significance_stars(p)}"
                    f"\n({se:.3f})"
                )

            else:

                row[name] = ""


        regression_table.append(row)


    regression_table = pd.DataFrame(
        regression_table
    )


    # Add statistics
    statistics_rows = [

        {
            "Variable": "Observations",
            "Model 1": f"{int(model_1.nobs)}",
            "Model 2": f"{int(model_2.nobs)}",
            "Model 3": f"{int(model_3.nobs)}",
            "Model 4": f"{int(model_4.nobs)}",
            "Model 5": f"{int(model_5.nobs)}"
        },

        {
            "Variable": "R-squared",
            "Model 1": f"{model_1.rsquared:.3f}",
            "Model 2": f"{model_2.rsquared:.3f}",
            "Model 3": f"{model_3.rsquared:.3f}",
            "Model 4": f"{model_4.rsquared:.3f}",
            "Model 5": f"{model_5.rsquared:.3f}"
        },

        {
            "Variable": "Adjusted R-squared",
            "Model 1": f"{model_1.rsquared_adj:.3f}",
            "Model 2": f"{model_2.rsquared_adj:.3f}",
            "Model 3": f"{model_3.rsquared_adj:.3f}",
            "Model 4": f"{model_4.rsquared_adj:.3f}",
            "Model 5": f"{model_5.rsquared_adj:.3f}"
        }
    ]


    regression_table = pd.concat(
        [
            regression_table,
            pd.DataFrame(statistics_rows)
        ],
        ignore_index=True
    )


    st.subheader(
        "Regression Results"
    )

    st.dataframe(
        regression_table,
        use_container_width=True
    )


    st.caption(
        "Standard errors in parentheses. "
        "* p < 0.10, ** p < 0.05, *** p < 0.01. "
        "Standard errors are clustered at the participant level."
    )


    # =====================================================
    # FULL STATS
    # =====================================================

    with st.expander(
        "Show full regression output"
    ):

        st.subheader("Model 1")
        st.text(
            model_1.summary().as_text()
        )

        st.subheader("Model 2")
        st.text(
            model_2.summary().as_text()
        )

        st.subheader("Model 3")
        st.text(
            model_3.summary().as_text()
        )

        st.subheader("Model 4")
        st.text(
            model_4.summary().as_text()
        )

        st.subheader("Model 5")
        st.text(
            model_5.summary().as_text()
        )


# =========================================================
# SEPARATE ANALYSIS BY AMOUNT
# =========================================================

if run_regressions:

    st.header(
        "Treatment Effect by Amount"
    )

    amount_results = []

    for amount in [30, 90]:

        temp = df_reg[
            df_reg["amount"] == amount
        ].copy()

        model = smf.ols(
            formula="diff ~ treatment",
            data=temp
        ).fit(
            cov_type="HC3"
        )

        amount_results.append(
            {
                "Amount": amount,
                "N": int(model.nobs),
                "Version 1 Mean": temp.loc[
                    temp["treatment"] == 0,
                    "diff"
                ].mean(),
                "Version 2 Mean": temp.loc[
                    temp["treatment"] == 1,
                    "diff"
                ].mean(),
                "Difference": model.params.get(
                    "treatment",
                    np.nan
                ),
                "SE": model.bse.get(
                    "treatment",
                    np.nan
                ),
                "p-value": model.pvalues.get(
                    "treatment",
                    np.nan
                )
            }
        )


    amount_results_df = pd.DataFrame(
        amount_results
    )


    st.dataframe(
        amount_results_df,
        use_container_width=True
    )


# =========================================================
# 60 CHF: LUCK ANALYSIS
# =========================================================

st.header(
    "60 CHF Luck Analysis"
)

st.markdown(
    """
    The 60 CHF scenario contains lucky and unlucky conditions rather
    than investor and non-investor conditions. It is therefore analyzed
    separately from the main 30/90 CHF regression.
    """
)


# Create lucky-unlucky difference

df_v1_60 = df_v1_long[
    df_v1_long["amount"] == 60
].copy()

df_v2_60 = df_v2_long[
    df_v2_long["amount"] == 60
].copy()


df_60 = pd.concat(
    [
        df_v1_60,
        df_v2_60
    ],
    ignore_index=True
)


# Pivot
df_60 = (
    df_60
    .pivot_table(
        index=["id"],
        columns="condition",
        values="value",
        aggfunc="first"
    )
    .reset_index()
)


df_60.columns.name = None


# Add version
df_60 = df_60.merge(
    df_balance[
        ["id", "version"]
    ],
    on="id",
    how="left"
)


# Difference
if (
    "lucky" in df_60.columns
    and
    "unlucky" in df_60.columns
):

    df_60["luck_diff"] = (
        df_60["lucky"] -
        df_60["unlucky"]
    )


    df_60["treatment"] = (
        df_60["version"] == 2
    ).astype(int)


    # Regression
    luck_model = smf.ols(
        formula="luck_diff ~ treatment",
        data=df_60
    ).fit(
        cov_type="HC3"
    )


    luck_results = pd.DataFrame(
        {
            "Coefficient": [
                luck_model.params.get(
                    "treatment",
                    np.nan
                )
            ],
            "Standard Error": [
                luck_model.bse.get(
                    "treatment",
                    np.nan
                )
            ],
            "p-value": [
                luck_model.pvalues.get(
                    "treatment",
                    np.nan
                )
            ],
            "N": [
                int(luck_model.nobs)
            ]
        }
    )


    st.subheader(
        "Version Effect on Lucky − Unlucky Allocation"
    )

    st.dataframe(
        luck_results,
        use_container_width=True
    )


# =========================================================
# EXPORT REGRESSION TABLE
# =========================================================

if run_regressions:

    st.header(
        "Export Results"
    )

    csv_regression = regression_table.to_csv(
        index=False
    )

    st.download_button(
        label="Download regression table as CSV",
        data=csv_regression,
        file_name="regression_results.csv",
        mime="text/csv"
    )


# =========================================================
# POWER ANALYSIS
# =========================================================

st.header(
    "Power Analysis"
)

power_analysis = TTestIndPower()


for amount in [30, 90]:

    st.subheader(
        f"Amount {amount}"
    )

    temp = df_analysis[
        df_analysis["amount"] == amount
    ].copy()

    v1 = temp[
        temp["version"] == 1
    ]["inv"].dropna()

    v2 = temp[
        temp["version"] == 2
    ]["inv"].dropna()


    if len(v1) == 0 or len(v2) == 0:

        st.write(
            "Insufficient data for power analysis."
        )

        continue


    mean_1 = v1.mean()
    mean_2 = v2.mean()

    sd_1 = v1.std(
        ddof=1
    )

    sd_2 = v2.std(
        ddof=1
    )

    n1 = len(v1)
    n2 = len(v2)


    pooled_sd = np.sqrt(
        (
            (n1 - 1) * sd_1**2
            +
            (n2 - 1) * sd_2**2
        )
        /
        (
            n1 + n2 - 2
        )
    )


    mean_difference = (
        mean_1 - mean_2
    )


    if pooled_sd > 0:

        effect_size = (
            mean_difference /
            pooled_sd
        )


        required_n = (
            power_analysis.solve_power(
                effect_size=abs(effect_size),
                alpha=0.05,
                power=0.80,
                ratio=1.0,
                alternative="two-sided"
            )
        )


        st.write(
            f"Version 1 mean: {mean_1:.3f}"
        )

        st.write(
            f"Version 2 mean: {mean_2:.3f}"
        )

        st.write(
            f"Mean difference: "
            f"{mean_difference:.3f}"
        )

        st.write(
            f"Pooled SD: "
            f"{pooled_sd:.3f}"
        )

        st.write(
            f"Cohen's d: "
            f"{effect_size:.3f}"
        )

        st.write(
            f"Required sample size per group "
            f"for 80% power: "
            f"{np.ceil(required_n):.0f}"
        )