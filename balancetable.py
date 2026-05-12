import pandas as pd
import streamlit as st
from scipy import stats
import numpy as np
from statsmodels.stats.power import TTestIndPower

st.title("Balance Table")

# Load data
cols_to_keep = [5, 20, 21, 22, 23, 24, 25, 26]
df = pd.read_csv("results-survey642194.csv", usecols=cols_to_keep)

# Rename columns
df.columns = [
    "version",
    "political_orientation",
    "financial_risk_willingness",
    "success_determinants",
    "education",
    "age",
    "gender",
    "salary"
]

# Show raw data (optional)
if st.checkbox("Show raw data"):
    st.write(df)

# FIX version
df["version"] = df["version"].astype(int)

# Split
df_v1 = df[df["version"] == 1]
df_v2 = df[df["version"] == 2]

st.write("N Version 1:", len(df_v1))
st.write("N Version 2:", len(df_v2))

# -----------------------
# FUNCTION
# -----------------------
def make_counts(df, var, categories):
    counts = (
        df[var]
        .value_counts(dropna=False)
        .reindex(categories, fill_value=0)
    )

    return pd.DataFrame({
        var: categories,
        "count": counts.values
    })

# -----------------------
# VARIABLES
# -----------------------
variables = [
    "political_orientation",
    "financial_risk_willingness",
    "success_determinants",
    "education",
    "age",
    "salary"
]

# -----------------------
# CATEGORIES
# -----------------------
categories_dict = {
    "political_orientation": [
        "1 (very left wing)",
        "2 (left wing)",
        "3 (slightly left wing)",
        "4 (moderate)",
        "5 (slightly right wing)",
        "6 (right wing)"
    ],

    "financial_risk_willingness": [
        "1 (not at all willing to take financial risks)",
        "2",
        "3",
        "4",
        "5 (neutral)",
        "6",
        "7",
        "8",
        "9",
        "10 (very willing to take financial risks)"
    ],

    "success_determinants": [
        "1 (success mainly depends on effort)",
        "2",
        "3",
        "4",
        "5 (success depends equally on effort and luck)",
        "6",
        "7",
        "8",
        "9",
        "10 (success mainly depends on luck)"
    ],

    "education": [
        "Other qualification",
        "Preparatory High School (Gymnasium)",
        "Professional education (apprenticeship / Lehre)",
        "Bachelor's degree (university and university of applied sciences)",
        "Master's degree (university and university of applied sciences)",
        "Doctoral degree / PhD",
        "Compulsory schooling",
        "Prefer not to say"
    ],

    "age": [
        "Below 20",
        "20-29",
        "30-39",
        "40-49",
        "50-59",
        "60-69",
        "70 and above",
        "Prefer not to say"
    ],

    "salary": [
        "Below CHF 2'000",
        "CHF 2'000 - 3'999",
        "CHF 4'000 – 5'999",
        "CHF 6'000 – 7'999",
        "CHF 8'000 – 9'999",
        "CHF 10'000 or more",
        "Prefer not to say"
    ]
}

# -----------------------
# DISPLAY
# -----------------------
col1, col2 = st.columns(2)

if checkbox := st.checkbox("Show balance tables"):
    with col1:
        st.header("Version 1")
        for var in variables:
            st.subheader(var)
            st.dataframe(make_counts(df_v1, var, categories_dict[var]))
    with col2:
        st.header("Version 2")
        for var in variables:
            st.subheader(var)
            st.dataframe(make_counts(df_v2, var, categories_dict[var]))


# ---------------------------------

results = []

def run_test(df, var):
    g1 = df[df["version"] == 1][var].dropna()
    g2 = df[df["version"] == 2][var].dropna()

    # numeric / Likert variables → t-test
    if pd.api.types.is_numeric_dtype(df[var]):
        t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)

        results.append({
            "variable": var,
            "test": "t-test",
            "mean_v1": g1.mean(),
            "mean_v2": g2.mean(),
            "p_value": p_val
        })

    # categorical variables → chi-square
    else:
        table = pd.crosstab(df["version"], df[var])
        chi2, p_val, dof, expected = stats.chi2_contingency(table)

        results.append({
            "variable": var,
            "test": "chi-square",
            "p_value": p_val
        })

variables = [
    "political_orientation",
    "financial_risk_willingness",
    "success_determinants",
    "education",
    "age",
    "salary"
]

for var in variables:
    run_test(df, var)

results_df = pd.DataFrame(results)

if st.checkbox("Show stats tests results"):
    st.write(results_df)

st.title("Clean Survey Analysis")

# -----------------------
# LOAD DATA
# -----------------------
cols_to_keep = [5, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19]
df = pd.read_csv("results-survey642194.csv", usecols=cols_to_keep)

df.columns = [
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

df["id"] = df.index

# -----------------------
# SPLIT VERSIONS
# -----------------------
v1_cols = ["id"] + [c for c in df.columns if c.startswith("v1_")]
v2_cols = ["id"] + [c for c in df.columns if c.startswith("v2_")]

df_v1 = df[v1_cols]
df_v2 = df[v2_cols]

# -----------------------
# MELT TO LONG FORMAT
# -----------------------
df_v1_long = df_v1.melt(id_vars="id", var_name="scenario", value_name="value")
df_v2_long = df_v2.melt(id_vars="id", var_name="scenario", value_name="value")

# -----------------------
# EXTRACT INFO
# -----------------------
for d in [df_v1_long, df_v2_long]:
    d[["v", "amount", "condition"]] = d["scenario"].str.split("_", expand=True)
    d["amount"] = pd.to_numeric(d["amount"], errors="coerce")
    d["value"] = pd.to_numeric(d["value"], errors="coerce")

# -----------------------
# CLEAN
# -----------------------
df_v1_long = df_v1_long.dropna(subset=["value"])
df_v2_long = df_v2_long.dropna(subset=["value"])

# -----------------------
# PIVOT: one row per person × amount
# -----------------------
df_v1_clean = df_v1_long.pivot_table(
    index=["id", "amount"],
    columns="condition",
    values="value",
    aggfunc="first"
).reset_index()

df_v2_clean = df_v2_long.pivot_table(
    index=["id", "amount"],
    columns="condition",
    values="value",
    aggfunc="first"
).reset_index()

# -----------------------
# DIFFERENCE SCORE
# -----------------------
df_v1_clean["diff"] = df_v1_clean["inv"] - df_v1_clean["noninv"]
df_v2_clean["diff"] = df_v2_clean["inv"] - df_v2_clean["noninv"]

# -----------------------
# DISPLAY CLEAN DATA
# -----------------------
st.subheader("Version 1 Data")
st.dataframe(df_v1_clean)

st.subheader("Version 2 Data")
st.dataframe(df_v2_clean)

# =========================================================
# 30 & 90 = INFERENTIAL ANALYSIS
# =========================================================

# =========================================================
# 30 & 90 = INFERENTIAL ANALYSIS (UNIFIED)
# only investor data, different SD scenarios
# =========================================================

power_analysis = TTestIndPower()

for amount in [30, 90]:

    st.subheader(f"Amount {amount} Analysis")

    # Filter data
    df_v1 = df_v1_clean[df_v1_clean["amount"] == amount].drop(
        columns=["lucky", "unlucky"], errors="ignore"
    )
    df_v2 = df_v2_clean[df_v2_clean["amount"] == amount].drop(
        columns=["lucky", "unlucky"], errors="ignore"
    )

    # Means
    mean_inv_1 = df_v1["inv"].mean()
    mean_inv_2 = df_v2["inv"].mean()

    st.write(f"Investor means for amount {amount}:")
    st.write(f"Version 1: {mean_inv_1:.2f}")
    st.write(f"Version 2: {mean_inv_2:.2f}")

    # Standard deviations
    std_inv_1 = df_v1["inv"].std(ddof=1)
    std_inv_2 = df_v2["inv"].std(ddof=1)

    st.write(f"Investor SDs for amount {amount}:")
    st.write(f"Version 1: {std_inv_1:.2f}")
    st.write(f"Version 2: {std_inv_2:.2f}")

    # Sample sizes
    n1 = df_v1["inv"].dropna().shape[0]
    n2 = df_v2["inv"].dropna().shape[0]

    # Pooled SD
    sd_base = np.sqrt(
        (
            (n1 - 1) * std_inv_1**2 +
            (n2 - 1) * std_inv_2**2
        ) / max(n1 + n2 - 2, 1)
    )

    # Different SD scenarios
    sd_low = 0.8 * sd_base
    sd_high = 1.2 * sd_base

    st.write(f"Pooled SD (base): {sd_base:.2f}")

    mean_diff = mean_inv_1 - mean_inv_2

    st.write(f"Mean difference: {mean_diff:.2f}")

    st.write(f"Power analysis for amount {amount}:")

    for label, sd in [
        ("Low SD (80%)", sd_low),
        ("Base SD", sd_base),
        ("High SD (120%)", sd_high)
    ]:

        if sd == 0:
            st.write(f"{label}: cannot compute (SD = 0)")
            continue

        effect_size = mean_diff / sd

        required_n = power_analysis.solve_power(
            effect_size=abs(effect_size),
            alpha=0.05,
            power=0.80,
            ratio=1.0,
            alternative='two-sided'
        )

        st.write(
            f"{label} → SD = {sd:.2f}, "
            f"Cohen's d = {effect_size:.2f}, "
            f"required n per group = {np.ceil(required_n):.0f}"
        )
