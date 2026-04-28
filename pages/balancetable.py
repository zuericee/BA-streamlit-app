import pandas as pd
import streamlit as st

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

st.header("Statistical Tests")

from scipy import stats
import pandas as pd

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

st.dataframe(results_df)

# cohen's d

import numpy as np

def cohens_d(x1, x2):
    x1 = pd.to_numeric(x1, errors='coerce')
    x2 = pd.to_numeric(x2, errors='coerce')

    x1 = x1.dropna()
    x2 = x2.dropna()

    m1, m2 = x1.mean(), x2.mean()
    s1, s2 = x1.std(), x2.std()

    n1, n2 = len(x1), len(x2)
    s_pooled = (((n1 - 1)*s1**2 + (n2 - 1)*s2**2) / (n1 + n2 - 2))**0.5

    return (m1 - m2) / s_pooled

d_results = []

for var in [
    "financial_risk_willingness",
    "success_determinants"
]:
    d = cohens_d(
        df_v1[var],
        df_v2[var]
    )
    
    d_results.append({
        "variable": var,
        "cohens_d": d
    })

d_df = pd.DataFrame(d_results)

st.subheader("Standardized Differences (Cohen's d)")
st.dataframe(d_df)

