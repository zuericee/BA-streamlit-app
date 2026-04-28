import pandas as pd
import numpy as np
import streamlit as st
from statsmodels.stats.power import TTestIndPower

st.title("Powertest")

# Load data
cols_to_keep = [5, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20]
df = pd.read_csv("results-survey642194.csv", usecols=cols_to_keep)

# Rename columns (FIXED comma!)
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

# Split data
df_v1 = df[df["version"] == 1]
df_v2 = df[df["version"] == 2]

# Columns to analyze
cols = df.columns.drop("version")

# Summary statistics
summary = pd.DataFrame({
    "mean_v1": df_v1[cols].mean(),
    "std_v1": df_v1[cols].std(),
    "n_v1": df_v1[cols].count(),
    "mean_v2": df_v2[cols].mean(),
    "std_v2": df_v2[cols].std(),
    "n_v2": df_v2[cols].count()
})

# Cohen's d
def cohens_d(m1, m2, s1, s2, n1, n2):
    pooled_sd = np.sqrt(((n1 - 1)*s1**2 + (n2 - 1)*s2**2) / (n1 + n2 - 2))
    return (m1 - m2) / pooled_sd

summary["cohens_d"] = cohens_d(
    summary["mean_v1"],
    summary["mean_v2"],
    summary["std_v1"],
    summary["std_v2"],
    summary["n_v1"],
    summary["n_v2"]
)

# Power analysis
analysis = TTestIndPower()

summary["power"] = summary.apply(
    lambda row: analysis.power(
        effect_size=abs(row["cohens_d"]),
        nobs1=row["n_v1"],
        ratio=row["n_v2"] / row["n_v1"],
        alpha=0.05
    ),
    axis=1
)

# Required sample size per group (for 80% power)
summary["required_n_per_group"] = summary["cohens_d"].apply(
    lambda d: analysis.solve_power(
        effect_size=abs(d),
        power=0.8,
        alpha=0.05
    )
)

# Output
st.write(summary)