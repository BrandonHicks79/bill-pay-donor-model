"""
app.py — Streamlit dashboard for the bill-pay donor model.

SELF-CONTAINED VERSION: the sample data lives in this file and the
regression model trains when the app loads (31 rows — it takes under a
second). No CSV file, no outputs folder, nothing else needed in the repo
besides this file and requirements.txt. That makes phone-based deploys
simple: upload app.py + requirements.txt to GitHub, point Streamlit at
app.py, done.

WHAT IT SHOWS:
    1. A ranked table of billionaires by donor-propensity score, with
       filters (net worth, industry, country, self-made, minimum score).
    2. A bill matcher: enter a bill amount and see the top-ranked donors,
       plus how many bills of that size each could cover at 0.01% of their
       net worth (an illustrative yardstick, NOT a promised gift).

WHAT IT DOES NOT DO:
    - It does not predict that any specific person WILL donate. The score
      is a ranking aid built from public data (see README.md).
    - It shows no contact info, no emails, no phone numbers, no addresses.
      Outreach happens through proper channels (advisors, nonprofits),
      not from this screen.
"""

import math

import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Sample data: 31 billionaires, approximate PUBLIC figures (net worth in $B).
# Same data as data/sample_billionaires.csv in the project.
DATA = [
    ("Elon Musk", 400, 54, "United States", "Technology", 1, 0),
    ("Jeff Bezos", 240, 61, "United States", "Technology", 1, 0),
    ("Mark Zuckerberg", 230, 41, "United States", "Technology", 1, 1),
    ("Larry Ellison", 200, 81, "United States", "Technology", 1, 1),
    ("Bernard Arnault", 180, 76, "France", "Fashion", 0, 0),
    ("Warren Buffett", 150, 95, "United States", "Finance", 1, 1),
    ("Larry Page", 130, 52, "United States", "Technology", 1, 0),
    ("Sergey Brin", 125, 52, "United States", "Technology", 1, 0),
    ("Bill Gates", 120, 70, "United States", "Technology", 1, 1),
    ("Steve Ballmer", 120, 69, "United States", "Technology", 1, 1),
    ("Amancio Ortega", 120, 89, "Spain", "Fashion", 1, 0),
    ("Jensen Huang", 120, 62, "United States", "Technology", 1, 0),
    ("Michael Dell", 110, 60, "United States", "Technology", 1, 0),
    ("Michael Bloomberg", 105, 83, "United States", "Finance", 1, 1),
    ("Rob Walton", 100, 81, "United States", "Retail", 0, 0),
    ("Jim Walton", 95, 77, "United States", "Retail", 0, 0),
    ("Mukesh Ambani", 95, 68, "India", "Energy", 0, 0),
    ("Alice Walton", 90, 76, "United States", "Retail", 0, 0),
    ("Gautam Adani", 90, 63, "India", "Energy", 1, 0),
    ("Carlos Slim", 85, 85, "Mexico", "Telecom", 1, 0),
    ("Francoise Bettencourt Meyers", 80, 72, "France", "Fashion", 0, 0),
    ("Julia Koch", 70, 63, "United States", "Energy", 0, 0),
    ("Charles Koch", 65, 89, "United States", "Energy", 0, 0),
    ("Pony Ma", 40, 54, "China", "Technology", 1, 0),
    ("Colin Zheng Huang", 45, 45, "China", "Technology", 1, 0),
    ("MacKenzie Scott", 35, 55, "United States", "Technology", 1, 1),
    ("Abigail Johnson", 35, 63, "United States", "Finance", 0, 0),
    ("Melinda French Gates", 30, 61, "United States", "Technology", 0, 1),
    ("Jack Ma", 25, 61, "China", "Technology", 1, 0),
    ("Dustin Moskovitz", 20, 41, "United States", "Technology", 1, 1),
    ("Ray Dalio", 15, 76, "United States", "Finance", 1, 1),
]
COLS = ["name", "net_worth_billions", "age", "country", "industry",
        "self_made", "giving_pledge"]
INDUSTRIES = ["Technology", "Finance", "Retail", "Fashion", "Energy", "Telecom"]


@st.cache_resource
def load_ranking():
    """Build features, train the logistic regression, return the ranking."""
    df = pd.DataFrame(DATA, columns=COLS)
    df["log_net_worth"] = df["net_worth_billions"].apply(
        lambda v: math.log10(max(v, 0.1)))
    df["us"] = (df["country"] == "United States").astype(int)
    for ind in INDUSTRIES:
        df[f"industry_{ind}"] = (df["industry"] == ind).astype(int)
    known = df[[f"industry_{i}" for i in INDUSTRIES]].sum(axis=1)
    df["industry_Other"] = (known == 0).astype(int)
    feature_cols = (["log_net_worth", "age", "self_made", "us"]
                    + [f"industry_{i}" for i in INDUSTRIES]
                    + ["industry_Other"])
    X = df[feature_cols].astype(float)
    y = df["giving_pledge"].astype(int)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=1000, C=1.0)),
    ])
    pipeline.fit(X, y)
    df["propensity_score"] = (pipeline.predict_proba(X)[:, 1] * 100).round(1)
    ranked = df.sort_values("propensity_score", ascending=False).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1
    return ranked


st.set_page_config(page_title="Billionaire Donor Prospects", layout="wide")
st.title("Billionaire Donor Prospects — Bill-Pay App")

st.warning(
    "Propensity scores are a RANKING AID built from public data "
    "(Giving Pledge membership, net worth, age, industry). They do NOT "
    "predict whether a specific person will donate. Outreach still needs "
    "a human and a warm introduction — ideally through a nonprofit partner "
    "or philanthropy advisor."
)

df = load_ranking()

# ---- Sidebar filters -----------------------------------------------------
st.sidebar.header("Filters")
min_worth = st.sidebar.slider(
    "Minimum net worth (billions $)", 0, int(df["net_worth_billions"].max()),
    0, step=5,
)
industries = sorted(df["industry"].unique())
chosen_industries = st.sidebar.multiselect(
    "Industry", industries, default=industries,
)
countries = sorted(df["country"].unique())
chosen_countries = st.sidebar.multiselect(
    "Country", countries, default=countries,
)
self_made_only = st.sidebar.checkbox("Self-made only", value=False)
min_score = st.sidebar.slider("Minimum propensity score", 0, 100, 0, step=5)

filtered = df[
    (df["net_worth_billions"] >= min_worth)
    & (df["industry"].isin(chosen_industries))
    & (df["country"].isin(chosen_countries))
    & (df["propensity_score"] >= min_score)
]
if self_made_only:
    filtered = filtered[filtered["self_made"] == 1]

st.subheader(f"Ranked donors ({len(filtered)} shown)")
st.dataframe(filtered, use_container_width=True, hide_index=True)
st.download_button(
    "Download filtered ranking (CSV)",
    filtered.to_csv(index=False),
    file_name="ranked_donors_filtered.csv",
    mime="text/csv",
)

# ---- Bill matcher --------------------------------------------------------
st.subheader("Bill matcher")
bill_amount = st.number_input(
    "Bill amount ($)", min_value=10, max_value=1_000_000, value=5000, step=100,
)
st.caption(
    "Capacity yardstick: bills of this size covered at 0.01% of net worth. "
    "This is an illustration of scale, not a promised or expected gift."
)
matched = filtered.copy()
matched["bills_at_0_01pct"] = (
    matched["net_worth_billions"] * 1_000_000_000 * 0.0001 / bill_amount
).astype(int)
matched = matched.sort_values("propensity_score", ascending=False).head(10)
st.dataframe(
    matched[["rank", "name", "net_worth_billions", "propensity_score",
             "giving_pledge", "bills_at_0_01pct"]],
    use_container_width=True, hide_index=True,
)
