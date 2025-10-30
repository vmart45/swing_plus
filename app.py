import pandas as pd
import streamlit as st
import plotly.express as px
import os

# =============================
# PAGE SETUP
# =============================
st.set_page_config(
    page_title="Swing+ & ProjSwing+ Dashboard",
    page_icon="⚾",
    layout="wide"
)

st.title("⚾ Swing+ & ProjSwing+ Dashboard")
st.markdown("""
Explore **Swing+**, **ProjSwing+**, **PowerIndex+**, and **GapPotential**  
""")

# =============================
# LOAD DATA (LOCAL FILE)
# =============================
DATA_PATH = "ProjSwingPlus_Output.csv"

if not os.path.exists(DATA_PATH):
    st.error(f"❌ Could not find `{DATA_PATH}` in the app directory.")
    st.stop()

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

df = load_data(DATA_PATH)

# =============================
# VALIDATE DATA
# =============================
required_cols = ["Name", "Age", "Swing+", "PowerIndex+", "ProjSwing+", "GapPotential"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# =============================
# SIDEBAR FILTERS
# =============================
st.sidebar.header("Filters")

min_age, max_age = int(df["Age"].min()), int(df["Age"].max())
age_range = st.sidebar.slider("Age Range", min_age, max_age, (min_age, 25))

df_filtered = df[(df["Age"] >= age_range[0]) & (df["Age"] <= age_range[1])]

search_name = st.sidebar.text_input("Search Player by Name")
if search_name:
    df_filtered = df_filtered[df_filtered["Name"].str.contains(search_name, case=False, na=False)]

# =============================
# PLAYER TABLE
# =============================
st.subheader("📊 Player Metrics Table")

styled_df = (
    df_filtered[["Name", "Age", "Swing+", "PowerIndex+", "ProjSwing+", "GapPotential"]]
    .sort_values("ProjSwing+", ascending=False)
    .style.background_gradient(subset=["ProjSwing+"], cmap="YlOrBr")
    .format(precision=1)
)

st.dataframe(styled_df, use_container_width=True)


# =============================
# LEADERBOARDS
# =============================
st.subheader("🏆 Top 10 Leaderboards")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Top 10 by ProjSwing+**")
    top_proj = df_filtered.sort_values("ProjSwing+", ascending=False).head(10)
    st.dataframe(
        top_proj[["Name", "Age", "ProjSwing+", "Swing+", "GapPotential"]]
        .style.background_gradient(subset=["ProjSwing+"], cmap="YlOrBr")
        .format(precision=1),
        use_container_width=True
    )

with col2:
    st.markdown("**Top 10 by PowerIndex+**")
    top_power = df_filtered.sort_values("PowerIndex+", ascending=False).head(10)
    st.dataframe(
        top_power[["Name", "Age", "PowerIndex+", "Swing+", "ProjSwing+"]]
        .style.background_gradient(subset=["PowerIndex+"], cmap="YlOrBr")
        .format(precision=1),
        use_container_width=True
    )

# =============================
# PLAYER DETAIL
# =============================
st.subheader("🔍 Player Detail View")

player_select = st.selectbox("Select a Player", sorted(df_filtered["Name"].unique()))
player_data = df[df["Name"] == player_select].iloc[0]

colA, colB, colC, colD = st.columns(4)
colA.metric("Swing+", round(player_data["Swing+"], 1))
colB.metric("PowerIndex+", round(player_data["PowerIndex+"], 1))
colC.metric("ProjSwing+", round(player_data["ProjSwing+"], 1))
colD.metric("GapPotential", round(player_data["GapPotential"], 1))
