import pandas as pd
import streamlit as st
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
Explore **Swing+**, **ProjSwing+**, and **PowerIndex+** —  
a modern approach to evaluating swing efficiency, scalability, and mechanical power.
""")

# =============================
# LOAD DATA
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
# VALIDATE REQUIRED COLUMNS
# =============================
core_cols = ["Name", "Age", "Swing+", "PowerIndex+", "ProjSwing+"]
extra_cols = ["avg_bat_speed", "swing_length", "attack_angle", "swing_tilt"]
required_cols = core_cols + [c for c in extra_cols if c in df.columns]

missing = [c for c in core_cols if c not in df.columns]
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
# COLOR SCHEMES
# =============================
main_cmap = "RdYlBu_r"   # red–white–blue for main metrics
elite_cmap = "Reds"      # solid red gradient for top performers

# =============================
# PLAYER METRICS TABLE
# =============================
st.subheader("📊 Player Metrics Table")

display_cols = [c for c in ["Name", "Age", "Swing+", "ProjSwing+", "PowerIndex+"] + extra_cols if c in df_filtered.columns]

rename_map = {
    "Swing+": "Swing+",
    "ProjSwing+": "ProjSwing+",
    "PowerIndex+": "PowerIndex+",
    "avg_bat_speed": "Avg Bat Speed (mph)",
    "swing_length": "Swing Length (m)",
    "attack_angle": "Attack Angle (°)",
    "swing_tilt": "Swing Tilt (°)"
}

styled_df = (
    df_filtered[display_cols]
    .rename(columns=rename_map)
    .sort_values("Swing+", ascending=False)
    .reset_index(drop=True)
    .style.background_gradient(
        subset=["Swing+", "ProjSwing+", "PowerIndex+"], cmap=main_cmap
    )
    .format(precision=1)
)

st.dataframe(styled_df, use_container_width=True, hide_index=True)

# =============================
# LEADERBOARDS
# =============================
st.subheader("🏆 Top 10 Leaderboards")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Top 10 by Swing+**")
    top_swing = df_filtered.sort_values("Swing+", ascending=False).head(10).reset_index(drop=True)
    st.dataframe(
        top_swing[["Name", "Age", "Swing+", "ProjSwing+", "PowerIndex+"]]
        .style.background_gradient(subset=["Swing+"], cmap=elite_cmap)
        .format(precision=1),
        use_container_width=True,
        hide_index=True
    )

with col2:
    st.markdown("**Top 10 by ProjSwing+**")
    top_proj = df_filtered.sort_values("ProjSwing+", ascending=False).head(10).reset_index(drop=True)
    st.dataframe(
        top_proj[["Name", "Age", "ProjSwing+", "Swing+", "PowerIndex+"]]
        .style.background_gradient(subset=["ProjSwing+"], cmap=elite_cmap)
        .format(precision=1),
        use_container_width=True,
        hide_index=True
    )

# =============================
# PLAYER DETAIL VIEW
# =============================
st.subheader("🔍 Player Detail View")

player_select = st.selectbox("Select a Player", sorted(df_filtered["Name"].unique()))
player_data = df[df["Name"] == player_select].iloc[0]

# Calculate ranks (1 = best)
total_players = len(df)
df["Swing+_rank"] = df["Swing+"].rank(ascending=False, method="min").astype(int)
df["ProjSwing+_rank"] = df["ProjSwing+"].rank(ascending=False, method="min").astype(int)
df["PowerIndex+_rank"] = df["PowerIndex+"].rank(ascending=False, method="min").astype(int)

p_swing_rank = df.loc[df["Name"] == player_select, "Swing+_rank"].iloc[0]
p_proj_rank = df.loc[df["Name"] == player_select, "ProjSwing+_rank"].iloc[0]
p_power_rank = df.loc[df["Name"] == player_select, "PowerIndex+_rank"].iloc[0]

colA, colB, colC = st.columns(3)
colA.metric("Swing+", f"{round(player_data['Swing+'], 1)} ({p_swing_rank} / {total_players})")
colB.metric("ProjSwing+", f"{round(player_data['ProjSwing+'], 1)} ({p_proj_rank} / {total_players})")
colC.metric("PowerIndex+", f"{round(player_data['PowerIndex+'], 1)} ({p_power_rank} / {total_players})")

# Optional: Mechanical context
if set(extra_cols).issubset(df.columns):
    st.markdown("**Swing Mechanics**")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Bat Speed", f"{round(player_data['avg_bat_speed'], 1)} mph")
    col2.metric("Swing Length", round(player_data["swing_length"], 2))
    col3.metric("Attack Angle", round(player_data["attack_angle"], 1))
    col4.metric("Swing Tilt", round(player_data["swing_tilt"], 1))
