import pandas as pd
import streamlit as st
import os
import requests
from PIL import Image
from io import BytesIO

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
# IMAGE DICTIONARY (example - replace with your image mapping)
# =============================
image_dict = {
    # Example: 'NYY': 'https://www.mlbstatic.com/team-logos/147.svg'
    # Add your actual abbreviation:logo_url mappings here
}

def fetch_team_abbreviation(player_id):
    url = f"https://statsapi.mlb.com/api/v1/people?personIds={player_id}&hydrate=currentTeam"
    data = requests.get(url).json()
    url_team = 'https://statsapi.mlb.com/' + data['people'][0]['currentTeam']['link']
    data_team = requests.get(url_team).json()
    return data_team['teams'][0]['abbreviation']

def get_team_abbreviation_for_df(df):
    if 'MLB_ID' not in df.columns:
        return [''] * len(df)
    abbs = []
    for pid in df['MLB_ID']:
        try:
            abbs.append(fetch_team_abbreviation(pid))
        except Exception:
            abbs.append('')
    return abbs

# =============================
# VALIDATE REQUIRED COLUMNS
# =============================
core_cols = ["Name", "Age", "Swing+", "PowerIndex+", "ProjSwing+"]
extra_cols = ["avg_bat_speed", "swing_length", "attack_angle", "swing_tilt"]
if "MLB_ID" in df.columns:
    core_cols = ["MLB_ID"] + core_cols
required_cols = core_cols + [c for c in extra_cols if c in df.columns]

missing = [c for c in core_cols if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# =============================
# Add Team Column
# =============================
if "MLB_ID" in df.columns:
    if "Team" not in df.columns:
        with st.spinner("Fetching team abbreviations..."):
            df["Team"] = get_team_abbreviation_for_df(df)
else:
    df["Team"] = ""

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
main_cmap = "RdYlBu_r"
elite_cmap = "Reds"

# =============================
# PLAYER METRICS TABLE
# =============================
st.subheader("📊 Player Metrics Table")

display_cols = [c for c in ["Name", "Team", "Age", "Swing+", "ProjSwing+", "PowerIndex+"] + extra_cols if c in df_filtered.columns]

rename_map = {
    "Team": "Team",
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
    leaderboard_cols = [c for c in ["Name", "Team", "Age", "Swing+", "ProjSwing+", "PowerIndex+"] if c in top_swing.columns]
    st.dataframe(
        top_swing[leaderboard_cols]
        .style.background_gradient(subset=["Swing+"], cmap=elite_cmap)
        .format(precision=1),
        use_container_width=True,
        hide_index=True
    )

with col2:
    st.markdown("**Top 10 by ProjSwing+**")
    top_proj = df_filtered.sort_values("ProjSwing+", ascending=False).head(10).reset_index(drop=True)
    leaderboard_cols = [c for c in ["Name", "Team", "Age", "ProjSwing+", "Swing+", "PowerIndex+"] if c in top_proj.columns]
    st.dataframe(
        top_proj[leaderboard_cols]
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

total_players = len(df)
df["Swing+_rank"] = df["Swing+"].rank(ascending=False, method="min").astype(int)
df["ProjSwing+_rank"] = df["ProjSwing+"].rank(ascending=False, method="min").astype(int)
df["PowerIndex+_rank"] = df["PowerIndex+"].rank(ascending=False, method="min").astype(int)

p_swing_rank = df.loc[df["Name"] == player_select, "Swing+_rank"].iloc[0]
p_proj_rank = df.loc[df["Name"] == player_select, "ProjSwing+_rank"].iloc[0]
p_power_rank = df.loc[df["Name"] == player_select, "PowerIndex+_rank"].iloc[0]

st.markdown(
    f"""
    <div style="display:flex; justify-content:space-around; margin-top:10px;">
        <div style="text-align:center;">
            <h3 style="margin-bottom:0;">Swing+</h3>
            <h2 style="margin:0;">{round(player_data['Swing+'], 1)}</h2>
            <p style="color:gray; font-size:12px; margin-top:0;">Rank: {p_swing_rank} / {total_players}</p>
        </div>
        <div style="text-align:center;">
            <h3 style="margin-bottom:0;">ProjSwing+</h3>
            <h2 style="margin:0;">{round(player_data['ProjSwing+'], 1)}</h2>
            <p style="color:gray; font-size:12px; margin-top:0;">Rank: {p_proj_rank} / {total_players}</p>
        </div>
        <div style="text-align:center;">
            <h3 style="margin-bottom:0;">PowerIndex+</h3>
            <h2 style="margin:0;">{round(player_data['PowerIndex+'], 1)}</h2>
            <p style="color:gray; font-size:12px; margin-top:0;">Rank: {p_power_rank} / {total_players}</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# =============================
# Optional Swing Mechanics Section
# =============================
if set(extra_cols).issubset(df.columns):
    st.markdown("**Swing Mechanics**")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Bat Speed", f"{round(player_data['avg_bat_speed'], 1)} mph")
    col2.metric("Swing Length", round(player_data["swing_length"], 2))
    col3.metric("Attack Angle", round(player_data["attack_angle"], 1))
    col4.metric("Swing Tilt", round(player_data["swing_tilt"], 1))
