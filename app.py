import pandas as pd
import streamlit as st
import requests
from PIL import Image
from io import BytesIO
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
# MLB TEAM LOGOS
# =============================
mlb_teams = [
    {"team": "AZ", "logo_url": "https://a.espncdn.com/i/teamlogos/mlb/500/scoreboard/ari.png"},
    {"team": "ATL", "logo_url": "https://a.espncdn.com/i/teamlogos/mlb/500/scoreboard/atl.png"},
    {"team": "BAL", "logo_url": "https://a.espncdn.com/i/teamlogos/mlb/500/scoreboard/bal.png"},
    {"team": "BOS", "logo_url": "https://a.espncdn.com/i/teamlogos/mlb/500/scoreboard/bos.png"},
    {"team": "CHC", "logo_url": "https://a.espncdn.com/i/teamlogos/mlb/500/scoreboard/chc.png"},
    {"team": "CWS", "logo_url": "https://a.espncdn.com/i/teamlogos/mlb/500/scoreboard/chw.png"},
    {"team": "CIN", "logo_url": "https://a.espncdn.com/i/teamlogos/mlb/500/scoreboard/cin.png"},
    {"team": "CLE", "logo_url": "https://a.espncdn.com/i/teamlogos/mlb/500/scoreboard/cle.png"},
    {"team": "COL", "logo_url": "https://a.espncdn.com/i/teamlogos/mlb/500/scoreboard/col.png"},
    {"team": "DET", "logo_url": "https://a.espncdn.com/i/teamlogos/mlb/500/scoreboard/det.png"},
    {"team": "HOU", "logo_url": "https://a.espncdn.com/i/teamlogos/mlb/500/scoreboard/hou.png"},
    {"team": "KC", "logo_url": "https://a.espncdn.com/i/teamlogos/mlb/500/scoreboard/kc.png"},
    {"team": "LAA", "logo_url": "https://a.espncdn.com/i/teamlogos/mlb/500/scoreboard/laa.png"},
    {"team": "LAD", "logo_url": "https://a.espncdn.com/i/teamlogos/mlb/500/scoreboard/lad.png"},
    {"team": "MIA", "logo_url": "https://a.espncdn.com/i/teamlogos/mlb/500/scoreboard/mia.png"},
    {"team": "MIL", "logo_url": "https://a.espncdn.com/i/teamlogos/mlb/500/scoreboard/mil.png"},
    {"team": "MIN", "logo_url": "https://a.espncdn.com/i/teamlogos/mlb/500/scoreboard/min.png"},
    {"team": "NYM", "logo_url": "https://a.espncdn.com/i/teamlogos/mlb/500/scoreboard/nym.png"},
    {"team": "NYY", "logo_url": "https://a.espncdn.com/i/teamlogos/mlb/500/scoreboard/nyy.png"},
    {"team": "OAK", "logo_url": "https://a.espncdn.com/i/teamlogos/mlb/500/scoreboard/oak.png"},
    {"team": "PHI", "logo_url": "https://a.espncdn.com/i/teamlogos/mlb/500/scoreboard/phi.png"},
    {"team": "PIT", "logo_url": "https://a.espncdn.com/i/teamlogos/mlb/500/scoreboard/pit.png"},
    {"team": "SD", "logo_url": "https://a.espncdn.com/i/teamlogos/mlb/500/scoreboard/sd.png"},
    {"team": "SF", "logo_url": "https://a.espncdn.com/i/teamlogos/mlb/500/scoreboard/sf.png"},
    {"team": "SEA", "logo_url": "https://a.espncdn.com/i/teamlogos/mlb/500/scoreboard/sea.png"},
    {"team": "STL", "logo_url": "https://a.espncdn.com/i/teamlogos/mlb/500/scoreboard/stl.png"},
    {"team": "TB", "logo_url": "https://a.espncdn.com/i/teamlogos/mlb/500/scoreboard/tb.png"},
    {"team": "TEX", "logo_url": "https://a.espncdn.com/i/teamlogos/mlb/500/scoreboard/tex.png"},
    {"team": "TOR", "logo_url": "https://a.espncdn.com/i/teamlogos/mlb/500/scoreboard/tor.png"},
    {"team": "WSH", "logo_url": "https://a.espncdn.com/i/teamlogos/mlb/500/scoreboard/wsh.png"}
]
image_dict = {team["team"]: team["logo_url"] for team in mlb_teams}

# =============================
# LOAD PLAYER DATA
# =============================
DATA_PATH = "ProjSwingPlus_Output.csv"

if not os.path.exists(DATA_PATH):
    st.error(f"❌ Could not find `{DATA_PATH}` in the app directory.")
    st.stop()

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

df = load_data(DATA_PATH)

if "Team" not in df.columns:
    st.error("❌ Missing `Team` column in ProjSwingPlus_Output.csv")
    st.stop()

# =============================
# FILTERS
# =============================
st.sidebar.header("Filters")

min_age, max_age = int(df["Age"].min()), int(df["Age"].max())
age_range = st.sidebar.slider("Age Range", min_age, max_age, (min_age, 25))

df_filtered = df[(df["Age"] >= age_range[0]) & (df["Age"] <= age_range[1])]

search_name = st.sidebar.text_input("Search Player by Name")
if search_name:
    df_filtered = df_filtered[df_filtered["Name"].str.contains(search_name, case=False, na=False)]

# =============================
# TABLE WITH LOGOS
# =============================
st.subheader("📊 Player Metrics Table")

def logo_html(team):
    if pd.isna(team) or team not in image_dict:
        return ""
    return f'<img src="{image_dict[team]}" width="35">'

df_filtered["Logo"] = df_filtered["Team"].apply(logo_html)

styled_html = (
    df_filtered.sort_values("Swing+", ascending=False)
    [["Logo", "Name", "Team", "Age", "Swing+", "ProjSwing+", "PowerIndex+"]]
    .to_html(escape=False, index=False)
)

st.markdown(styled_html, unsafe_allow_html=True)

# =============================
# PLAYER DETAIL VIEW WITH LOGO
# =============================
st.subheader("🔍 Player Detail View")

player_select = st.selectbox("Select a Player", sorted(df_filtered["Name"].unique()))
player_data = df[df["Name"] == player_select].iloc[0]
team_logo = image_dict.get(player_data["Team"], None)

# Header with logo + name
if team_logo:
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:10px;">
            <img src="{team_logo}" width="60">
            <h2 style="margin:0;">{player_data['Name']}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.header(player_data["Name"])

# =============================
# RANKS + METRICS
# =============================
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
