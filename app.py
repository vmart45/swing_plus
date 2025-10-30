import pandas as pd
import streamlit as st
import requests
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
# MLB LOGOS DICTIONARY
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
image_dict = {t["team"]: t["logo_url"] for t in mlb_teams}

TEAM_MAP = {
    "ARI": "AZ", "CHW": "CWS", "KCR": "KC", "SDP": "SD", "SFG": "SF",
    "TBR": "TB", "WSN": "WSH", "LAA": "LAA", "LAD": "LAD"
}

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

if "id" not in df.columns:
    st.error("❌ Missing `id` column in ProjSwingPlus_Output.csv")
    st.stop()

# =============================
# FETCH TEAMS USING MLB API
# =============================
@st.cache_data(show_spinner=False)
def get_team_from_api(pid: int):
    """Fetch current team abbreviation for a player via MLB Stats API."""
    try:
        url = f"https://statsapi.mlb.com/api/v1/people?personIds={pid}&hydrate=currentTeam"
        r = requests.get(url, timeout=5).json()
        team_link = r["people"][0]["currentTeam"]["link"]
        team_data = requests.get(f"https://statsapi.mlb.com{team_link}", timeout=5).json()
        raw_abbr = team_data["teams"][0]["abbreviation"]
        return TEAM_MAP.get(raw_abbr, raw_abbr)
    except Exception:
        return None

st.info("Fetching team data from MLB API (cached). Runs once per player ID…")

df["Team"] = df["id"].apply(get_team_from_api)
df["Logo"] = df["Team"].map(image_dict)

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
# TABLE WITH TEAM LOGOS
# =============================
st.subheader("📊 Player Metrics Table")

def logo_html(url):
    return f'<img src="{url}" width="35">' if pd.notna(url) else ""

df_filtered["Logo"] = df_filtered["Logo"].apply(logo_html)

table_html = (
    df_filtered.sort_values("Swing+", ascending=False)
    [["Logo", "Name", "Team", "Age", "Swing+", "ProjSwing+", "PowerIndex+"]]
    .to_html(escape=False, index=False)
)
st.markdown(table_html, unsafe_allow_html=True)

# =============================
# PLAYER DETAIL VIEW
# =============================
st.subheader("🔍 Player Detail View")

player_select = st.selectbox("Select a Player", sorted(df_filtered["Name"].unique()))
player_row = df[df["Name"] == player_select].iloc[0]

team_logo = player_row["Logo"]
team_name = player_row["Team"]

# Player header with logo
if team_logo:
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:10px;">
            {team_logo}
            <h2 style="margin:0;">{player_row['Name']} ({team_name})</h2>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.header(player_row["Name"])

# =============================
# RANKED METRICS
# =============================
total_players = len(df)
for col in ["Swing+", "ProjSwing+", "PowerIndex+"]:
    df[f"{col}_rank"] = df[col].rank(ascending=False, method="min").astype(int)

col1, col2, col3 = st.columns(3)
col1.metric("Swing+", f"{player_row['Swing+']:.1f}", f"Rank {int(player_row['Swing+_rank'])}/{total_players}")
col2.metric("ProjSwing+", f"{player_row['ProjSwing+']:.1f}", f"Rank {int(player_row['ProjSwing+_rank'])}/{total_players}")
col3.metric("PowerIndex+", f"{player_row['PowerIndex+']:.1f}", f"Rank {int(player_row['PowerIndex+_rank'])}/{total_players}")
