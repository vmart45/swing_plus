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
Explore **Swing+**, **ProjSwing+**, and **PowerIndex+**""")

# =============================
# MLB LOGOS
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
    {"team": "ATH", "logo_url": "https://a.espncdn.com/i/teamlogos/mlb/500/scoreboard/oak.png"},
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
TEAM_MAP = {"ARI": "AZ", "CHW": "CWS", "KCR": "KC", "SDP": "SD", "SFG": "SF", "TBR": "TB", "WSN": "WSH"}

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
# FETCH TEAMS VIA MLB API
# =============================
@st.cache_data(show_spinner=False)
def get_team_from_api(pid: int):
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
# COMPUTE RANKS (BEFORE SELECTION)
# =============================
total_players = len(df)
for col in ["Swing+", "ProjSwing+", "PowerIndex+"]:
    df[f"{col}_rank"] = df[col].rank(ascending=False, method="min").astype(int)

# =============================
# PLAYER METRICS TABLE (SCROLLABLE)
# =============================
st.subheader("📊 Player Metrics Table")

logo_md = df_filtered["Team"].map(lambda t: f"![]({image_dict.get(t, '')})" if pd.notna(t) else "")
display_df = pd.concat([logo_md.rename("Logo"), df_filtered[["Name", "Team", "Age", "Swing+", "ProjSwing+", "PowerIndex+"]]], axis=1)

st.dataframe(display_df.style.format(precision=1), use_container_width=True)

# =============================
# LEADERBOARDS
# =============================
st.subheader("🏆 Top 10 Leaderboards")
col1, col2 = st.columns(2)

def make_leaderboard(df_in, sort_col):
    df_top = df_in.sort_values(sort_col, ascending=False).head(10)
    df_top["Logo"] = df_top["Team"].map(lambda t: f"![]({image_dict.get(t, '')})" if pd.notna(t) else "")
    cols = ["Logo", "Name", "Team", sort_col, "Swing+", "ProjSwing+" if sort_col != "ProjSwing+" else "PowerIndex+"]
    return df_top[cols]

with col1:
    st.markdown("**Top 10 by ProjSwing+**")
    st.dataframe(make_leaderboard(df_filtered, "ProjSwing+").style.format(precision=1), use_container_width=True)

with col2:
    st.markdown("**Top 10 by PowerIndex+**")
    st.dataframe(make_leaderboard(df_filtered, "PowerIndex+").style.format(precision=1), use_container_width=True)

# =============================
# PLAYER DETAIL VIEW
# =============================
st.subheader("🔍 Player Detail View")

player_select = st.selectbox("Select a Player", sorted(df_filtered["Name"].unique()))
player_row = df[df["Name"] == player_select].iloc[0]
team_logo = image_dict.get(player_row["Team"], "")
team_name = player_row["Team"]

if team_logo:
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:10px;">
            <img src="{team_logo}" width="45">
            <h2 style="margin:0;">{player_row['Name']} ({team_name})</h2>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.header(player_row["Name"])

colA, colB, colC = st.columns(3)
colA.metric("Swing+", f"{player_row['Swing+']:.1f}", f"Rank {int(player_row['Swing+_rank'])}/{total_players}")
colB.metric("ProjSwing+", f"{player_row['ProjSwing+']:.1f}", f"Rank {int(player_row['ProjSwing+_rank'])}/{total_players}")
colC.metric("PowerIndex+", f"{player_row['PowerIndex+']:.1f}", f"Rank {int(player_row['PowerIndex+_rank'])}/{total_players}")
