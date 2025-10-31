import pandas as pd
import streamlit as st
import os
from PIL import Image
import requests
from io import BytesIO
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors

st.set_page_config(
    page_title="Swing+ & ProjSwing+ Dashboard",
    page_icon="⚾",
    layout="wide"
)

st.markdown(
    """
    <h1 style="text-align:center; margin-bottom:0.8em; font-size:2.7em; letter-spacing:0.02em; color:#183153;">
        Swing+ & ProjSwing+ Dashboard
    </h1>
    """,
    unsafe_allow_html=True
)

DATA_PATH = "ProjSwingPlus_Output_with_team.csv"

if not os.path.exists(DATA_PATH):
    st.error(f"Could not find `{DATA_PATH}` in the app directory.")
    st.stop()

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

df = load_data(DATA_PATH)

mlb_teams = [
    {"team": "AZ", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/ari.png&h=500&w=500"},
    {"team": "ATL", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/atl.png&h=500&w=500"},
    {"team": "BAL", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/bal.png&h=500&w=500"},
    {"team": "BOS", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/bos.png&h=500&w=500"},
    {"team": "CHC", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/chc.png&h=500&w=500"},
    {"team": "CWS", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/chw.png&h=500&w=500"},
    {"team": "CIN", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/cin.png&h=500&w=500"},
    {"team": "CLE", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/cle.png&h=500&w=500"},
    {"team": "COL", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/col.png&h=500&w=500"},
    {"team": "DET", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/det.png&h=500&w=500"},
    {"team": "HOU", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/hou.png&h=500&w=500"},
    {"team": "KC", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/kc.png&h=500&w=500"},
    {"team": "LAA", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/laa.png&h=500&w=500"},
    {"team": "LAD", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/lad.png&h=500&w=500"},
    {"team": "MIA", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/mia.png&h=500&w=500"},
    {"team": "MIL", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/mil.png&h=500&w=500"},
    {"team": "MIN", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/min.png&h=500&w=500"},
    {"team": "NYM", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/nym.png&h=500&w=500"},
    {"team": "NYY", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/nyy.png&h=500&w=500"},
    {"team": "OAK", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/oak.png&h=500&w=500"},
    {"team": "PHI", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/phi.png&h=500&w=500"},
    {"team": "PIT", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/pit.png&h=500&w=500"},
    {"team": "SD", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/sd.png&h=500&w=500"},
    {"team": "SF", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/sf.png&h=500&w=500"},
    {"team": "SEA", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/sea.png&h=500&w=500"},
    {"team": "STL", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/stl.png&h=500&w=500"},
    {"team": "TB", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/tb.png&h=500&w=500"},
    {"team": "TEX", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/tex.png&h=500&w=500"},
    {"team": "TOR", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/tor.png&h=500&w=500"},
    {"team": "WSH", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/wsh.png&h=500&w=500"}
]
df_image = pd.DataFrame(mlb_teams)
image_dict = df_image.set_index('team')['logo_url'].to_dict()

core_cols = ["Name", "Age", "Swing+", "PowerIndex+", "ProjSwing+"]
extra_cols = [
    "avg_bat_speed", "swing_length", "attack_angle", "swing_tilt", "attack_direction",
    "avg_intercept_y_vs_plate", "avg_intercept_y_vs_batter", "avg_batter_y_position", "avg_batter_x_position"
]
metric_extras = ["est_woba", "xwOBA_pred"]

if "id" in df.columns:
    core_cols = ["id"] + core_cols
if "Team" in df.columns and "Team" not in core_cols:
    core_cols.insert(1, "Team")
required_cols = core_cols + [c for c in extra_cols + metric_extras if c in df.columns]

missing = [c for c in core_cols if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

st.sidebar.header("Filters", divider="gray")

min_age, max_age = int(df["Age"].min()), int(df["Age"].max())
age_range = st.sidebar.slider("Age Range", min_age, max_age, (min_age, max_age))

df_filtered = df[(df["Age"] >= age_range[0]) & (df["Age"] <= age_range[1])]

search_name = st.sidebar.text_input("Search Player by Name")
if search_name:
    df_filtered = df_filtered[df_filtered["Name"].str.contains(search_name, case=False, na=False)]

if "swings_competitive" in df.columns:
    swings_min = int(df["swings_competitive"].min())
    swings_max = int(df["swings_competitive"].max())
    swings_range = st.sidebar.slider("Swings Competitive", swings_min, swings_max, (swings_min, swings_max))
    df_filtered = df_filtered[
        (df_filtered["swings_competitive"] >= swings_range[0]) &
        (df_filtered["swings_competitive"] <= swings_range[1])
    ]

if "batted_ball_events" in df.columns:
    bbe_min = int(df["batted_ball_events"].min())
    bbe_max = int(df["batted_ball_events"].max())
    bbe_range = st.sidebar.slider("Batted Ball Events", bbe_min, bbe_max, (bbe_min, bbe_max))
    df_filtered = df_filtered[
        (df_filtered["batted_ball_events"] >= bbe_range[0]) &
        (df_filtered["batted_ball_events"] <= bbe_range[1])
    ]

main_cmap = "RdYlBu_r"
elite_cmap = "Reds"

st.markdown(
    """
    <h2 style="text-align:center; margin-top:1.8em; margin-bottom:0.8em; font-size:2em; letter-spacing:0.01em; color:#2a3757;">
        Player Metrics Table
    </h2>
    """,
    unsafe_allow_html=True
)

display_cols = [
    c for c in [
        "Name", "Team", "Age", "Swing+", "ProjSwing+", "PowerIndex+",
        "est_woba", "xwOBA_pred",
        "avg_bat_speed", "swing_length", "attack_angle", "swing_tilt", "attack_direction",
        "avg_intercept_y_vs_plate", "avg_intercept_y_vs_batter", "avg_batter_y_position", "avg_batter_x_position"
    ] if c in df_filtered.columns
]

rename_map = {
    "Team": "Team",
    "Swing+": "Swing+",
    "ProjSwing+": "ProjSwing+",
    "PowerIndex+": "PowerIndex+",
    "avg_bat_speed": "Avg Bat Speed (mph)",
    "swing_length": "Swing Length (m)",
    "attack_angle": "Attack Angle (°)",
    "swing_tilt": "Swing Tilt (°)",
    "attack_direction": "Attack Direction",
    "avg_intercept_y_vs_plate": "Intercept Y vs Plate",
    "avg_intercept_y_vs_batter": "Intercept Y vs Batter",
    "avg_batter_y_position": "Batter Y Pos",
    "avg_batter_x_position": "Batter X Pos",
    "est_woba": "xwOBA",
    "xwOBA_pred": "Predicted xwOBA"
}

styled_df = (
    df_filtered[display_cols]
    .rename(columns=rename_map)
    .sort_values("Swing+", ascending=False)
    .reset_index(drop=True)
    .style.background_gradient(
        subset=[c for c in ["Swing+", "ProjSwing+", "PowerIndex+", "xwOBA", "Predicted xwOBA"] if c in rename_map.values()],
        cmap=main_cmap
    )
    .format(precision=3)
)

st.dataframe(styled_df, use_container_width=True, hide_index=True)

st.markdown(
    """
    <h2 style="text-align:center; margin-top:1.8em; margin-bottom:0.8em; font-size:2em; letter-spacing:0.01em; color:#2a3757;">
        Top 10 Leaderboards
    </h2>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        <div style="text-align:center; font-size:1.25em; font-weight:600; margin-bottom:0.7em; color:#385684;">
            Top 10 by Swing+
        </div>
        """,
        unsafe_allow_html=True
    )
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
    st.markdown(
        """
        <div style="text-align:center; font-size:1.25em; font-weight:600; margin-bottom:0.7em; color:#385684;">
            Top 10 by ProjSwing+
        </div>
        """,
        unsafe_allow_html=True
    )
    top_proj = df_filtered.sort_values("ProjSwing+", ascending=False).head(10).reset_index(drop=True)
    leaderboard_cols = [c for c in ["Name", "Team", "Age", "ProjSwing+", "Swing+", "PowerIndex+"] if c in top_proj.columns]
    st.dataframe(
        top_proj[leaderboard_cols]
        .style.background_gradient(subset=["ProjSwing+"], cmap=elite_cmap)
        .format(precision=1),
        use_container_width=True,
        hide_index=True
    )

st.markdown(
    """
    <h2 style="text-align:center; margin-top:1.8em; margin-bottom:0.7em; font-size:2em; letter-spacing:0.01em; color:#2a3757;">
        Player Detail View
    </h2>
    """,
    unsafe_allow_html=True
)

player_select = st.selectbox(
    "Select a Player",
    sorted(df_filtered["Name"].unique()),
    key="player_select"
)
player_row = df[df["Name"] == player_select].iloc[0]

headshot_size = 96
logo_size = 80

team_abb = player_row["Team"] if "Team" in player_row and pd.notnull(player_row["Team"]) else ""
logo_url = image_dict.get(team_abb, "")

headshot_html = ""
if "id" in player_row and pd.notnull(player_row["id"]):
    player_id = str(int(player_row["id"]))
    headshot_url = f"https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/w_640,q_auto:best/v1/people/{player_id}/headshot/silo/current.png"
    headshot_html = f'<img src="{headshot_url}" style="height:{headshot_size}px;width:{headshot_size}px;object-fit:cover;border-radius:14px;vertical-align:middle;box-shadow:0 1px 6px #0001;margin-right:30px;" alt="headshot"/>'
else:
    fallback_url = "https://img.mlbstatic.com/mlb-photos/image/upload/v1/people/0/headshot/silo/current.png"
    headshot_html = f'<img src="{fallback_url}" style="height:{headshot_size}px;width:{headshot_size}px;object-fit:cover;border-radius:14px;vertical-align:middle;box-shadow:0 1px 6px #0001;margin-right:30px;" alt="headshot"/>'

logo_html = ""
if logo_url:
    logo_html = f'<img src="{logo_url}" style="height:{logo_size}px;width:{logo_size}px;vertical-align:middle;margin-left:46px;background:transparent;border-radius:0;" alt="logo"/>'

player_name_html = f'<span style="font-size:2.3em;font-weight:800;color:#183153;letter-spacing:0.01em;vertical-align:middle;margin:0 20px;">{player_select}</span>'

player_bio = ""
bat_side = "R"
if "id" in player_row and pd.notnull(player_row["id"]):
    player_id = str(int(player_row["id"]))
    mlb_bio_url = f"https://statsapi.mlb.com/api/v1/people/{player_id}"
    try:
        resp = requests.get(mlb_bio_url, timeout=4)
        if resp.status_code == 200:
            data = resp.json()
            if "people" in data and len(data["people"]) > 0:
                person = data["people"][0]
                if "batSide" in person and "code" in person["batSide"]:
                    bat_side = person["batSide"]["code"]
                bio_parts = []
                if "height" in person and "weight" in person:
                    bio_parts.append(f"{person['height']}, {person['weight']} lbs")
                bt = []
                if "batSide" in person and "code" in person["batSide"]:
                    bt.append(person["batSide"]["code"])
                if "pitchHand" in person and "code" in person["pitchHand"]:
                    bt.append(person["pitchHand"]["code"])
                if bt:
                    bio_parts.append(f"B/T: {'/'.join(bt)}")
                if "currentAge" in person:
                    bio_parts.append(f"Age: {person['currentAge']}")
                location = []
                if "birthCity" in person:
                    location.append(person["birthCity"])
                if "birthStateProvince" in person and person["birthStateProvince"]:
                    location.append(person["birthStateProvince"])
                if "birthCountry" in person:
                    location.append(person["birthCountry"])
                if location:
                    bio_parts.append(", ".join(location))
                player_bio = " &nbsp; | &nbsp; ".join(bio_parts)
    except Exception:
        player_bio = ""

st.markdown(
    f"""
    <div style="display:flex;justify-content:center;align-items:center;margin-bottom:0px;margin-top:8px;">
        {headshot_html}
        <div style="display:flex;flex-direction:column;align-items:center;">
            {player_name_html}
            {"<span style='font-size:0.98em;color:#495366;margin-top:7px;margin-bottom:0;font-weight:500;letter-spacing:0.02em;opacity:0.82;'>" + player_bio + "</span>" if player_bio else ""}
        </div>
        {logo_html}
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("<div style='height:26px;'></div>", unsafe_allow_html=True)

total_players = len(df)
df["Swing+_rank"] = df["Swing+"].rank(ascending=False, method="min").astype(int)
df["ProjSwing+_rank"] = df["ProjSwing+"].rank(ascending=False, method="min").astype(int)
df["PowerIndex+_rank"] = df["PowerIndex+"].rank(ascending=False, method="min").astype(int)

p_swing_rank = df.loc[df["Name"] == player_select, "Swing+_rank"].iloc[0]
p_proj_rank = df.loc[df["Name"] == player_select, "ProjSwing+_rank"].iloc[0]
p_power_rank = df.loc[df["Name"] == player_select, "PowerIndex+_rank"].iloc[0]

def plus_color(val, vmin=663, vmax=1400, cmap="RdYlBu_r"):
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    rgb = cm.get_cmap(cmap)(norm(val))[:3]
    hex_color = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
    return hex_color

swing_color = plus_color(player_row['Swing+'])
proj_color = plus_color(player_row['ProjSwing+'])
power_color = plus_color(player_row['PowerIndex+'])

st.markdown(
    f"""
    <div style="display: flex; justify-content: center; gap: 32px; margin-top: 0px; margin-bottom: 28px;">
      <div style="background: #fff; border-radius: 16px; box-shadow: 0 2px 12px #0001; padding: 24px 32px; text-align: center; min-width: 160px;">
        <div style="font-size: 2.2em; font-weight: 700; color: {swing_color};">{player_row['Swing+']:.1f}</div>
        <div style="font-size: 1.1em; color: #888; font-weight: 600; letter-spacing: 0.5px; margin-bottom: 4px;">Swing+</div>
        <span style="background: #FFC10733; color: #B71C1C; border-radius: 10px; font-size: 0.98em; padding: 2px 10px 2px 10px;">Rank {p_swing_rank} of {total_players}</span>
      </div>
      <div style="background: #fff; border-radius: 16px; box-shadow: 0 2px 12px #0001; padding: 24px 32px; text-align: center; min-width: 160px;">
        <div style="font-size: 2.2em; font-weight: 700; color: {proj_color};">{player_row['ProjSwing+']:.1f}</div>
        <div style="font-size: 1.1em; color: #888; font-weight: 600; letter-spacing: 0.5px; margin-bottom: 4px;">ProjSwing+</div>
        <span style="background: #C8E6C933; color: #1B5E20; border-radius: 10px; font-size: 0.98em; padding: 2px 10px 2px 10px;">Rank {p_proj_rank} of {total_players}</span>
      </div>
      <div style="background: #fff; border-radius: 16px; box-shadow: 0 2px 12px #0001; padding: 24px 32px; text-align: center; min-width: 160px;">
        <div style="font-size: 2.2em; font-weight: 700; color: {power_color};">{player_row['PowerIndex+']:.1f}</div>
        <div style="font-size: 1.1em; color: #888; font-weight: 600; letter-spacing: 0.5px; margin-bottom: 4px;">PowerIndex+</div>
        <span style="background: #B3E5FC33; color: #01579B; border-radius: 10px; font-size: 0.98em; padding: 2px 10px 2px 10px;">Rank {p_power_rank} of {total_players}</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

video_url = f"https://builds.mlbstatic.com/baseballsavant.mlb.com/swing-path/splendid-splinter/cut/{player_id}-2025-{bat_side}.mp4"

DEFAULT_ONEIL_CRUZ_IDS = ['665833-2025-L', '665833-2025-R', '665833-2025-S']
default_name = "Oneil Cruz"
showing_default = f'{player_id}-2025-{bat_side}' in DEFAULT_ONEIL_CRUZ_IDS

if showing_default:
    video_note = (
        f"No custom video data available for this player — showing a default example ({default_name})."
    )
else:
    video_note = (
        "Below is the Baseball Savant Swing Path / Attack Angle visualization for this player."
    )

st.markdown(
    f"""
    <h3 style="text-align:center; margin-top:1.3em; font-size:1.08em; color:#183153; letter-spacing:0.01em;">
        Baseball Savant Swing Path / Attack Angle Visualization
    </h3>
    <div style="text-align:center; color: #7a7a7a; font-size: 0.99em; margin-bottom:10px">
        {video_note}
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <div id="savantviz-anchor"></div>
    <div style="display: flex; justify-content: center;">
        <video id="player-savant-video" width="900" height="480" style="border-radius:9px; box-shadow:0 2px 12px #0002;" autoplay muted playsinline key="{player_id}-{bat_side}">
            <source src="{video_url}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <h3 style="text-align:center; margin-top:2em; font-size:1.22em; color:#183153; letter-spacing:0.01em;">
        Mechanical Similarity Cluster
    </h3>
    """,
    unsafe_allow_html=True
)

mechanical_features = [
    "avg_bat_speed",
    "swing_tilt",
    "attack_angle",
    "attack_direction",
    "avg_intercept_y_vs_plate",
    "avg_intercept_y_vs_batter",
    "avg_batter_y_position",
    "avg_batter_x_position",
    "swing_length"
]

name_col = "Name"
TOP_N = 10

mech_features_available = [f for f in mechanical_features if f in df.columns]
if len(mech_features_available) >= 2 and name_col in df.columns:
    df_mech = df.dropna(subset=mech_features_available + [name_col]).reset_index(drop=True)
    if player_select in df_mech[name_col].values and len(df_mech) > TOP_N:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_mech[mech_features_available])
        similarity_matrix = cosine_similarity(X_scaled)
        similarity_df = pd.DataFrame(similarity_matrix, index=df_mech[name_col], columns=df_mech[name_col])

        similar_players = (
            similarity_df.loc[player_select]
            .sort_values(ascending=False)
            .iloc[1:TOP_N+1]
        )

        top_names = [player_select] + list(similar_players.index)
        sim_rows = []
        for sim_name in similar_players.index:
            sim_row = df_mech[df_mech[name_col] == sim_name]
            if "id" in sim_row.columns and pd.notnull(sim_row.iloc[0]["id"]):
                sim_id = str(int(sim_row.iloc[0]["id"]))
                sim_headshot_url = f"https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/w_640,q_auto:best/v1/people/{sim_id}/headshot/silo/current.png"
            else:
                sim_headshot_url = "https://img.mlbstatic.com/mlb-photos/image/upload/v1/people/0/headshot/silo/current.png"
            sim_score = similar_players[sim_name]
            sim_rows.append({
                "name": sim_name,
                "headshot_url": sim_headshot_url,
                "score": sim_score
            })

        st.markdown(
            "<div style='display:flex;flex-direction:column;align-items:center;max-width:920px;margin:0 auto;'>",
            unsafe_allow_html=True
        )

        for row in range(2):  # two rows
            st.markdown(
                "<div style='display:flex;flex-wrap:nowrap;justify-content:center;gap:18px;margin-bottom:20px;'>",
                unsafe_allow_html=True
            )
            for col in range(5):  # five columns
                idx = row * 5 + col
                if idx >= len(sim_rows):
                    continue
                sim = sim_rows[idx]
                st.markdown(
                    f"""
                    <div style="background:#fff;border-radius:14px;box-shadow:0 2px 8px #0001;
                                padding:18px 13px 13px 13px;width:168px;text-align:center;">
                      <img src="{sim['headshot_url']}" style="height:74px;width:74px;object-fit:cover;
                                border-radius:12px;box-shadow:0 1px 5px #0001;margin-bottom:8px;" alt="headshot"/>
                  <div style="font-size:1.01em;font-weight:700;color:#183153;margin:2px 0 2px 0;
                                      white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{sim['name']}</div>
                      <div style="font-size:0.98em;font-weight:600;color:#385684;margin-top:4px;">
                        Similarity: <span style='color:#B71036;'>{sim['score']:.2f}</span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True
                )
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("Show Heatmap"):
            fig, ax = plt.subplots(figsize=(6, 4.2))
            heatmap_data = similarity_df.loc[top_names, top_names]
            sns.heatmap(
                heatmap_data,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                linewidths=0.5,
                cbar_kws={"label": "Cosine Similarity"},
                ax=ax,
                annot_kws={"fontsize":8}
            )
            ax.set_title(f"Mechanical Similarity Cluster: {player_select}", fontsize=12, weight="bold")
            plt.xticks(rotation=45, ha='right', fontsize=8)
            plt.yticks(fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
        st.markdown(
            f"<div style='text-align:center;margin-top:10px;font-size:1.08em;color:#385684;'>Top {TOP_N} mechanically similar players to <b>{player_select}</b> shown above.</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div style='text-align:center;margin-top:10px;font-size:1.08em;color:#C62828;'>No mechanical similarity data available for this player.</div>",
            unsafe_allow_html=True
        )
else:
    st.markdown(
        "<div style='text-align:center;margin-top:10px;font-size:1.08em;color:#C62828;'>No mechanical similarity data available for this player.</div>",
        unsafe_allow_html=True
    )

if set(extra_cols).issubset(df.columns):
    st.markdown(
        """
        <h3 style="text-align:center; margin-top:2em; font-size:1.22em; color:#183153; letter-spacing:0.01em;">
            Swing Mechanics
        </h3>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <div style="display: flex; justify-content: center; gap: 32px; margin-top: 10px; margin-bottom: 15px;">
        """,
        unsafe_allow_html=True
    )
    mech_metrics = [
        ("Avg Bat Speed", f"{round(player_row['avg_bat_speed'], 1)} mph" if "avg_bat_speed" in player_row else ""),
        ("Swing Length", f"{round(player_row['swing_length'], 2)}" if "swing_length" in player_row else ""),
        ("Attack Angle", f"{round(player_row['attack_angle'], 1)}" if "attack_angle" in player_row else ""),
        ("Swing Tilt", f"{round(player_row['swing_tilt'], 1)}" if "swing_tilt" in player_row else ""),
        ("Attack Direction", f"{round(player_row['attack_direction'], 1)}" if "attack_direction" in player_row else "")
    ]
    for label, value in mech_metrics:
        st.markdown(
            f"""
            <div style="background:#f9fafc;border-radius:12px;box-shadow:0 1px 6px #0001;padding:18px 22px;min-width:110px;text-align:center;">
              <div style="font-size:1.1em;color:#385684;font-weight:600;margin-bottom:2px;">{label}</div>
              <div style="font-size:1.5em;font-weight:700;color:#183153;">{value}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    st.markdown(
        """
        </div>
        """,
        unsafe_allow_html=True
    )
