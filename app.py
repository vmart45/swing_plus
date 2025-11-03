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
import numpy as np
import pickle
import joblib
import shap
import plotly.graph_objects as go
import streamlit.components.v1 as components
from urllib.parse import quote, unquote

st.set_page_config(
    page_title="Swing+ & ProjSwing+ Dashboard",
    page_icon="⚾",
    layout="wide"
)

st.markdown(
    """
    <h1 style="text-align:center; margin-bottom:0.8em; font-size:2.4em; letter-spacing:0.02em; color:#183153;">
        Swing+ & ProjSwing+ Dashboard
    </h1>
    """,
    unsafe_allow_html=True
)

DATA_PATH = "ProjSwingPlus_Output_with_team.csv"
MODEL_PATH = "swingplus_model.pkl"
DOC_FILENAME = "SwingPlus_Documentation.pdf"
DOC_RAW_URL = "https://raw.githubusercontent.com/vmart45/swing_plus/14381a10958c94c746c86b971b07136f4557f855/SwingPlus_Documentation.pdf"

if not os.path.exists(DATA_PATH):
    st.error(f"Could not find `{DATA_PATH}` in the app directory.")
    st.stop()

st.sidebar.header("Resources", divider="gray")
st.sidebar.markdown(
    f'<a href="{DOC_RAW_URL}" target="_blank" rel="noopener noreferrer">Download SwingPlus Documentation</a>',
    unsafe_allow_html=True
)

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

df = load_data(DATA_PATH)

# =====================================================
# Fix potential renamed column: avg_batter_position -> avg_batter_x_position
# If the CSV contains avg_batter_position (old name) but code expects avg_batter_x_position,
# create the expected column so downstream code works without changes.
# Also handle the inverse if somehow only avg_batter_x_position exists but code expects avg_batter_position.
# =====================================================
if "avg_batter_position" in df.columns and "avg_batter_x_position" not in df.columns:
    df["avg_batter_x_position"] = df["avg_batter_position"]
elif "avg_batter_x_position" in df.columns and "avg_batter_position" not in df.columns:
    # keep both for safety
    df["avg_batter_position"] = df["avg_batter_x_position"]

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
    swings_range = st.sidebar.slider("Competitive Swings", swings_min, swings_max, (swings_min, swings_max))
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

# Create top-level tabs: Main (metrics + leaderboards), Player (detail), Glossary
tab_main, tab_player, tab_glossary = st.tabs(["Main", "Player", "Glossary"])

# ---------------- Main tab: Metrics table and leaderboards ----------------
with tab_main:
    st.markdown(
        """
        <h2 style="text-align:center; margin-top:1.2em; margin-bottom:0.6em; font-size:1.6em; letter-spacing:0.01em; color:#2a3757;">
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

    # Friendly display names for mechanical features
    FEATURE_LABELS = {
        "avg_bat_speed": "Avg Bat Speed (mph)",
        "swing_length": "Swing Length (m)",
        "attack_angle": "Attack Angle (°)",
        "swing_tilt": "Swing Tilt (°)",
        "attack_direction": "Attack Direction",
        "avg_intercept_y_vs_plate": "Intercept Y vs Plate",
        "avg_intercept_y_vs_batter": "Intercept Y vs Batter",
        "avg_batter_y_position": "Batter Y Pos",
        "avg_batter_x_position": "Batter X Pos"
    }

    rename_map = {
        "Team": "Team",
        "Swing+": "Swing+",
        "ProjSwing+": "ProjSwing+",
        "PowerIndex+": "PowerIndex+",
        "est_woba": "xwOBA",
        "xwOBA_pred": "Predicted xwOBA"
    }
    # extend rename_map with friendly names
    for k, v in FEATURE_LABELS.items():
        if k in df.columns:
            rename_map[k] = v

    styled_df = (
        df_filtered[display_cols]
        .rename(columns=rename_map)
        .sort_values("Swing+", ascending=False)
        .reset_index(drop=True)
        .style.background_gradient(
            subset=[c for c in ["Swing+", "ProjSwing+", "PowerIndex+", "xwOBA", "Predicted xwOBA"] if c in rename_map.values()],
            cmap=main_cmap
        )
        .format(precision=2)
    )

    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    st.markdown(
        """
        <h2 style="text-align:center; margin-top:1.2em; margin-bottom:0.6em; font-size:1.6em; letter-spacing:0.01em; color:#2a3757;">
            Top 10 Leaderboards
        </h2>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div style="text-align:center; font-size:1.15em; font-weight:600; margin-bottom:0.6em; color:#385684;">
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
            <div style="text-align:center; font-size:1.15em; font-weight:600; margin-bottom:0.6em; color:#385684;">
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

# ---------------- Player tab: Player Detail view ----------------
with tab_player:
    st.markdown(
        """
        <h2 style="text-align:center; margin-top:1.2em; margin-bottom:0.6em; font-size:1.6em; letter-spacing:0.01em; color:#2a3757;">
            Player Detail
        </h2>
        """,
        unsafe_allow_html=True
    )

    # Allow deep-linking to a player via URL query param ?player=Player+Name
    params = st.experimental_get_query_params()
    qp_player = None
    if "player" in params and len(params["player"]) > 0:
        try:
            qp_player = unquote(params["player"][0])
        except Exception:
            qp_player = params["player"][0]

    player_options = sorted(df_filtered["Name"].unique())
    default_index = 0
    if qp_player and qp_player in player_options:
        default_index = player_options.index(qp_player)

    player_select = st.selectbox(
        "Select a Player",
        player_options,
        key="player_select",
        index=default_index
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
        headshot_html = (
            f'<img src="{headshot_url}" '
            f'style="height:{headshot_size}px;width:{headshot_size}px;object-fit:cover;border-radius:14px;vertical-align:middle;'
            f'box-shadow:0 1px 6px rgba(0,0,0,0.06);margin-right:18px;" alt="headshot"/>'
        )
    else:
        fallback_url = "https://img.mlbstatic.com/mlb-photos/image/upload/v1/people/0/headshot/silo/current.png"
        headshot_html = (
            f'<img src="{fallback_url}" '
            f'style="height:{headshot_size}px;width:{headshot_size}px;object-fit:cover;border-radius:14px;vertical-align:middle;'
            f'box-shadow:0 1px 6px rgba(0,0,0,0.06);margin-right:18px;" alt="headshot"/>'
        )

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
        <div style="display:flex;justify-content:center;align-items:center;margin-bottom:6px;margin-top:8px;">
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

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

    total_players = len(df)
    df["Swing+_rank"] = df["Swing+"].rank(ascending=False, method="min").astype(int)
    df["ProjSwing+_rank"] = df["ProjSwing+"].rank(ascending=False, method="min").astype(int)
    df["PowerIndex+_rank"] = df["PowerIndex+"].rank(ascending=False, method="min").astype(int)

    p_swing_rank = df.loc[df["Name"] == player_select, "Swing+_rank"].iloc[0]
    p_proj_rank = df.loc[df["Name"] == player_select, "ProjSwing+_rank"].iloc[0]
    p_power_rank = df.loc[df["Name"] == player_select, "PowerIndex+_rank"].iloc[0]

    # New plus_color: color by rank (1 = reddest, max = bluest)
    def plus_color_by_rank(rank, total, start_hex="#D32F2F", end_hex="#3B82C4"):
        # clamp
        if total <= 1:
            ratio = 0.0
        else:
            ratio = (rank - 1) / (total - 1)  # 0 => best (rank 1), 1 => worst
        # We want rank=1 -> red (start_hex), rank=total -> blue (end_hex)
        def hex_to_rgb(h):
            h = h.lstrip("#")
            return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
        def rgb_to_hex(rgb):
            return "#{:02x}{:02x}{:02x}".format(*[int(max(0, min(255, round(x)))) for x in rgb])
        sr, sg, sb = hex_to_rgb(start_hex)
        er, eg, eb = hex_to_rgb(end_hex)
        rr = sr + (er - sr) * ratio
        rg = sg + (eg - sg) * ratio
        rb = sb + (eb - sb) * ratio
        return rgb_to_hex((rr, rg, rb))

    # Use colors computed from ranks so best players (rank=1) are redder, worst are bluer.
    swing_color = plus_color_by_rank(p_swing_rank, total_players)
    proj_color = plus_color_by_rank(p_proj_rank, total_players)
    power_color = plus_color_by_rank(p_power_rank, total_players)

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

    # ------------------- Load Swing+ model and create SHAP explainer -------------------
    model = None
    explainer = None
    model_loaded = False
    model_error = None

    if os.path.exists(MODEL_PATH):
        try:
            try:
                model = joblib.load(MODEL_PATH)
            except Exception:
                with open(MODEL_PATH, "rb") as f:
                    model = pickle.load(f)
            model_loaded = True
            try:
                explainer = shap.TreeExplainer(model)
            except Exception:
                try:
                    explainer = shap.Explainer(model)
                except Exception as e:
                    explainer = None
                    model_error = str(e)
        except Exception as e:
            model_loaded = False
            model_error = str(e)
    else:
        model_loaded = False
        model_error = f"Model file not found at {MODEL_PATH}"

    def prepare_model_input_for_player(player_row, feature_list_fallback, model_obj, df_reference=None):
        expected = None
        try:
            if hasattr(model_obj, "feature_name_") and model_obj.feature_name_ is not None:
                expected = list(model_obj.feature_name_)
            elif hasattr(model_obj, "booster_") and hasattr(model_obj.booster_, "feature_name"):
                expected = list(model_obj.booster_.feature_name())
            elif hasattr(model_obj, "get_booster") and hasattr(model_obj.get_booster(), "feature_name"):
                expected = list(model_obj.get_booster().feature_name())
        except Exception:
            expected = None

        if expected is None or len(expected) == 0:
            expected = list(feature_list_fallback)

        row = {}
        for feat in expected:
            if feat in player_row and pd.notna(player_row[feat]):
                row[feat] = player_row[feat]
            else:
                alt_found = False
                for alt in [feat, feat.replace(" ", "_"), feat.lower()]:
                    if alt in player_row and pd.notna(player_row[alt]):
                        row[feat] = player_row[alt]
                        alt_found = True
                        break
                if not alt_found:
                    if df_reference is not None and feat in df_reference.columns:
                        try:
                            row[feat] = float(df_reference[feat].mean())
                        except Exception:
                            row[feat] = 0.0
                    else:
                        row[feat] = 0.0

        X_raw = pd.DataFrame([row], columns=expected)
        for c in X_raw.columns:
            X_raw[c] = pd.to_numeric(X_raw[c], errors="coerce").astype(float)
            if X_raw[c].isna().any():
                if df_reference is not None and c in df_reference.columns:
                    X_raw[c] = X_raw[c].fillna(float(df_reference[c].mean()))
                else:
                    X_raw[c] = X_raw[c].fillna(0.0)
        return X_raw

    # Compute SHAP values for the selected player (Swing+ only)
    shap_df = None
    shap_base = None
    shap_pred = None
    shap_values_arr = None

    mech_features_available = [f for f in mechanical_features if f in df.columns]

    if model_loaded and explainer is not None and len(mech_features_available) >= 2:
        try:
            X_player = prepare_model_input_for_player(player_row, mech_features_available, model, df_reference=df)

            try:
                expected_names = None
                if hasattr(model, "feature_name_") and model.feature_name_ is not None:
                    expected_names = list(model.feature_name_)
                elif hasattr(model, "booster_") and hasattr(model.booster_, "feature_name"):
                    expected_names = list(model.booster_.feature_name())
                elif hasattr(model, "get_booster") and hasattr(model.get_booster(), "feature_name"):
                    expected_names = list(model.get_booster().feature_name())
                if expected_names is not None and X_player.shape[1] != len(expected_names):
                    model_error = f"Prepared input has {X_player.shape[1]} features but model expects {len(expected_names)} features."
                    raise ValueError(model_error)
            except Exception:
                pass

            try:
                shap_pred = float(model.predict(X_player)[0])
            except Exception:
                shap_pred = float(model.predict(X_player.values.reshape(1, -1))[0])

            try:
                shap_values = explainer(X_player)
            except Exception:
                shap_values = explainer(X_player.values)

            if hasattr(shap_values, "values"):
                shap_values_arr = np.array(shap_values.values).flatten()
                shap_base = float(shap_values.base_values) if np.size(shap_values.base_values) == 1 else float(shap_values.base_values.flatten()[0])
            else:
                shap_values_arr = np.array(shap_values).flatten()
                shap_base = None

            shap_df = pd.DataFrame({
                "feature": X_player.columns.tolist(),
                "raw": [float(X_player.iloc[0][f]) if pd.notna(X_player.iloc[0][f]) else np.nan for f in X_player.columns],
                "shap_value": shap_values_arr
            })
            shap_df["abs_shap"] = np.abs(shap_df["shap_value"])
            total_abs = shap_df["abs_shap"].sum() if shap_df["abs_shap"].sum() != 0 else 1.0
            shap_df["pct_of_abs"] = shap_df["abs_shap"] / total_abs
            shap_df = shap_df.sort_values("abs_shap", ascending=False).reset_index(drop=True)

        except Exception as e:
            shap_df = None
            model_error = str(e)

    # ------------------ Display Swing+ SHAP panel (interactive chart + table) ------------------
    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <h3 style="text-align:center; margin-top:6px; font-size:1.08em; color:#183153; letter-spacing:0.01em;">
            Swing+ Feature Contributions (SHAP)
        </h3>
        <div style="text-align:center; color:#6b7280; margin-bottom:6px; font-size:0.95em;">
            How each mechanical feature moved the model's Swing+ prediction for this player.
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([1, 1])

    shap_pred_label = f"{shap_pred:.2f}" if (shap_pred is not None and not pd.isna(shap_pred)) else "N/A"
    swing_actual_label = f"{player_row['Swing+']:.2f}" if (player_row.get("Swing+") is not None and not pd.isna(player_row.get("Swing+"))) else "N/A"
    base_label = f"{shap_base:.2f}" if (shap_base is not None and not pd.isna(shap_base)) else "N/A"

    with col1:
        st.markdown(f"<div style='text-align:center;font-weight:700;color:#183153;'>Model prediction: {shap_pred_label} &nbsp; | &nbsp; Actual Swing+: {swing_actual_label}</div>", unsafe_allow_html=True)
        if not model_loaded or explainer is None or shap_df is None or len(shap_df) == 0:
            st.info("Swing+ model or SHAP explainer not available. Ensure swingplus_model.pkl is a supported model/pipeline.")
            if model_error:
                st.caption(f"Model load error: {model_error}")
        else:
            TOP_SHOW = min(8, len(shap_df))
            df_plot_top = shap_df.head(TOP_SHOW).copy()
            # Order by pct_of_abs descending so largest importance at top
            df_plot_top = df_plot_top.sort_values("pct_of_abs", ascending=False).reset_index(drop=True)

            y = df_plot_top["feature"].map(lambda x: FEATURE_LABELS.get(x, x)).tolist()
            x_vals = df_plot_top["shap_value"].astype(float).tolist()
            pct_vals = df_plot_top["pct_of_abs"].astype(float).tolist()
            colors = ["#D8573C" if float(v) > 0 else "#3B82C4" for v in x_vals]

            # Keep text inside bars and show both contribution and percentage
            text_labels = [f"{val:.3f}  ({pct:.0%})" for val, pct in zip(x_vals, pct_vals)]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=x_vals,
                y=y,
                orientation='h',
                marker_color=colors,
                hoverinfo='text',
                hovertext=[f"Contribution: {v:.3f}<br>Importance: {p:.0%}" for v, p in zip(x_vals, pct_vals)],
                text=text_labels,
                textposition='inside',
                insidetextanchor='middle'
            ))
            fig.update_layout(
                margin=dict(l=160, r=24, t=12, b=60),
                xaxis_title="SHAP contribution to Swing+ (signed)",
                yaxis=dict(autorange="reversed"),
                height=420,
                showlegend=False,
                font=dict(size=11)
            )
            st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True, "displayModeBar": False})

    with col2:
        st.markdown(f"<div style='text-align:center;font-weight:700;color:#183153;'>Model baseline: {base_label}</div>", unsafe_allow_html=True)
        if shap_df is None or len(shap_df) == 0:
            st.write("No SHAP data to show.")
        else:
            display_df = shap_df.copy()
            display_df["feature_label"] = display_df["feature"].map(lambda x: FEATURE_LABELS.get(x, x))
            display_df = display_df.sort_values("abs_shap", ascending=False).head(12)
            display_df = display_df[["feature_label", "raw", "shap_value", "pct_of_abs"]].rename(columns={
                "feature_label": "Feature",
                "raw": "Value",
                "shap_value": "Contribution",
                "pct_of_abs": "PctImportance"
            })
            # Round 'Value' to 2 decimal points as requested
            display_df["Value"] = display_df["Value"].apply(lambda v: f"{v:.2f}" if pd.notna(v) else "NaN")
            display_df["Contribution"] = display_df["Contribution"].apply(lambda v: f"{v:.3f}")
            display_df["PctImportance"] = display_df["PctImportance"].apply(lambda v: f"{v:.0%}")
            display_df = display_df.reset_index(drop=True)
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ------------------ Mechanical similarity cluster (PLAYER-SPECIFIC) ------------------
    # This block MUST be inside the player tab. It was previously leaking into the Glossary tab.
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
                """
                <style>
                .sim-container {
                    width: 100%;
                    max-width: 1160px;
                    margin: 12px auto 10px auto;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                }
                .sim-list {
                    width: 100%;
                    display: flex;
                    flex-direction: column;
                    gap: 10px;
                    align-items: center;
                }
                .sim-item {
                    display: flex;
                    align-items: center;
                    background: #ffffff;
                    border-radius: 12px;
                    padding: 10px 14px;
                    gap: 12px;
                    width: 100%;
                    border: 1px solid #eef4f8;
                    box-shadow: 0 6px 18px rgba(15,23,42,0.04);
                }
                .sim-rank {
                    font-size: 1em;
                    font-weight: 700;
                    color: #183153;
                    min-width: 36px;
                    text-align: center;
                }
                .sim-headshot-compact {
                    height: 48px;
                    width: 48px;
                    border-radius: 8px;
                    object-fit: cover;
                    box-shadow: 0 1px 6px rgba(0,0,0,0.06);
                }
                .sim-name-compact {
                    flex: 1;
                    font-size: 1em;
                    color: #183153;
                }
                .sim-score-compact {
                    font-size: 0.98em;
                    font-weight: 700;
                    color: #333;
                    margin-right: 12px;
                    min-width: 72px;
                    text-align: right;
                }
                .sim-bar-mini {
                    width: 220px;
                    height: 10px;
                    background: #f4f7fa;
                    border-radius: 999px;
                    overflow: hidden;
                    margin-left: 8px;
                }
                .sim-bar-fill {
                    height: 100%;
                    border-radius: 999px;
                    transition: width 0.5s ease;
                }
                .sim-link {
                    color: inherit;
                    text-decoration: none;
                    font-weight: 700;
                }
                @media (max-width: 1100px) {
                    .sim-container { max-width: 92%; }
                    .sim-bar-mini { width: 160px; height: 8px; }
                    .sim-headshot-compact { height: 40px; width: 40px; }
                }
                </style>
                """,
                unsafe_allow_html=True
            )

            st.markdown(f'<div class="sim-container"><div class="sim-header" style="text-align:center;color:#183153;font-weight:700;margin-bottom:10px;">Top {TOP_N} mechanically similar players to <span style="font-weight:800;">{player_select}</span></div>', unsafe_allow_html=True)
            st.markdown('<div class="sim-list">', unsafe_allow_html=True)

            for idx, sim in enumerate(sim_rows, 1):
                pct = max(0.0, min(1.0, float(sim['score'])))
                width_pct = int(round(pct * 100))

                start_color = "#D32F2F"
                end_color = "#FFB648"

                sim_pct_text = f"{pct:.1%}"

                # Make the player name a clickable link that deep-links to the player page via query param.
                # The link points to the same page with ?player=Player+Name (URL-encoded) and opens in the same tab.
                player_link = f"?player={quote(sim['name'])}"

                st.markdown(
                    f"""
                    <div class="sim-item">
                        <div class="sim-rank">{idx}</div>
                        <img src="{sim['headshot_url']}" class="sim-headshot-compact" alt="headshot"/>
                        <div class="sim-name-compact"><a class="sim-link" href="{player_link}" target="_self" rel="noopener noreferrer">{sim['name']}</a></div>
                        <div class="sim-score-compact">{sim_pct_text}</div>
                        <div class="sim-bar-mini" aria-hidden="true">
                            <div class="sim-bar-fill" style="width:{width_pct}%; background: linear-gradient(90deg, {start_color}, {end_color});"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown('</div></div>', unsafe_allow_html=True)

            with st.expander("Show Detailed Heatmap"):
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

# ---------------- Glossary tab ----------------
with tab_glossary:
    glossary = {
        "Swing+": "A standardized measure of swing efficiency that evaluates how mechanically optimized a hitter's swing is compared to the league average. A score of 100 is average, while every 10 points is one standard deviation.",
        "ProjSwing+": "A projection-based version of Swing+ that combines current swing efficiency with physical power traits to estimate how a swing is likely to scale over time. It rewards hitters who show both efficient mechanics and physical attributes that suggest future growth.",
        "PowerIndex+": "A normalized measure of raw swing-driven power potential, built from metrics like bat speed, swing length, and attack angle. It represents how much force and lift a hitter's swing can generate relative to peers.",
        "xwOBA (Expected Weighted On-Base Average)": "An advanced Statcast metric estimating a hitter's overall offensive quality based on exit velocity and launch angle. It reflects what a player's outcomes should be given batted-ball quality.",
        "Predicted xwOBA": "A model-generated estimate of expected offensive production using a player's swing or biomechanical data (rather than batted-ball outcomes). It predicts what a player's xwOBA might look like given their swing profile.",
        "Avg Bat Speed": "The average velocity of the bat head at the point of contact, measured in miles per hour. Higher bat speed typically translates to higher exit velocity and more power potential.",
        "Avg Swing Length": "The average distance the bat travels from launch to contact. Longer swings can generate more leverage and power but may reduce contact consistency.",
        "Avg Attack Angle": "The vertical angle of the bat's path at contact, measured relative to the ground. Positive values indicate an upward swing plane; moderate positive angles (around 10–20°) often correlate with better launch angles for power.",
        "Avg Swing Tilt": "The overall body tilt or lateral bend during the swing. It reflects how the hitter's upper body moves through the swing plane, often influencing contact quality and pitch coverage.",
        "Avg Attack Direction": "The horizontal direction of the bat's movement at contact — whether the swing path moves toward right field (positive) or left field (negative). It captures how the swing path favors certain batted-ball directions.",
        "Avg Intercept Y vs. Plate": "The vertical position (height) at which the bat's swing plane crosses the plate area. It helps identify how 'flat' or 'steep' a hitter's swing path is through the strike zone.",
        "Avg Intercept Y vs. Batter": "The same intercept concept, but relative to the hitter's body position instead of the plate. It contextualizes swing height based on a hitter's individual setup and posture.",
        "Avg Batter Y Pos": "The average vertical position of the hitter's body (typically the torso or bat knob) at the moment of contact. It helps quantify a hitter's posture and body control through the swing.",
        "Avg Batter X Pos": "The average horizontal position of the bat or hands at contact, relative to the center of the plate. This reflects how far out in front or deep in the zone the hitter tends to make contact."
    }

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    st.markdown('<div style="max-width:1200px;margin:0 auto;padding:0 12px;">', unsafe_allow_html=True)

    # Search input
    q = st.text_input("Search terms...", value="", placeholder="Type to filter glossary (term or text)...")
    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    # Build DataFrame for filtering
    gloss_df = pd.DataFrame([{"term": k, "definition": v} for k, v in glossary.items()])
    if q and q.strip():
        qn = q.strip().lower()
        mask = gloss_df["term"].str.lower().str.contains(qn) | gloss_df["definition"].str.lower().str.contains(qn)
        filtered = gloss_df[mask].reset_index(drop=True)
    else:
        filtered = gloss_df.copy().reset_index(drop=True)

    # Use columns instead of custom HTML grid
    cols_per_row = 3
    rows = [filtered.iloc[i:i+cols_per_row] for i in range(0, len(filtered), cols_per_row)]
    
    for row_data in rows:
        cols = st.columns(cols_per_row, gap="large")
        for idx, (_, item) in enumerate(row_data.iterrows()):
            if idx < len(cols):
                with cols[idx]:
                    st.markdown(f"""
                    <div style="background: #fff; border-radius: 12px; padding: 18px; border: 1px solid #eef4f8; 
                                box-shadow: 0 6px 18px rgba(15,23,42,0.04); height: 220px; 
                                display: flex; flex-direction: column; justify-content: center;">
                        <div style="font-weight: 700; color: #0b1320; font-size: 1.03rem; margin-bottom: 12px;">
                            {item['term']}
                        </div>
                        <div style="color: #475569; font-size: 0.95rem; line-height: 1.45;">
                            {item['definition']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Add spacing between rows
        st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

    if filtered.shape[0] == 0:
        st.info("No matching terms found.")

    st.markdown("</div>", unsafe_allow_html=True)

# Inject JavaScript at the end of the page so it runs after Streamlit has rendered the tabs.
# This script:
# - When a link with ?player=... is clicked, it will update the location.search (same-tab navigation).
# - On page load it will check for ?player=... and click the "Player" tab after Streamlit renders the tab buttons.
components.html(
    """
    <script>
    (function(){
        try {
            // Intercept clicks on anchor links that include ?player=... so they navigate in the same tab without target=_blank behavior.
            document.addEventListener('click', function(e) {
                var el = e.target;
                // climb up until anchor or body
                while (el && el.tagName !== 'A' && el !== document.body) {
                    el = el.parentElement;
                }
                if (!el || el.tagName !== 'A') return;
                var href = el.getAttribute('href') || '';
                if (href.indexOf('?player=') !== -1) {
                    // allow Streamlit to handle same-tab navigation by setting location.search
                    e.preventDefault();
                    // replace preserves session history better than assign
                    window.location.href = href;
                }
            }, true);
        } catch (e) {
            // noop
        }

        try {
            const params = new URLSearchParams(window.location.search);
            const p = params.get('player');
            if (p) {
                // Use MutationObserver to wait for tabs to be fully rendered
                let tabClicked = false;
                const observer = new MutationObserver(() => {
                    if (tabClicked) return;
                    
                    const tabs = Array.from(document.querySelectorAll('[role="tab"]'));
                    if (tabs && tabs.length >= 3) {
                        // find the Player tab (should be second tab, index 1)
                        const playerTab = tabs.find(t => (t.innerText || t.textContent || '').trim().toLowerCase() === 'player');
                        if (playerTab && !playerTab.getAttribute('aria-selected')) {
                            // Only click if not already selected
                            playerTab.click();
                            tabClicked = true;
                            observer.disconnect();
                        } else if (playerTab && playerTab.getAttribute('aria-selected') === 'true') {
                            // Already on player tab, just disconnect observer
                            tabClicked = true;
                            observer.disconnect();
                        }
                    }
                });
                
                // Start observing
                observer.observe(document.body, {
                    childList: true,
                    subtree: true
                });
                
                // Fallback: disconnect after 5 seconds
                setTimeout(() => {
                    if (!tabClicked) {
                        observer.disconnect();
                    }
                }, 5000);
            }
        } catch (e) {
            // noop
        }
    })();
    </script>
    """,
    height=0,
)
