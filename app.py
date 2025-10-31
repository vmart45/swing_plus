import pandas as pd
import streamlit as st
import os
from PIL import Image
import requests
from io import BytesIO

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
extra_cols = ["avg_bat_speed", "swing_length", "attack_angle", "swing_tilt", "attack_direction"]
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
        "avg_bat_speed", "swing_length", "attack_angle", "swing_tilt", "attack_direction"
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

player_select = st.selectbox("Select a Player", sorted(df_filtered["Name"].unique()))
player_row = df[df["Name"] == player_select].iloc[0]

st.markdown(
    f"""
    <div style="display:flex; align-items:center; justify-content:center; gap:28px; margin-bottom:12px;">
        <div style="flex-shrink:0; display:flex; align-items:center;">
    """,
    unsafe_allow_html=True
)

headshot_width = 52  # px, just slightly larger than the logo
logo_width = 40      # logo size for reference

if "id" in player_row and pd.notnull(player_row["id"]):
    player_id = str(int(player_row["id"]))
    headshot_url = f"https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/w_640,q_auto:best/v1/people/{player_id}/headshot/silo/current.png"
    try:
        response = requests.get(headshot_url, timeout=5)
        if response.status_code == 200:
            headshot_img = Image.open(BytesIO(response.content))
            st.image(headshot_img, width=headshot_width, caption="", use_column_width=False)
        else:
            st.image("https://img.mlbstatic.com/mlb-photos/image/upload/v1/people/0/headshot/silo/current.png", width=headshot_width, use_column_width=False)
    except Exception:
        st.image("https://img.mlbstatic.com/mlb-photos/image/upload/v1/people/0/headshot/silo/current.png", width=headshot_width, use_column_width=False)
else:
    st.image("https://img.mlbstatic.com/mlb-photos/image/upload/v1/people/0/headshot/silo/current.png", width=headshot_width, use_column_width=False)

st.markdown(
    f"""
        </div>
        <div style="flex-shrink:0;">
    """,
    unsafe_allow_html=True
)

team_abb = player_row["Team"] if "Team" in player_row and pd.notnull(player_row["Team"]) else ""
logo_url = image_dict.get(team_abb, "")
if logo_url:
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:14px;">
            <span style="font-size:1.7em; font-weight:700; color:#183153; letter-spacing:0.01em;">{player_select}</span>
            <img src="{logo_url}" style="height:{logo_width}px; vertical-align:middle; border-radius:7px; background:#eee;"/>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        f"""
        <div style="font-size:1.7em; font-weight:700; color:#183153; letter-spacing:0.01em;">{player_select}</div>
        """,
        unsafe_allow_html=True
    )

st.markdown(
    """
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

total_players = len(df)
df["Swing+_rank"] = df["Swing+"].rank(ascending=False, method="min").astype(int)
df["ProjSwing+_rank"] = df["ProjSwing+"].rank(ascending=False, method="min").astype(int)
df["PowerIndex+_rank"] = df["PowerIndex+"].rank(ascending=False, method="min").astype(int)

p_swing_rank = df.loc[df["Name"] == player_select, "Swing+_rank"].iloc[0]
p_proj_rank = df.loc[df["Name"] == player_select, "ProjSwing+_rank"].iloc[0]
p_power_rank = df.loc[df["Name"] == player_select, "PowerIndex+_rank"].iloc[0]

st.markdown(
    f"""
    <div style="display: flex; justify-content: center; gap: 32px; margin-top: 20px; margin-bottom: 28px;">
      <div style="background: #fff; border-radius: 16px; box-shadow: 0 2px 12px #0001; padding: 24px 32px; text-align: center; min-width: 160px;">
        <div style="font-size: 2.2em; font-weight: 700; color: #C62828;">{player_row['Swing+']:.1f}</div>
        <div style="font-size: 1.1em; color: #888; font-weight: 600; letter-spacing: 0.5px; margin-bottom: 4px;">Swing+</div>
        <span style="background: #FFC10733; color: #B71C1C; border-radius: 10px; font-size: 0.98em; padding: 2px 10px 2px 10px;">Rank {p_swing_rank} of {total_players}</span>
      </div>
      <div style="background: #fff; border-radius: 16px; box-shadow: 0 2px 12px #0001; padding: 24px 32px; text-align: center; min-width: 160px;">
        <div style="font-size: 2.2em; font-weight: 700; color: #2E7D32;">{player_row['ProjSwing+']:.1f}</div>
        <div style="font-size: 1.1em; color: #888; font-weight: 600; letter-spacing: 0.5px; margin-bottom: 4px;">ProjSwing+</div>
        <span style="background: #C8E6C933; color: #1B5E20; border-radius: 10px; font-size: 0.98em; padding: 2px 10px 2px 10px;">Rank {p_proj_rank} of {total_players}</span>
      </div>
      <div style="background: #fff; border-radius: 16px; box-shadow: 0 2px 12px #0001; padding: 24px 32px; text-align: center; min-width: 160px;">
        <div style="font-size: 2.2em; font-weight: 700; color: #1565C0;">{player_row['PowerIndex+']:.1f}</div>
        <div style="font-size: 1.1em; color: #888; font-weight: 600; letter-spacing: 0.5px; margin-bottom: 4px;">PowerIndex+</div>
        <span style="background: #B3E5FC33; color: #01579B; border-radius: 10px; font-size: 0.98em; padding: 2px 10px 2px 10px;">Rank {p_power_rank} of {total_players}</span>
      </div>
    </div>
    """,
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
