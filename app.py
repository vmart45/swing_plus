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

# FEATURE_LABELS at module level for consistent mapping
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

# Define mechanical_features early to avoid NameError and to reuse across tabs
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

@st.cache_data
def compute_shap(player_row, mech_features_available):
    # returns (shap_series, shap_pred, shap_base) or (None, None, None)
    if not model_loaded or explainer is None:
        return None, None, None
    try:
        X_player = prepare_model_input_for_player(player_row, mech_features_available, model, df_reference=df)
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
        shap_df = pd.Series(shap_values_arr, index=X_player.columns)
        return shap_df, shap_pred, shap_base
    except Exception:
        return None, None, None

@st.cache_data
def get_scaler_and_scaled_df(features):
    scaler = StandardScaler()
    X = df[features].astype(float)
    X_scaled = scaler.fit_transform(X)
    df_scaled = pd.DataFrame(X_scaled, columns=features, index=df.index)
    return scaler, df_scaled

def compute_cosine_similarity_between_rows(vecA, vecB):
    sim = np.dot(vecA, vecB) / (np.linalg.norm(vecA) * np.linalg.norm(vecB) + 1e-12)
    return float(sim)

# Create top-level tabs via page radio so query param navigation works
params = st.experimental_get_query_params()
qp_player = None
qp_player_b = None
qp_page = None

if "player" in params and len(params["player"]) > 0:
    try:
        qp_player = unquote(params["player"][0])
    except Exception:
        qp_player = params["player"][0]

if "playerA" in params and len(params["playerA"]) > 0:
    try:
        qp_player = unquote(params["playerA"][0])
    except Exception:
        qp_player = params["playerA"][0]

if "playerB" in params and len(params["playerB"]) > 0:
    try:
        qp_player_b = unquote(params["playerB"][0])
    except Exception:
        qp_player_b = params["playerB"][0]

if "page" in params and len(params["page"]) > 0:
    try:
        qp_page = unquote(params["page"][0])
    except Exception:
        qp_page = params["page"][0]

page_options = ["Main", "Player", "Compare", "Glossary"]
default_page = 0
if qp_page and qp_page in page_options:
    default_page = page_options.index(qp_page)
elif qp_player:
    default_page = page_options.index("Player")
elif qp_player_b:
    default_page = page_options.index("Compare")

st.markdown("<div style='display:flex;justify-content:center;margin-bottom:6px;'>", unsafe_allow_html=True)
page = st.radio("", page_options, index=default_page, horizontal=True, key="top_nav")
st.markdown("</div>", unsafe_allow_html=True)

def open_compare_in_same_tab(playerA, playerB):
    try:
        st.experimental_set_query_params(playerA=playerA, playerB=playerB, page="Compare")
    except Exception:
        try:
            st.experimental_set_query_params(player=playerA, playerB=playerB, page="Compare")
        except Exception:
            pass

# ---------------- Main tab: Metrics table and leaderboards ----------------
if page == "Main":
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
        # format numbers to 2 decimals where relevant
        top_swing_display = top_swing.copy()
        for col in ["Swing+", "ProjSwing+", "PowerIndex+"]:
            if col in top_swing_display.columns:
                top_swing_display[col] = top_swing_display[col].apply(lambda v: f"{v:.2f}" if pd.notna(v) else v)
        st.dataframe(
            top_swing_display[leaderboard_cols]
            .style.background_gradient(subset=["Swing+"], cmap=elite_cmap),
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
        top_proj_display = top_proj.copy()
        for col in ["Swing+", "ProjSwing+", "PowerIndex+"]:
            if col in top_proj_display.columns:
                top_proj_display[col] = top_proj_display[col].apply(lambda v: f"{v:.2f}" if pd.notna(v) else v)
        st.dataframe(
            top_proj_display[leaderboard_cols]
            .style.background_gradient(subset=["ProjSwing+"], cmap=elite_cmap),
            use_container_width=True,
            hide_index=True
        )

# ---------------- Player tab: Player Detail view ----------------
elif page == "Player":
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

    player_name_html = f'<span style="font-size:2.3em;font-weight:800;color:#183153;letter-spacing:0.01em;vertical-align:middle;margin:0 20px;">{player_select}</span>'

    # Add team logo back to player page (to the right of the name) when available
    team_logo_html = ""
    if "Team" in player_row and pd.notnull(player_row["Team"]):
        team_abbr = str(player_row["Team"]).strip()
        team_logo_url = image_dict.get(team_abbr, "")
        if team_logo_url:
            team_logo_html = f'<div style="margin-left:14px; display:flex; align-items:center;"><img src="{team_logo_url}" style="height:{logo_size}px;width:{logo_size}px;border-radius:8px;box-shadow:none;background:transparent;border:none;object-fit:contain;" alt="team logo"/></div>'

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
            {team_logo_html}
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
            <div style="font-size: 2.2em; font-weight: 700; color: {swing_color};">{player_row['Swing+']:.2f}</div>
            <div style="font-size: 1.1em; color: #888; font-weight: 600; letter-spacing: 0.5px; margin-bottom: 4px;">Swing+</div>
            <span style="background: #FFC10733; color: #B71C1C; border-radius: 10px; font-size: 0.98em; padding: 2px 10px 2px 10px;">Rank {p_swing_rank} of {total_players}</span>
          </div>
          <div style="background: #fff; border-radius: 16px; box-shadow: 0 2px 12px #0001; padding: 24px 32px; text-align: center; min-width: 160px;">
            <div style="font-size: 2.2em; font-weight: 700; color: {proj_color};">{player_row['ProjSwing+']:.2f}</div>
            <div style="font-size: 1.1em; color: #888; font-weight: 600; letter-spacing: 0.5px; margin-bottom: 4px;">ProjSwing+</div>
            <span style="background: #C8E6C933; color: #1B5E20; border-radius: 10px; font-size: 0.98em; padding: 2px 10px 2px 10px;">Rank {p_proj_rank} of {total_players}</span>
          </div>
          <div style="background: #fff; border-radius: 16px; box-shadow: 0 2px 12px #0001; padding: 24px 32px; text-align: center; min-width: 160px;">
            <div style="font-size: 2.2em; font-weight: 700; color: {power_color};">{player_row['PowerIndex+']:.2f}</div>
            <div style="font-size: 1.1em; color: #888; font-weight: 600; letter-spacing: 0.5px; margin-bottom: 4px;">PowerIndex+</div>
            <span style="background: #B3E5FC33; color: #01579B; border-radius: 10px; font-size: 0.98em; padding: 2px 10px 2px 10px;">Rank {p_power_rank} of {total_players}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    video_url = None
    if "id" in player_row and pd.notnull(player_row["id"]):
        player_id = str(int(player_row["id"]))
        video_url = f"https://builds.mlbstatic.com/baseballsavant.mlb.com/swing-path/splendid-splinter/cut/{player_id}-2025-{bat_side}.mp4"

    DEFAULT_ONEIL_CRUZ_IDS = ['665833-2025-L', '665833-2025-R', '665833-2025-S']
    default_name = "Oneil Cruz"
    showing_default = False
    if video_url:
        showing_default = f'{player_id}-2025-{bat_side}' in DEFAULT_ONEIL_CRUZ_IDS

    if video_url:
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

    mech_features_available = [f for f in mechanical_features if f in df.columns]

    # Compute SHAP values for the selected player (Swing+ only)
    shap_df = None
    shap_base = None
    shap_pred = None
    shap_values_arr = None

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

            # Show the top-N similar players list (unchanged) and provide an expander to toggle the cluster heatmap (default: collapsed)
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
                /* Compare button: simple white background with black text and subtle border */
                .sim-compare-btn {
                    background: #ffffff;
                    color: #000000;
                    padding: 8px 12px;
                    border-radius: 10px;
                    text-decoration: none;
                    font-weight: 800;
                    border: 1px solid #d1d5db;
                    cursor: pointer;
                }
                .sim-compare-btn:hover { background: #f3f4f6; transform: translateY(-1px); }
                @media (max-width: 1100px) {
                    .sim-container { max-width: 92%; }
                    .sim-bar-mini { width: 160px; height: 8px; }
                    .sim-headshot-compact { height: 40px; width: 40px; }
                }
                </style>
                """,
                unsafe_allow_html=True
            )

            st.markdown(f'<div class="sim-container"><div class="sim-header" style="text-align:center;color:#183153;font-weight:700;margin-bottom:10px;">Top {TOP_N} mechanically similar players to {player_select}</div>', unsafe_allow_html=True)
            st.markdown('<div class="sim-list">', unsafe_allow_html=True)

            for idx, sim in enumerate(sim_rows, 1):
                pct = max(0.0, min(1.0, float(sim['score'])))
                width_pct = int(round(pct * 100))

                start_color = "#D32F2F"
                end_color = "#FFB648"

                sim_pct_text = f"{pct:.1%}"

                href = f"?playerA={quote(player_select)}&playerB={quote(sim['name'])}&page=Compare"

                st.markdown(
                    f"""
                    <div class="sim-item">
                        <div class="sim-rank">{idx}</div>
                        <img src="{sim['headshot_url']}" class="sim-headshot-compact" alt="headshot"/>
                        <div class="sim-name-compact"><a href="?player={quote(sim['name'])}" style="color:#183153;text-decoration:none;font-weight:700;">{sim['name']}</a></div>
                        <div style="display:flex;align-items:center;gap:8px;">
                            <div class="sim-score-compact">{sim_pct_text}</div>
                            <div class="sim-bar-mini" aria-hidden="true">
                                <div class="sim-bar-fill" style="width:{width_pct}%; background: linear-gradient(90deg, {start_color}, {end_color});"></div>
                            </div>
                            <a class="sim-compare-btn" href="{href}" onclick="window.history.pushState(null,'','{href}'); setTimeout(()=>window.location.reload(),30); return false;">Compare</a>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown('</div></div>', unsafe_allow_html=True)

            # Use an expander (click-to-open) for the cluster heatmap so it stays out of the way until requested
            with st.expander("Mechanical similarity cluster (click to expand)", expanded=False):
                try:
                    heat_names = [player_select] + list(similar_players.index)
                    # build indices within df_mech to extract corresponding rows from similarity_matrix
                    heat_idx = [df_mech[df_mech[name_col] == n].index[0] for n in heat_names]
                    heat_mat = similarity_matrix[np.ix_(heat_idx, heat_idx)]

                    # Plot heatmap with annotations and colorbar
                    fig_h, axh = plt.subplots(figsize=(8, 6))
                    sns.heatmap(
                        heat_mat,
                        xticklabels=heat_names,
                        yticklabels=heat_names,
                        cmap="RdYlBu_r",
                        vmin=0.0,
                        vmax=1.0,
                        annot=True,
                        fmt=".2f",
                        annot_kws={"fontsize": 9},
                        square=True,
                        cbar_kws={"shrink": 0.6, "label": "Cosine Similarity"},
                        ax=axh
                    )
                    axh.set_title(f"Mechanical Similarity Cluster: {player_select}", fontsize=16, pad=12)
                    axh.set_xlabel("Name")
                    axh.set_ylabel("Name")
                    plt.tight_layout()
                    st.pyplot(fig_h)
                except Exception:
                    st.info("Could not render cluster heatmap due to data issues.")

# ---------------- Compare tab ----------------
elif page == "Compare":
    st.markdown(
        """
        <h2 style="text-align:center; margin-top:10px; margin-bottom:6px; font-size:1.4em; color:#183153;">
            Compare Players
        </h2>
        """,
        unsafe_allow_html=True
    )

    player_options = sorted(df_filtered["Name"].unique())
    if not player_options:
        st.info("No players available for comparison with current filters.")
    else:
        default_a_idx = 0
        default_b_idx = 1 if len(player_options) > 1 else 0
        if qp_player and qp_player in player_options:
            default_a_idx = player_options.index(qp_player)
        if qp_player_b and qp_player_b in player_options:
            default_b_idx = player_options.index(qp_player_b)
        col_a, col_b = st.columns([1, 1])
        with col_a:
            playerA = st.selectbox("Player A", player_options, index=default_a_idx, key="compare_player_a")
        with col_b:
            playerB = st.selectbox("Player B", player_options, index=default_b_idx, key="compare_player_b")

        # Removed the non-functional "Open comparison (update URL)" button per request.

        if playerA == playerB:
            st.warning("Select two different players to compare.")
        else:
            rowA = df[df["Name"] == playerA].iloc[0]
            rowB = df[df["Name"] == playerB].iloc[0]

            mech_features_available = [f for f in mechanical_features if f in df.columns]
            if len(mech_features_available) >= 2:
                scaler, df_scaled = get_scaler_and_scaled_df(mech_features_available)
                try:
                    idxA = df[df["Name"] == playerA].index[0]
                    idxB = df[df["Name"] == playerB].index[0]
                    vecA = df_scaled.loc[idxA, mech_features_available].values
                    vecB = df_scaled.loc[idxB, mech_features_available].values
                except Exception:
                    vecA = scaler.transform(df[df["Name"] == playerA][mech_features_available].astype(float))[0]
                    vecB = scaler.transform(df[df["Name"] == playerB][mech_features_available].astype(float))[0]
                cosine_sim = compute_cosine_similarity_between_rows(vecA, vecB)
                sim_pct = f"{cosine_sim*100:.1f}%"
            else:
                cosine_sim = None
                sim_pct = "N/A"

            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                teamA = rowA["Team"] if "Team" in rowA and pd.notnull(rowA["Team"]) else ""
                logoA = image_dict.get(teamA, "")
                imgA = ""
                if "id" in rowA and pd.notnull(rowA["id"]):
                    pid = str(int(rowA["id"]))
                    imgA = f"https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/w_640,q_auto:best/v1/people/{pid}/headshot/silo/current.png"
                else:
                    imgA = "https://img.mlbstatic.com/mlb-photos/image/upload/v1/people/0/headshot/silo/current.png"
                # team logo on compare page — remove white border / box and show inline
                logo_html_a = f'<div style="margin-top:8px;"><img src="{logoA}" style="height:40px;width:40px;border-radius:6px;box-shadow:none;background:transparent;border:none;"></div>' if logoA else ""
                st.markdown(f'<div style="text-align:center;"><img src="{imgA}" style="height:84px;width:84px;border-radius:12px;"><div style="font-weight:800;margin-top:6px;color:#183153;">{playerA}</div>{logo_html_a}</div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div style="text-align:center;padding:8px;border-radius:10px;"><div style="font-size:1.25em;font-weight:800;color:#0b6efd;">Similarity</div><div style="font-size:1.6em;font-weight:800;color:#183153;margin-top:6px;">{sim_pct}</div></div>', unsafe_allow_html=True)
            with col3:
                teamB = rowB["Team"] if "Team" in rowB and pd.notnull(rowB["Team"]) else ""
                logoB = image_dict.get(teamB, "")
                imgB = ""
                if "id" in rowB and pd.notnull(rowB["id"]):
                    pid = str(int(rowB["id"]))
                    imgB = f"https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/w_640,q_auto:best/v1/people/{pid}/headshot/silo/current.png"
                else:
                    imgB = "https://img.mlbstatic.com/mlb-photos/image/upload/v1/people/0/headshot/silo/current.png"
                logo_html_b = f'<div style="margin-top:8px;"><img src="{logoB}" style="height:40px;width:40px;border-radius:6px;box-shadow:none;background:transparent;border:none;"></div>' if logoB else ""
                st.markdown(f'<div style="text-align:center;"><img src="{imgB}" style="height:84px;width:84px;border-radius:12px;"><div style="font-weight:800;margin-top:6px;color:#183153;">{playerB}</div>{logo_html_b}</div>', unsafe_allow_html=True)

            st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

            stats = ["Age", "Swing+", "ProjSwing+", "PowerIndex+"]
            cols_stats = st.columns(len(stats)*2)
            for i, stat in enumerate(stats):
                valA = rowA.get(stat, "N/A")
                valA_disp = f"{valA:.2f}" if isinstance(valA, (int, float, np.floating, np.integer)) and not pd.isna(valA) else valA
                cols_stats[i].markdown(f'<div style="text-align:center;"><div style="font-weight:700;color:#183153;">{valA_disp}</div><div style="color:#64748b;">{stat} (A)</div></div>', unsafe_allow_html=True)
            for i, stat in enumerate(stats):
                valB = rowB.get(stat, "N/A")
                valB_disp = f"{valB:.2f}" if isinstance(valB, (int, float, np.floating, np.integer)) and not pd.isna(valB) else valB
                cols_stats[i+len(stats)].markdown(f'<div style="text-align:center;"><div style="font-weight:700;color:#183153;">{valB_disp}</div><div style="color:#64748b;">{stat} (B)</div></div>', unsafe_allow_html=True)

            st.markdown("<hr />", unsafe_allow_html=True)

            if len(mech_features_available) >= 2:
                feats = mech_features_available
                mean_series = df[feats].mean()
                std_series = df[feats].std().replace(0, 1e-9)
                valsA = rowA[feats].astype(float)
                valsB = rowB[feats].astype(float)
                zA = (valsA - mean_series) / std_series
                zB = (valsB - mean_series) / std_series
                abs_diff = (valsA - valsB).abs()
                z_diff = (zA - zB).abs()
                pctile = df[feats].rank(pct=True)
                pctA = pctile.loc[rowA.name]
                pctB = pctile.loc[rowB.name]

                shapA, predA, baseA = compute_shap(rowA, feats)
                shapB, predB, baseB = compute_shap(rowB, feats)

                if model_loaded and explainer is not None and shapA is not None:
                    try:
                        sampleX = df[feats].head(200).copy()
                        sampleX = sampleX.fillna(sampleX.mean())
                        try:
                            samp_shap = explainer(sampleX)
                            if hasattr(samp_shap, "values"):
                                mean_abs_shap = np.mean(np.abs(samp_shap.values), axis=0)
                                importance = pd.Series(mean_abs_shap, index=feats)
                            else:
                                importance = pd.Series(np.ones(len(feats)), index=feats)
                        except Exception:
                            importance = pd.Series(np.ones(len(feats)), index=feats)
                    except Exception:
                        importance = pd.Series(np.ones(len(feats)), index=feats)
                else:
                    importance = pd.Series(np.ones(len(feats)), index=feats)

                # Friendly, less-technical headings
                st.markdown("### Quick Takeaways")
                weighted_score = (1 - (z_diff / (z_diff.max() + 1e-9))).clip(0, 1) * importance
                top_sim_idxs = weighted_score.sort_values(ascending=False).head(3).index.tolist()
                top_diff_idxs = (z_diff * importance).sort_values(ascending=False).head(3).index.tolist()
                bullets = []
                if cosine_sim is not None:
                    bullets.append(f"Overall mechanical similarity: {cosine_sim*100:.1f}%.")
                for f in top_sim_idxs:
                    bullets.append(f"Similarity driver: {FEATURE_LABELS.get(f,f)} — both players are close in normalized space.")
                for f in top_diff_idxs:
                    bullets.append(f"Difference driver: {FEATURE_LABELS.get(f,f)} — notable normalized difference; check distribution.")
                for b in bullets:
                    st.markdown(f"- {b}")

                st.markdown("### Feature comparison")
                table_df = pd.DataFrame({
                    "Feature": [FEATURE_LABELS.get(f, f) for f in feats],
                    "A (raw)": [f"{valsA[f]:.2f}" if pd.notna(valsA[f]) else "NaN" for f in feats],
                    "B (raw)": [f"{valsB[f]:.2f}" if pd.notna(valsB[f]) else "NaN" for f in feats],
                    "Raw diff": [f"{(valsA[f]-valsB[f]):.2f}" for f in feats],
                    "Z diff": [f"{z_diff[f]:.2f}" for f in feats],
                    "Pct A": [f"{pctA[f]:.0%}" for f in feats],
                    "Pct B": [f"{pctB[f]:.0%}" for f in feats],
                    "Importance": [f"{importance[f]:.3f}" for f in feats]
                })
                st.dataframe(table_df.style.format(precision=2), use_container_width=True, hide_index=True)

                st.markdown("### Model contributions")
                if shapA is None or shapB is None:
                    st.info("Model SHAP not available for one or both players.")
                else:
                    # Use same visual style as Player page: two single-player horizontal SHAP bar charts (one per column)
                    order = importance.sort_values(ascending=False).index.tolist()
                    labels = [FEATURE_LABELS.get(f, f) for f in order]
                    shapA_ord = shapA[order]
                    shapB_ord = shapB[order]

                    col_shap_a, col_shap_b = st.columns([1, 1])
                    with col_shap_a:
                        vals = shapA_ord.values.astype(float)
                        colors = ["#D8573C" if v > 0 else "#3B82C4" for v in vals]
                        text_labels = [f"{v:.3f}" for v in vals]
                        figA = go.Figure()
                        figA.add_trace(go.Bar(
                            x=vals,
                            y=labels,
                            orientation='h',
                            marker_color=colors,
                            hoverinfo='text',
                            hovertext=[f"Contribution: {v:.3f}" for v in vals],
                            text=text_labels,
                            textposition='inside',
                            insidetextanchor='middle'
                        ))
                        figA.update_layout(
                            margin=dict(l=160, r=24, t=28, b=60),
                            xaxis_title="SHAP contribution to Swing+ (signed)",
                            yaxis=dict(autorange="reversed"),
                            height=420,
                            showlegend=False,
                            title=dict(text=f"{playerA} — Model contribution", x=0.01, xanchor='left'),
                            font=dict(size=11)
                        )
                        st.plotly_chart(figA, use_container_width=True, config={"displayModeBar": False})

                    with col_shap_b:
                        vals = shapB_ord.values.astype(float)
                        colors = ["#F59E0B" if v > 0 else "#60A5FA" for v in vals]
                        text_labels = [f"{v:.3f}" for v in vals]
                        figB = go.Figure()
                        figB.add_trace(go.Bar(
                            x=vals,
                            y=labels,
                            orientation='h',
                            marker_color=colors,
                            hoverinfo='text',
                            hovertext=[f"Contribution: {v:.3f}" for v in vals],
                            text=text_labels,
                            textposition='inside',
                            insidetextanchor='middle'
                        ))
                        figB.update_layout(
                            margin=dict(l=160, r=24, t=28, b=60),
                            xaxis_title="SHAP contribution to Swing+ (signed)",
                            yaxis=dict(autorange="reversed"),
                            height=420,
                            showlegend=False,
                            title=dict(text=f"{playerB} — Model contribution", x=0.01, xanchor='left'),
                            font=dict(size=11)
                        )
                        st.plotly_chart(figB, use_container_width=True, config={"displayModeBar": False})

                st.markdown("### Percentiles (radar)")
                try:
                    pctA_vals = pctA[feats].values
                    pctB_vals = pctB[feats].values
                    labels_radar = [FEATURE_LABELS.get(f, f) for f in feats]
                    fig_r = go.Figure()
                    # Change color of one of the radar plots to orange to avoid blue-on-blue
                    fig_r.add_trace(go.Scatterpolar(r=pctA_vals, theta=labels_radar, fill='toself', name=playerA, marker_color="#FF7A1A", fillcolor="rgba(255,122,26,0.25)"))
                    fig_r.add_trace(go.Scatterpolar(r=pctB_vals, theta=labels_radar, fill='toself', name=playerB, marker_color="#0b6efd", fillcolor="rgba(11,110,253,0.15)"))
                    fig_r.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), showlegend=True, height=450)
                    st.plotly_chart(fig_r, use_container_width=True, config={"displayModeBar": False})
                except Exception:
                    st.info("Radar chart not available due to data issues.")

                st.markdown("### Feature distribution")
                # Fix names in dropdown to more readable text using FEATURE_LABELS
                sel_feat_map = {FEATURE_LABELS.get(f, f): f for f in feats}
                # Show a visible dropdown (selectbox) on the page to pick the feature for distribution
                sel_feat_label = st.selectbox("Choose feature for distribution", list(sel_feat_map.keys()), index=0, key="dist_select")
                sel_feat = sel_feat_map.get(sel_feat_label, feats[0])

                # Render the distribution plot at a clear on-screen size (use_container_width=True so it fills available width)
                fig2, ax2 = plt.subplots(figsize=(8, 3))
                try:
                    sns.kdeplot(df[sel_feat].dropna(), fill=True, ax=ax2, color="#93c5fd")
                    ax2.axvline(valsA[sel_feat], color="#FF7A1A", linestyle="--", label=f"{playerA}")
                    ax2.axvline(valsB[sel_feat], color="#ef4444", linestyle="--", label=f"{playerB}")
                    ax2.legend(fontsize=9)
                    ax2.set_xlabel(FEATURE_LABELS.get(sel_feat, sel_feat))
                    ax2.tick_params(axis='both', which='major', labelsize=10)
                    plt.tight_layout()
                    st.pyplot(fig2, use_container_width=True)
                except Exception:
                    st.info("Distribution plot not available for this feature.")

# ---------------- Glossary tab ----------------
else:
    glossary = {
        "Swing+": "A standardized measure of swing efficiency that evaluates how mechanically optimized a hitter’s swing is compared to the league average. A score of 100 is average, while every 10 points above or below represents roughly one standard deviation. Higher values indicate more efficient, well-sequenced swings.",
        "ProjSwing+": "A projection-based version of Swing+ that combines current swing efficiency with physical power traits to estimate how a swing is likely to scale over time. It rewards hitters whose mechanical foundation and power potential suggest strong long-term growth.",
        "PowerIndex+": "A normalized measure of raw swing-driven power potential, built from metrics like bat speed, swing length, and attack angle. It represents how much force and lift a hitter’s swing can naturally generate, independent of game results.",
        "xwOBA (Expected Weighted On-Base Average)": "An advanced Statcast metric estimating a hitter’s overall offensive quality based on exit velocity and launch angle. It reflects what a player’s on-base performance should be, given contact quality, rather than what actually happened.",
        "Predicted xwOBA": "A model-generated estimate of expected offensive production using a player’s swing or biomechanical data (rather than batted-ball outcomes). It predicts what a player’s xwOBA would be based on their swing traits alone.",
        "Avg Bat Speed": "The average velocity of the bat head at the point of contact, measured in miles per hour. Higher bat speed typically translates to higher exit velocity and more power potential.",
        "Avg Swing Length": "The average distance the bat travels from launch to contact. Longer swings can generate more leverage and power but may reduce contact consistency.",
        "Avg Attack Angle": "The vertical angle of the bat’s path at contact, measured relative to the ground. Positive values indicate an upward swing plane; moderate positive angles (around 10–20°) are generally optimal for line drives and power.",
        "Avg Swing Tilt": "The overall body tilt or lateral bend during the swing. It reflects how the hitter’s upper body moves through the swing plane, often influencing contact quality and pitch coverage.",
        "Avg Attack Direction": "The horizontal direction of the bat’s movement at contact — whether the swing path moves toward right field (positive) or left field (negative). It captures how the hitter matches their bat path to pitch location.",
        "Avg Intercept Y vs. Plate": "The vertical position (height) at which the bat’s swing plane crosses the plate area. It helps identify how “flat” or “steep” a hitter’s swing path is through the hitting zone.",
        "Avg Intercept Y vs. Batter": "The same intercept concept, but relative to the hitter’s body position instead of the plate. It contextualizes swing height based on a hitter’s individual setup and stance.",
        "Avg Batter Y Pos": "The average vertical position of the hitter’s body (typically the torso or bat knob) at the moment of contact. It helps quantify a hitter’s posture and body control through the swing.",
        "Avg X Pos": "The average horizontal position of the bat or hands at contact, relative to the center of the plate. This reflects how far out in front or deep in the zone the hitter tends to make contact."
    }

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    st.markdown('<div style="max-width:1200px;margin:0 auto;padding:0 12px;">', unsafe_allow_html=True)

    q = st.text_input("Search terms...", value="", placeholder="Type to filter glossary (term or text)...")
    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    gloss_df = pd.DataFrame([{"term": k, "definition": v} for k, v in glossary.items()])
    if q and q.strip():
        qn = q.strip().lower()
        mask = gloss_df["term"].str.lower().str.contains(qn) | gloss_df["definition"].str.lower().str.contains(qn)
        filtered = gloss_df[mask].reset_index(drop=True)
    else:
        filtered = gloss_df.copy().reset_index(drop=True)

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
        st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

# Inject JS helper to ensure Compare links do same-tab navigation
components.html(
    """
    <script>
    (function() {
        document.addEventListener('click', function(e) {
            try {
                var el = e.target;
                while (el && el.tagName !== 'A' && el !== document.body) el = el.parentElement;
                if (!el || el.tagName !== 'A') return;
                var href = el.getAttribute('href') || '';
                if (href.indexOf('?playerA=') !== -1 || href.indexOf('?player=') !== -1) {
                    e.preventDefault();
                    var newUrl = href.startsWith('?') ? window.location.pathname + href : href;
                    try { history.pushState(null, '', newUrl); } catch (err) {}
                    setTimeout(function(){ window.location.reload(); }, 40);
                }
            } catch (err) {}
        }, true);
    })();
    """,
    height=0
)
