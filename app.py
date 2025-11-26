import pandas as pd
import streamlit as st
import os
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import joblib
import shap
import plotly.graph_objects as go
import streamlit.components.v1 as components
from urllib.parse import quote, unquote
import matplotlib.colors as mcolors
import json
import html

st.set_page_config(
    page_title="Swing+ Dashboard",
    page_icon="icon.PNG",
    layout="wide"
)

DATA_PATH = "Main.csv"
MODEL_PATH = "SwingPlus.pkl"

def create_centered_cmap(center=100, vmin=70, vmax=130):
    """
    Create a diverging colormap centered at a specific value (default 100).
    Below center = blue, at center = white, above center = red
    """
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    center_normalized = norm(center)

    colors_below = plt.cm.Blues_r(np.linspace(0.4, 0.95, max(2, int(center_normalized * 256))))
    colors_center = np.array([[1, 1, 1, 1]])  # white at center
    colors_above = plt.cm.Reds(np.linspace(0.05, 0.95, max(2, int((1 - center_normalized) * 256))))

    all_colors = np.vstack((colors_below, colors_center, colors_above))
    custom_cmap = mcolors.LinearSegmentedColormap.from_list('centered_coolwarm', all_colors)

    return custom_cmap

if not os.path.exists(DATA_PATH):
    st.error(f"Could not find `{DATA_PATH}` in the app directory.")
    st.stop()

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

df = load_data(DATA_PATH)

def normalize_columns(df):
    lower_cols = {c.lower(): c for c in df.columns}
    def find(*variants):
        for v in variants:
            if v is None:
                continue
            lc = v.lower()
            if lc in lower_cols:
                return lower_cols[lc]
        return None

    mappings = {
        "Name": find("Name", "name", "player", "player_name"),
        "id": find("id", "playerid", "player_id"),
        "Team": find("Team", "team"),
        "Age": find("Age", "age", "currentage", "current_age"),
        "Swing+": find("Swing+", "swingplus", "swing_plus", "swing_plus"),
        "HitSkillPlus": find("HitSkillPlus", "hitskillplus", "hit_skill_plus", "proj_swing_plus", "projswingplus"),
        "ImpactPlus": find("ImpactPlus", "impactplus", "impact_plus", "powerindex+", "powerindexplus", "powerindex"),
        "avg_bat_speed": find("avg_bat_speed", "avg_batspeed", "avg_bat_speed_mph", "bat_speed"),
        "swing_length": find("swing_length", "avg_swing_length", "swinglength"),
        "attack_angle": find("attack_angle", "avg_attack_angle", "attackangle"),
        "swing_tilt": find("swing_tilt", "avg_swing_tilt", "swingtilt"),
        "attack_direction": find("attack_direction", "avg_attack_direction", "attackdirection"),
        "avg_intercept_y_vs_plate": find("avg_intercept_y_vs_plate", "avg_intercepty_vs_plate"),
        "avg_intercept_y_vs_batter": find("avg_intercept_y_vs_batter", "avg_intercepty_vs_batter"),
        "avg_batter_y_position": find("avg_batter_y_position", "avg_batter_y_pos", "avg_batter_yposition"),
        "avg_batter_x_position": find("avg_batter_x_position", "avg_batter_x_pos", "avg_batter_xposition", "avg_batter_position"),
        "swings_competitive": find("swings_competitive", "competitive_swings", "competitive_swings"),
        "batted_ball_events": find("batted_ball_events", "bbe", "battedballevents"),
        "pa": find("pa", "plate_appearances"),
        "year": find("year", "season", "yr"),
        "avg_foot_sep": find("avg_foot_sep", "avgfootsep"),
        "avg_stance_angle": find("avg_stance_angle", "avgstanceangle"),
        "batter_run_value": find("batter_run_value", "run_value", "runvalue"),
        "side": find("side", "bat_side", "batside"),
        "est_ba": find("est_ba", "estba"),
        "est_slg": find("est_slg", "estslg"),
        "est_woba": find("est_woba", "estwoba"),
        "xwOBA_pred": find("xwOBA_pred", "xwoba_pred", "xwoba"),
        "xba_pred": find("xba_pred"),
        "xslg_pred": find("xslg_pred")
    }

    for canonical, actual in mappings.items():
        if actual is not None and canonical not in df.columns:
            df[canonical] = df[actual]

    return df

df = normalize_columns(df)

def normalize_name(name):
    try:
        if not isinstance(name, str):
            return name
        name = name.strip()
        if "," in name:
            parts = [p.strip() for p in name.split(",", 1)]
            return f"{parts[1]} {parts[0]}".strip()
        return " ".join(name.split())
    except Exception:
        return name

if "Name" in df.columns:
    df["Name"] = df["Name"].apply(normalize_name)

expected_core = ["Name", "Age", "Swing+", "HitSkillPlus", "ImpactPlus"]
missing_core = [c for c in expected_core if c not in df.columns]
if missing_core:
    st.error(f"Missing required columns from data: {missing_core}")
    st.stop()

mlb_teams = [
    {"team": "ARI", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/ari.png&h=500&w=500"},
    {"team": "TOT", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/leagues/500/mlb.png&w=500&h=500"},
    {"team": "ATH", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/oak.png&h=500&w=500"},
    {"team": "ATL", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/atl.png&h=500&w=500"},
    {"team": "BAL", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/bal.png&h=500&w=500"},
    {"team": "BOS", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/bos.png&h=500&w=500"},
    {"team": "CHC", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/chc.png&h=500&w=500"},
    {"team": "CHW", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/chw.png&h=500&w=500"},
    {"team": "CIN", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/cin.png&h=500&w=500"},
    {"team": "CLE", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/cle.png&h=500&w=500"},
    {"team": "COL", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/col.png&h=500&w=500"},
    {"team": "DET", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/det.png&h=500&w=500"},
    {"team": "HOU", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/hou.png&h=500&w=500"},
    {"team": "KCR", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/kc.png&h=500&w=500"},
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
    {"team": "SDP", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/sd.png&h=500&w=500"},
    {"team": "SFG", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/sf.png&h=500&w=500"},
    {"team": "SEA", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/sea.png&h=500&w=500"},
    {"team": "STL", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/stl.png&h=500&w=500"},
    {"team": "TBR", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/tb.png&h=500&w=500"},
    {"team": "TEX", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/tex.png&h=500&w=500"},
    {"team": "TOR", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/tor.png&h=500&w=500"},
    {"team": "WSN", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/wsh.png&h=500&w=500"}
]
df_image = pd.DataFrame(mlb_teams)
image_dict = df_image.set_index('team')['logo_url'].to_dict()

FEATURE_LABELS = {
    "avg_bat_speed": "Avg Bat Speed (mph)",
    "swing_length": "Swing Length",
    "attack_angle": "Attack Angle (°)",
    "swing_tilt": "Swing Tilt (°)",
    "attack_direction": "Attack Direction",
    "avg_intercept_y_vs_plate": "Intercept Y vs Plate",
    "avg_intercept_y_vs_batter": "Intercept Y vs Batter",
    "avg_batter_y_position": "Batter Y Pos",
    "avg_batter_x_position": "Batter X Pos",
    "avg_foot_sep": "Avg Foot Sep",
    "avg_stance_angle": "Avg Stance Angle",
    "batted_ball_events": "Batted Ball Events",
    "pa": "PA",
    "year": "Season"
}

main_cmap = "RdYlBu_r"
elite_cmap = "Reds"

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

season_col = None
for c in ["year", "Year", "season"]:
    if c in df.columns:
        season_col = c
        break

comp_col = None
for c in ["swings_competitive", "competitive_swings"]:
    if c in df.columns:
        comp_col = c
        break

name_col = "Name"

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
def get_scaler_and_scaled_df(features, df_for_scaling):
    scaler = StandardScaler()
    X = df_for_scaling[features].astype(float)
    X_scaled = scaler.fit_transform(X)
    df_scaled = pd.DataFrame(X_scaled, columns=features, index=df_for_scaling.index)
    return scaler, df_scaled

def compute_cosine_similarity_between_rows(vecA, vecB):
    sim = np.dot(vecA, vecB) / (np.linalg.norm(vecA) * np.linalg.norm(vecB) + 1e-12)
    return float(sim)

def safe_rank_column(df_in, col):
    ranks = df_in[col].rank(ascending=False, method="min")
    filled = ranks.fillna(len(df_in) + 1).astype(int)
    return filled

params = st.experimental_get_query_params()
qp_player = None
qp_player_b = None
qp_page = None
qp_season = None

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

# capture a single 'season' param for Player links (so clicking link with &season=2024 will set player tab season)
if "season" in params and len(params["season"]) > 0:
    try:
        qp_season = unquote(params["season"][0])
    except Exception:
        qp_season = params["season"][0]

qp_season_a = params.get("seasonA", [None])[0] if "seasonA" in params else None
qp_season_b = params.get("seasonB", [None])[0] if "seasonB" in params else None

page_options = ["Main", "Player", "Compare"]
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

def open_compare_in_same_tab(playerA, playerB, seasonA=None, seasonB=None):
    try:
        params = {"playerA": playerA, "playerB": playerB, "page": "Compare"}
        if seasonA is not None:
            params["seasonA"] = seasonA
        if seasonB is not None:
            params["seasonB"] = seasonB
        st.experimental_set_query_params(**params)
    except Exception:
        try:
            st.experimental_set_query_params(player=playerA, playerB=playerB, page="Compare")
        except Exception:
            pass

if page == "Main":

    st.markdown(
        """
        <h2 style="text-align:center; margin-top:1.2em; margin-bottom:0.6em; font-size:1.6em; letter-spacing:0.01em; color:#2a3757;">
            Advanced Swing Metrics & SHAP Breakdown
        </h2>
        """,
        unsafe_allow_html=True
    )

    # Inject custom CSS for HTML-style filters
    st.markdown(
        """
        <style>
        .filter-container {
            display: flex;
            gap: 20px;
            margin-bottom: 10px;
        }
        .filter-box {
            background: #f8f9fa;
            padding: 10px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .filter-box label {
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # ===== DETECT SEASON COLUMN =====
    season_col = None
    for c in ["year", "Year", "season"]:
        if c in df.columns:
            season_col = c
            break

    # ===== FILTER UI =====
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)

    with filter_col1:
        season_selected_global = None
        if season_col:
            unique_years = sorted(df[season_col].dropna().unique())
            default_season = 2025 if 2025 in unique_years else (unique_years[-1] if unique_years else None)
            if unique_years:
                default_index = unique_years.index(default_season)
                season_selected_global = st.selectbox("Season", unique_years, index=default_index, key="main_season")

    with filter_col2:
        # Age min / max side by side (compact)
        min_age_val = int(df["Age"].min())
        max_age_val = int(df["Age"].max())

        age_c1, age_c2 = st.columns(2)
        with age_c1:
            age_min = st.number_input("Min Age", min_value=min_age_val, max_value=max_age_val, value=min_age_val, step=1, key="age_min")
        with age_c2:
            age_max = st.number_input("Max Age", min_value=min_age_val, max_value=max_age_val, value=max_age_val, step=1, key="age_max")

    with filter_col3:
        comp_col = None
        for c in ["swings_competitive", "competitive_swings"]:
            if c in df.columns:
                comp_col = c
                break

        swings_min_input = None
        swings_max_input = None
        if comp_col:
            swings_min_val = int(df[comp_col].min())
            swings_max_val = int(df[comp_col].max())

            swing_c1, swing_c2 = st.columns(2)
            with swing_c1:
                swings_min_input = st.number_input("Min Swings", min_value=swings_min_val, max_value=swings_max_val, value=swings_min_val, step=1, key="swings_min")
            with swing_c2:
                swings_max_input = st.number_input("Max Swings", min_value=swings_min_val, max_value=swings_max_val, value=swings_max_val, step=1, key="swings_max")

    with filter_col4:

        search_name = st.text_input("Search Player by Name", key="main_search")

    st.markdown("---")

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <h3 style="text-align:center; margin-top:6px; font-size:1.08em; color:#183153; letter-spacing:0.01em;">
            Player Metrics Table
        </h3>
        <div style="text-align:center; color:#6b7280; margin-bottom:6px; font-size:0.95em;">
            Raw feature values for every player.
        </div>
        """,
        unsafe_allow_html=True
    )

    # ===== APPLY FILTERS =====
    df_main_filtered = df.copy()

    if season_col and season_selected_global is not None:
        df_main_filtered = df_main_filtered[df_main_filtered[season_col] == season_selected_global]

    if search_name:
        df_main_filtered = df_main_filtered[df_main_filtered["Name"].str.contains(search_name, case=False, na=False)]

    # Age filter
    df_main_filtered = df_main_filtered[(df_main_filtered["Age"] >= age_min) & (df_main_filtered["Age"] <= age_max)]

    # Competitive swings filter
    if comp_col:
        df_main_filtered = df_main_filtered[
            (df_main_filtered[comp_col] >= swings_min_input) &
            (df_main_filtered[comp_col] <= swings_max_input)
        ]

    # ===== COLUMN SELECTION =====
    all_stats = ["Name", "Team"]
    if "year" in df_main_filtered.columns:
        all_stats.append("year")

    for c in ["pa", comp_col, "batted_ball_events"]:
        if c and c in df_main_filtered.columns and c not in all_stats:
            all_stats.append(c)

    for c in ["Swing+", "HitSkillPlus", "ImpactPlus"]:
        if c in df_main_filtered.columns and c not in all_stats:
            all_stats.append(c)

    remaining = [
        "swing_length", "avg_bat_speed", "swing_tilt", "attack_angle", "attack_direction",
        "avg_intercept_y_vs_plate", "avg_intercept_y_vs_batter", "avg_batter_y_position", "avg_batter_x_position",
        "avg_foot_sep", "avg_stance_angle"
    ]

    for c in remaining:
        if c in df_main_filtered.columns and c not in all_stats:
            all_stats.append(c)

    removed_cols = ["bip", "batter_run_value", "est_ba", "est_slg", "est_woba", "xwOBA_pred", "xba_pred", "xslg_pred", "side"]
    display_cols = [c for c in all_stats if c not in removed_cols]

    display_df = df_main_filtered[display_cols].copy()

    for c in ["pa", comp_col, "batted_ball_events"]:
        if c and c in display_df.columns:
            display_df[c] = display_df[c].round(0).astype("Int64")

    if "Age" in display_df.columns:
        display_df["Age"] = display_df["Age"].round(0).astype("Int64")

    rename_map = {}
    if "year" in display_df.columns:
        rename_map["year"] = "Season"
    if "pa" in display_df.columns:
        rename_map["pa"] = "PA"
    if comp_col and comp_col in display_df.columns:
        rename_map[comp_col] = "Competitive Swings"
    if "batted_ball_events" in display_df.columns:
        rename_map["batted_ball_events"] = "Batted Ball Events"
    if "HitSkillPlus" in display_df.columns:
        rename_map["HitSkillPlus"] = "BatToBall+"
    if "ImpactPlus" in display_df.columns:
        rename_map["ImpactPlus"] = "Impact+"

    for k, v in FEATURE_LABELS.items():
        if k in display_df.columns:
            rename_map[k] = v

    sort_col = "Swing+" if "Swing+" in display_df.columns else display_df.columns[0]

    styled = (
        display_df
        .rename(columns=rename_map)
        .sort_values(sort_col, ascending=False)
        .reset_index(drop=True)
    )
    
    plus_labels = []
    for p in ["Swing+", "HitSkillPlus", "ImpactPlus"]:
        if p in display_df.columns:
            plus_labels.append(rename_map.get(p, p))
    
    # Value coloring function
    def value_to_color(val, center=100, vmin=70, vmax=130):
        try:
            if pd.isna(val):
                return "#ffffff"
            val = float(val)
            val = max(min(val, vmax), vmin)
            if val == center:
                return "#ffffff"
            if val < center:
                ratio = (center - val) / (center - vmin)
                r, g, b = (31, 119, 180)
            else:
                ratio = (val - center) / (vmax - center)
                r, g, b = (214, 39, 40)
            r = int(255 + (r - 255) * ratio)
            g = int(255 + (g - 255) * ratio)
            b = int(255 + (b - 255) * ratio)
            return f"#{r:02x}{g:02x}{b:02x}"
        except Exception:
            return "#ffffff"
        
    # Cell formatting
    def format_cell(val):
        if pd.isna(val):
            return ""
        try:
            if isinstance(val, (float, np.floating)):
                return f"{val:.2f}"
            if isinstance(val, (int, np.integer)):
                return f"{val:d}"
            return str(val)
        except Exception:
            return str(val)
        
    abbrev_map = {
        "Competitive Swings": "CS",
        "Batted Ball Events": "BBE",
        "Swing Length": "SwL",
        "Avg Bat Speed (mph)": "BatS",
        "Swing Tilt (°)": "SwT",
        "Attack Angle (°)": "AA",
        "Attack Direction": "AD",
        "Intercept Y vs Plate": "IvP",
        "Intercept Y vs Batter": "IvB",
        "Batter Y Pos": "BatterY",
        "Batter X Pos": "BatterX",
        "Avg Foot Sep": "FS",
        "Avg Stance Angle": "StA"
    }
        
    columns_order = ["#"] + list(styled.columns)
    table_data = []
        
    # Base URL template provided by user — used for opening player page in new tab
    base_player_url = "https://swing-plus.streamlit.app/~/+/"
    # Season param for links (respect the global season filter)
    season_param = ""
    if season_selected_global is not None:
        try:
            season_param = f"&season={quote(str(season_selected_global))}"
        except Exception:
            season_param = f"&season={quote(str(season_selected_global))}"

    for idx, (_, row) in enumerate(styled.iterrows(), start=1):
        row_cells = [{"text": str(idx), "bg": ""}]
        for c in styled.columns:
            val = row[c]
            if c == "Team" and val in image_dict:
                content = f'<img src="{image_dict[val]}" alt="{val}" style="height:28px; display:block; margin:0 auto;" />'
                text_for_sort = str(val)
            else:
                # Make player names clickable: open player page in a new tab, preserve season filter
                if c == "Name" and pd.notna(val):
                    player_name = str(val)
                    try:
                        player_q = quote(player_name)
                    except Exception:
                        player_q = quote(str(player_name))
                    link = f'{base_player_url}?player={player_q}&page=Player{season_param}'
                    safe_name = html.escape(player_name)
                    content = f'<a href="{link}" target="_blank" rel="noopener noreferrer" style="color:#183153;text-decoration:none;font-weight:700;">{safe_name}</a>'
                    text_for_sort = player_name
                else:
                    content = format_cell(val)
                    text_for_sort = format_cell(val)
            bg = value_to_color(val) if c in plus_labels else ""
            # keep the cell text as rendered HTML/strings so JS will place it directly into the table
            row_cells.append({"text": content, "bg": bg, "sort_text": text_for_sort})
        table_data.append(row_cells)
        
    # rest of your HTML/JS script stays the same, but update the client JS to prefer sort_text when present
    html_table = f"""
    <style>
        .main-table-container {{
            width: 100%;
            margin: 0 auto;
            background: #f8fafc;
            border-radius: 14px;
            border: 1px solid #e3e8f0;
            box-shadow: 0 6px 18px rgba(42, 55, 87, 0.08);
            padding: 18px 18px 12px;
            box-sizing: border-box;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }}
        .main-table-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
            color: #24324c;
            font-weight: 600;
            font-size: 0.95rem;
            letter-spacing: 0.01em;
        }}
        .main-table-wrapper {{
            overflow-x: auto;
            overflow-y: hidden;
            max-height: none;
            border-radius: 10px;
            border: 1px solid #e0e6ef;
            background: #fff;
            padding: 10px 6px;
        }}
        table.custom-main-table {{
            width: 100%;
            border-collapse: collapse;
            font-family: inherit;
            font-size: 0.85rem;
            color: #1e293b;
            table-layout: auto;
        }}
        table.custom-main-table thead th {{
            background: #f9fafb;
            font-weight: 600;
            text-align: left;
            padding: 8px 12px;
            border-bottom: 1px solid #e2e8f0;
            font-variant-numeric: tabular-nums;
            white-space: nowrap;
            cursor: pointer;
        }}
        table.custom-main-table thead th.sorted-asc::after {{
            content: " ▲";
        }}
        table.custom-main-table thead th.sorted-desc::after {{
            content: " ▼";
        }}
        table.custom-main-table tbody td {{
            padding: 6px 12px;
            border-bottom: 1px solid #f1f5f9;
            font-variant-numeric: tabular-nums;
        }}
        table.custom-main-table tbody tr:hover td {{
            background: #f1f5f9;
        }}
        .table-foot {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 12px;
            flex-wrap: wrap;
            gap: 12px;
        }}
        .pagination-controls {{
            display: flex;
            gap: 8px;
        }}
        .pagination-controls button {{
            border: 1px solid #cbd5e1;
            background: #fff;
            color: #1f2937;
            padding: 6px 10px;
            border-radius: 10px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.15s ease;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        }}
        .pagination-controls button:disabled {{
            opacity: 0.5;
            cursor: default;
        }}
        .page-size-selector {{
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 0.85rem;
        }}
        .page-size-selector select {{
            padding: 6px 10px;
            border-radius: 8px;
            border: 1px solid #cbd5e1;
        }}
    </style>
    
    <div class="main-table-container">
        <div class="main-table-header">
            <span>Player Metrics (sorted by {sort_col})</span>
            <span id="row-count"></span>
        </div>
    
        <div class="main-table-wrapper">
            <table class="custom-main-table">
                <thead>
                    <tr>
                        {''.join([
                            f"<th title='{c}' data-col='{i}'>{abbrev_map.get(c, c)}</th>"
                            for i, c in enumerate(columns_order)
                        ])}
                    </tr>
                </thead>
                <tbody id="main-table-body"></tbody>
            </table>
        </div>
    
        <div class="table-foot">
            <div class="page-size-selector">
                <label for="page-size-select">Rows per page:</label>
                <select id="page-size-select">
                    <option value="30" selected>30</option>
                    <option value="50">50</option>
                    <option value="100">100</option>
                    <option value="200">200</option>
                </select>
            </div>
            <div class="pagination-controls">
                <button id="first-page">« First</button>
                <button id="prev-page">‹ Prev</button>
                <span id="page-info"></span>
                <button id="next-page">Next ›</button>
                <button id="last-page">Last »</button>
            </div>
        </div>
    </div>
    
    <script>
        // table_data rows include objects with optional "sort_text" used for sorting to avoid HTML interference
        const data = {json.dumps(table_data)};
        const columns = {json.dumps(columns_order)};
        let pageSize = 30;
        let currentPage = 1;
        let sortColumn = null;
        let sortDirection = 1;
    
        const bodyEl = document.getElementById('main-table-body');
        const rowCountEl = document.getElementById('row-count');
        const pageInfoEl = document.getElementById('page-info');
        const firstBtn = document.getElementById('first-page');
        const prevBtn = document.getElementById('prev-page');
        const nextBtn = document.getElementById('next-page');
        const lastBtn = document.getElementById('last-page');
        const headers = document.querySelectorAll('th[data-col]');
        const pageSizeSelect = document.getElementById('page-size-select');
    
        headers.forEach((th) => {{
            th.addEventListener('click', () => {{
                const colIndex = parseInt(th.getAttribute('data-col'));
                if (sortColumn === colIndex) {{
                    sortDirection = -sortDirection;
                }} else {{
                    sortColumn = colIndex;
                    sortDirection = 1;
                }}
                headers.forEach(header => {{
                    header.classList.remove('sorted-asc', 'sorted-desc');
                }});
                th.classList.add(sortDirection === 1 ? 'sorted-asc' : 'sorted-desc');
                renderTable();
            }});
        }});
    
        firstBtn.addEventListener('click', () => {{
            currentPage = 1;
            renderTable();
        }});
        prevBtn.addEventListener('click', () => {{
            if (currentPage > 1) currentPage--;
            renderTable();
        }});
        nextBtn.addEventListener('click', () => {{
            const totalPages = Math.max(1, Math.ceil(data.length / pageSize));
            if (currentPage < totalPages) currentPage++;
            renderTable();
        }});
        lastBtn.addEventListener('click', () => {{
            currentPage = Math.max(1, Math.ceil(data.length / pageSize));
            renderTable();
        }});
    
        pageSizeSelect.addEventListener('change', (e) => {{
            pageSize = parseInt(e.target.value, 10);
            currentPage = 1;
            renderTable();
        }});
    
        function getCellComparableText(cell) {{
            // Prefer explicit sort_text if provided (plain text), otherwise use raw text (may contain HTML)
            if (cell && typeof cell === 'object') {{
                if ('sort_text' in cell && cell.sort_text !== null && cell.sort_text !== undefined) {{
                    return String(cell.sort_text);
                }}
                return String(cell.text).replace(/<[^>]*>/g, ''); // strip HTML tags fallback
            }}
            return String(cell);
        }}
    
        function renderTable() {{
            let sortedData = [...data];
            if (sortColumn !== null) {{
                sortedData.sort((a, b) => {{
                    const aText = getCellComparableText(a[sortColumn]);
                    const bText = getCellComparableText(b[sortColumn]);
                    const aVal = parseFloat(aText.replace(/[^0-9+-.]/g, ''));
                    const bVal = parseFloat(bText.replace(/[^0-9+-.]/g, ''));
                    if (!isNaN(aVal) && !isNaN(bVal)) {{
                        return sortDirection * (aVal - bVal);
                    }}
                    return sortDirection * aText.localeCompare(bText);
                }});
            }}
    
            const totalRows = sortedData.length;
            const totalPages = Math.max(1, Math.ceil(totalRows / pageSize));
            if (currentPage > totalPages) currentPage = totalPages;
    
            const start = (currentPage - 1) * pageSize;
            const end = Math.min(start + pageSize, totalRows);
            const pageRows = sortedData.slice(start, end);
    
            bodyEl.innerHTML = pageRows.map(row => {{
                const cells = row.map((cell, i) => {{
                    const raw = (cell && cell.text !== undefined) ? cell.text : (cell || "");
                    const isNum = !isNaN((cell && cell.sort_text) ? cell.sort_text : (String(raw).replace(/<[^>]*>/g, ''))) && String((cell && cell.sort_text) ? cell.sort_text : raw).trim() !== "";
                    const align = isNum ? 'text-align: right;' : '';
                    const bg = (cell && cell.bg) ? `background:${{cell.bg}};` : '';
                    const style = (bg || align) ? ` style="${{bg}}{{align}}"` : '';
                    return `<td${{style}}>${{raw}}</td>`;
                }}).join('');
                return `<tr>${{cells}}</tr>`;
            }}).join('');
    
            rowCountEl.textContent = `Showing ${{start + 1}}–${{end}} of ${{totalRows}}`;
            pageInfoEl.textContent = `Page ${{currentPage}} / ${{totalPages}}`;
    
            prevBtn.disabled = currentPage === 1;
            firstBtn.disabled = currentPage === 1;
            nextBtn.disabled = currentPage === totalPages;
            lastBtn.disabled = currentPage === totalPages;
        }}
    
        renderTable();
    </script>
    """

    # Choose a dynamic height for the table component so when filtered down to few rows it doesn't leave
    # excessive blank vertical space. Height scales with number of rows.
    try:
        nrows_main = len(table_data)
        computed_height_main = max(420, min(1200, 110 + nrows_main * 36))
    except Exception:
        computed_height_main = 800

    components.html(html_table, height=computed_height_main, scrolling=True)

    # ===== LOAD NEW SHAP CSV =====
    shap_df = pd.read_csv("SwingPlus_SHAP_20_80_Profile.csv")
    
    # =============================
    # FIX NAME & TEAM FORMATTING
    # =============================
    if "Name" not in shap_df.columns and "name" in shap_df.columns:
        shap_df["Name"] = shap_df["name"].apply(
            lambda x: " ".join(x.split(", ")[::-1]) if isinstance(x, str) and "," in x else x
        )
    
    if "Team" not in shap_df.columns and "team" in shap_df.columns:
        shap_df["Team"] = shap_df["team"]
    
    # =============================
    # NORMALIZE COMPETITIVE SWINGS → CS
    # =============================
    if "competitive_swings" in shap_df.columns:
        shap_df = shap_df.rename(columns={"competitive_swings": "CS"})
        comp_col = "CS"
    elif "CS" in shap_df.columns:
        comp_col = "CS"
    else:
        comp_col = comp_col  # keep previous comp_col if exists
    
    # =============================
    # RENAME PLUS METRICS
    # =============================
    plus_rename_map = {
        "SwingPlus": "Swing+",
        "HitSkillPlus": "BatToBall+",
        "ImpactPlus": "Impact+"
    }
    shap_df = shap_df.rename(columns=plus_rename_map)
    
    # =============================
    # APPLY FILTERS (NOW INCLUDES search_name)
    # =============================
    df_shap_filtered = shap_df.copy()
    
    if season_col and season_selected_global is not None and "year" in df_shap_filtered.columns:
        df_shap_filtered = df_shap_filtered[df_shap_filtered[season_col] == season_selected_global]
    
    if comp_col and comp_col in df_shap_filtered.columns and swings_min_input is not None and swings_max_input is not None:
        df_shap_filtered = df_shap_filtered[
            (df_shap_filtered[comp_col] >= swings_min_input) &
            (df_shap_filtered[comp_col] <= swings_max_input)
        ]
    
    if "Age" in df_shap_filtered.columns:
        df_shap_filtered = df_shap_filtered[
            (df_shap_filtered["Age"] >= age_min) &
            (df_shap_filtered["Age"] <= age_max)
        ]

    # IMPORTANT: apply the same search_name filter to the SHAP table so it shows the same subset
    if 'search_name' in locals() and search_name:
        if "Name" in df_shap_filtered.columns:
            df_shap_filtered = df_shap_filtered[df_shap_filtered["Name"].str.contains(search_name, case=False, na=False)]
    
    # =============================
    # COLUMN STRUCTURE (NEW FORMAT)
    # =============================
    all_stats_shap = ["Name", "Team", "year", "pa"]
    
    if comp_col:
        all_stats_shap.append(comp_col)
    
    all_stats_shap.append("batted_ball_events")
    
    for plus in ["Swing+", "BatToBall+", "Impact+"]:
        if plus in df_shap_filtered.columns:
            all_stats_shap.append(plus)
    
    for base in [
        "avg_bat_speed", "swing_tilt", "attack_angle", "attack_direction",
        "avg_intercept_y_vs_plate", "avg_intercept_y_vs_batter",
        "avg_batter_y_position", "avg_batter_x_position",
        "swing_length", "avg_foot_sep", "avg_stance_angle"
    ]:
        for suffix in ["_grade", "_importance_pct"]:
            col = f"{base}{suffix}"
            if col in df_shap_filtered.columns:
                all_stats_shap.append(col)
    
    removed_cols_shap = [
        "id", "bip", "batter_run_value", "est_ba", "est_slg",
        "est_woba", "side"
    ]
    
    display_cols_shap = [
        c for c in all_stats_shap
        if c in df_shap_filtered.columns and c not in removed_cols_shap
    ]
    
    display_df_shap = df_shap_filtered[display_cols_shap].copy()
    
    # =============================
    # ROUNDING
    # =============================
    for col in display_df_shap.columns:
        if col.endswith("_grade"):
            display_df_shap[col] = display_df_shap[col].round(1)
        elif col.endswith("_importance_pct"):
            display_df_shap[col] = display_df_shap[col].round(1)
        elif display_df_shap[col].dtype.kind in "fc":
            display_df_shap[col] = display_df_shap[col].round(2)
    
    # =============================
    # UPDATED ABBREVIATION MAP
    # =============================
    abbrev_map = {
        "batted_ball_events": "BBE",
        "pa": "PA",
        "year": "Season",
    
        "avg_bat_speed_grade": "BatS (20-80)",
        "avg_bat_speed_importance_pct": "BatS (%)",
    
        "swing_tilt_grade": "SwT (20-80)",
        "swing_tilt_importance_pct": "SwT (%)",
    
        "attack_angle_grade": "AA (20-80)",
        "attack_angle_importance_pct": "AA (%)",
    
        "attack_direction_grade": "AD (20-80)",
        "attack_direction_importance_pct": "AD (%)",
    
        "avg_intercept_y_vs_plate_grade": "IvP (20-80)",
        "avg_intercept_y_vs_plate_importance_pct": "IvP (%)",
    
        "avg_intercept_y_vs_batter_grade": "IvB (20-80)",
        "avg_intercept_y_vs_batter_importance_pct": "IvB (%)",
    
        "avg_batter_y_position_grade": "BY (20-80)",
        "avg_batter_y_position_importance_pct": "BY (%)",
    
        "avg_batter_x_position_grade": "BX (20-80)",
        "avg_batter_x_position_importance_pct": "BX (%)",
    
        "swing_length_grade": "SwL (20-80)",
        "swing_length_importance_pct": "SwL (%)",
    
        "avg_foot_sep_grade": "FS (20-80)",
        "avg_foot_sep_importance_pct": "FS (%)",
    
        "avg_stance_angle_grade": "StA (20-80)",
        "avg_stance_angle_importance_pct": "StA (%)"
    }
    
    display_df_shap = display_df_shap.rename(columns=abbrev_map)
    
    # =============================
    # SORT BY Swing+
    # =============================
    sort_col_shap = "Swing+" if "Swing+" in display_df_shap.columns else (display_df_shap.columns[0] if len(display_df_shap.columns)>0 else None)
    if sort_col_shap:
        styled_shap = display_df_shap.sort_values(by=sort_col_shap, ascending=False).reset_index(drop=True)
    else:
        styled_shap = display_df_shap.reset_index(drop=True)

    def format_cell_shap(val, col_name=None):
        if pd.isna(val):
            return ""
    
        # Importance percentage columns
        if col_name and col_name.endswith("(%)"):
            return f"{val:.1f}%"
    
        # 20–80 grade columns
        if col_name and col_name.endswith("(20-80)"):
            return f"{val:.1f}"
    
        if isinstance(val, (float, np.floating)):
            return f"{val:.1f}"
    
        if isinstance(val, (int, np.integer)):
            return f"{val:d}"
    
        return str(val)


    def value_to_color_20_80(val):
        try:
            if pd.isna(val):
                return "#ffffff"
    
            val = float(val)
    
            # Clamp to 20–80
            val = max(min(val, 80), 20)
    
            # Normalize around 50
            if val == 50:
                return "#ffffff"
    
            if val < 50:
                # Blue scale (poor)
                ratio = (50 - val) / 30
                r, g, b = (31, 119, 180)
            else:
                # Red scale (elite)
                ratio = (val - 50) / 30
                r, g, b = (214, 39, 40)
    
            r = int(255 + (r - 255) * ratio)
            g = int(255 + (g - 255) * ratio)
            b = int(255 + (b - 255) * ratio)
    
            return f"#{r:02x}{g:02x}{b:02x}"
    
        except Exception:
            return "#ffffff"


    
    # =============================
    # FINAL TABLE OUTPUT
    # =============================
    columns_order_shap = ["#"] + list(styled_shap.columns)
    table_data_shap = []
    
    # Build clickable name links for SHAP table as well, respecting the global season filter
    for idx, (_, row) in enumerate(styled_shap.iterrows(), start=1):
        row_cells = [{"text": str(idx), "bg": ""}]
    
        for c in styled_shap.columns:
            val = row[c]
    
            if c == "Team" and val in image_dict:
                content = f'<img src="{image_dict[val]}" alt="{val}" style="height:28px; display:block; margin:0 auto;" />'
                sort_text = str(val)
            elif c == "Name" and pd.notna(val):
                player_name = str(val)
                try:
                    player_q = quote(player_name)
                except Exception:
                    player_q = quote(str(player_name))
                link = f'{base_player_url}?player={player_q}&page=Player{season_param}'
                safe_name = html.escape(player_name)
                content = f'<a href="{link}" target="_blank" rel="noopener noreferrer" style="color:#183153;text-decoration:none;font-weight:700;">{safe_name}</a>'
                sort_text = player_name
            else:
                content = format_cell_shap(val, c)
                sort_text = format_cell_shap(val, c)
    
            bg = value_to_color_20_80(val) if c.endswith("(20-80)") else ""
            row_cells.append({"text": content, "bg": bg, "sort_text": sort_text})
    
        table_data_shap.append(row_cells)

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <h3 style="text-align:center; margin-top:6px; font-size:1.08em; color:#183153; letter-spacing:0.01em;">
            SHAP Breakdown Table
        </h3>
        <div style="text-align:center; color:#6b7280; margin-bottom:6px; font-size:0.95em;">
            How each mechanical feature moved the model's Swing+ prediction for individual players.
        </div>
        """,
        unsafe_allow_html=True
    )

    html_table_shap = f"""
    <style>
        .main-table-container {{
            width: 100%;
            margin: 0 auto;
            background: #f8fafc;
            border-radius: 14px;
            border: 1px solid #e3e8f0;
            box-shadow: 0 6px 18px rgba(42, 55, 87, 0.08);
            padding: 18px 18px 12px;
            box-sizing: border-box;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }}
        .main-table-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
            color: #24324c;
            font-weight: 600;
            font-size: 0.95rem;
            letter-spacing: 0.01em;
        }}
        .main-table-wrapper {{
            overflow-x: auto;
            overflow-y: hidden;
            max-height: none;
            border-radius: 10px;
            border: 1px solid #e0e6ef;
            background: #fff;
            padding: 10px 6px;
        }}
        table.custom-main-table {{
            width: 100%;
            border-collapse: collapse;
            font-family: inherit;
            font-size: 0.85rem;
            color: #1e293b;
            table-layout: auto;
        }}
        table.custom-main-table thead th {{
            background: #f9fafb;
            font-weight: 600;
            text-align: left;
            padding: 8px 12px;
            border-bottom: 1px solid #e2e8f0;
            font-variant-numeric: tabular-nums;
            white-space: nowrap;
            cursor: pointer;
        }}
        table.custom-main-table thead th.sorted-asc::after {{
            content: " ▲";
        }}
        table.custom-main-table thead th.sorted-desc::after {{
            content: " ▼";
        }}
        table.custom-main-table tbody td {{
            padding: 6px 12px;
            border-bottom: 1px solid #f1f5f9;
            font-variant-numeric: tabular-nums;
        }}
        table.custom-main-table tbody tr:hover td {{
            background: #f1f5f9;
        }}
        .table-foot {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 12px;
            flex-wrap: wrap;
            gap: 12px;
        }}
        .pagination-controls {{
            display: flex;
            gap: 8px;
        }}
        .pagination-controls button {{
            border: 1px solid #cbd5e1;
            background: #fff;
            color: #1f2937;
            padding: 6px 10px;
            border-radius: 10px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.15s ease;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        }}
        .pagination-controls button:disabled {{
            opacity: 0.5;
            cursor: default;
        }}
        .page-size-selector {{
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 0.85rem;
        }}
        .page-size-selector select {{
            padding: 6px 10px;
            border-radius: 8px;
            border: 1px solid #cbd5e1;
        }}
    </style>
    
    <div class="main-table-container">
        <div class="main-table-header">
            <span>Player SHAP Metrics (sorted by {sort_col_shap})</span>
            <span id="row-count-shap"></span>
        </div>
    
        <div class="main-table-wrapper">
            <table class="custom-main-table">
                <thead>
                    <tr>
                        {''.join([
                            f"<th title='{c}' data-col='{i}'>{abbrev_map.get(c, c)}</th>"
                            for i, c in enumerate(['#'] + list(styled_shap.columns))
                        ])}
                    </tr>
                </thead>
                <tbody id="shap-table-body"></tbody>
            </table>
        </div>
    
        <div class="table-foot">
            <div class="page-size-selector">
                <label for="page-size-select-shap">Rows per page:</label>
                <select id="page-size-select-shap">
                    <option value="30" selected>30</option>
                    <option value="50">50</option>
                    <option value="100">100</option>
                    <option value="200">200</option>
                </select>
            </div>
            <div class="pagination-controls">
                <button id="first-page-shap">« First</button>
                <button id="prev-page-shap">‹ Prev</button>
                <span id="page-info-shap"></span>
                <button id="next-page-shap">Next ›</button>
                <button id="last-page-shap">Last »</button>
            </div>
        </div>
    </div>
    
    <script>
        const data_shap = {json.dumps(table_data_shap)};
        const columns_shap = {json.dumps(['#'] + list(styled_shap.columns))};
        let pageSize_shap = 30;
        let currentPage_shap = 1;
        let sortColumn_shap = null;
        let sortDirection_shap = 1;
    
        const bodyEl_shap = document.getElementById('shap-table-body');
        const rowCountEl_shap = document.getElementById('row-count-shap');
        const pageInfoEl_shap = document.getElementById('page-info-shap');
        const firstBtn_shap = document.getElementById('first-page-shap');
        const prevBtn_shap = document.getElementById('prev-page-shap');
        const nextBtn_shap = document.getElementById('next-page-shap');
        const lastBtn_shap = document.getElementById('last-page-shap');
        const headers_shap = document.querySelectorAll('th[data-col]');
        const pageSizeSelect_shap = document.getElementById('page-size-select-shap');
    
        headers_shap.forEach((th) => {{
            th.addEventListener('click', () => {{
                const colIndex = parseInt(th.getAttribute('data-col'));
                if (sortColumn_shap === colIndex) {{
                    sortDirection_shap = -sortDirection_shap;
                }} else {{
                    sortColumn_shap = colIndex;
                    sortDirection_shap = 1;
                }}
                headers_shap.forEach(header => {{
                    header.classList.remove('sorted-asc', 'sorted-desc');
                }});
                th.classList.add(sortDirection_shap === 1 ? 'sorted-asc' : 'sorted-desc');
                renderTableShap();
            }});
        }});
    
        firstBtn_shap.addEventListener('click', () => {{
            currentPage_shap = 1;
            renderTableShap();
        }});
        prevBtn_shap.addEventListener('click', () => {{
            if (currentPage_shap > 1) currentPage_shap--;
            renderTableShap();
        }});
        nextBtn_shap.addEventListener('click', () => {{
            const totalPages = Math.max(1, Math.ceil(data_shap.length / pageSize_shap));
            if (currentPage_shap < totalPages) currentPage_shap++;
            renderTableShap();
        }});
        lastBtn_shap.addEventListener('click', () => {{
            currentPage_shap = Math.max(1, Math.ceil(data_shap.length / pageSize_shap));
            renderTableShap();
        }});
        pageSizeSelect_shap.addEventListener('change', (e) => {{
            pageSize_shap = parseInt(e.target.value, 10);
            currentPage_shap = 1;
            renderTableShap();
        }});
    
        function getCellComparableText(cell) {{
            if (cell && typeof cell === 'object') {{
                if ('sort_text' in cell && cell.sort_text !== null && cell.sort_text !== undefined) {{
                    return String(cell.sort_text);
                }}
                return String(cell.text).replace(/<[^>]*>/g, '');
            }}
            return String(cell);
        }}
    
        function renderTableShap() {{
            let sortedData = [...data_shap];
            if (sortColumn_shap !== null) {{
                sortedData.sort((a, b) => {{
                    const aText = getCellComparableText(a[sortColumn_shap]);
                    const bText = getCellComparableText(b[sortColumn_shap]);
                    const aVal = parseFloat(aText.replace(/[^0-9+-.]/g, ''));
                    const bVal = parseFloat(bText.replace(/[^0-9+-.]/g, ''));
                    if (!isNaN(aVal) && !isNaN(bVal)) {{
                        return sortDirection_shap * (aVal - bVal);
                    }}
                    return sortDirection_shap * aText.localeCompare(bText);
                }});
            }}
    
            const totalRows = sortedData.length;
            const totalPages = Math.max(1, Math.ceil(totalRows / pageSize_shap));
            if (currentPage_shap > totalPages) currentPage_shap = totalPages;
    
            const start = (currentPage_shap - 1) * pageSize_shap;
            const end = Math.min(start + pageSize_shap, totalRows);
            const pageRows = sortedData.slice(start, end);
    
            bodyEl_shap.innerHTML = pageRows.map(row => {{
                const cells = row.map((cell, i) => {{
                    const raw = (cell && cell.text !== undefined) ? cell.text : (cell || "");
                    const isNum = !isNaN((cell && cell.sort_text) ? cell.sort_text : (String(raw).replace(/<[^>]*>/g, ''))) && String((cell && cell.sort_text) ? cell.sort_text : raw).trim() !== "";
                    const align = isNum ? 'text-align: right;' : '';
                    const bg = (cell && cell.bg) ? `background:${{cell.bg}};` : '';
                    const style = (bg || align) ? ` style="${{bg}}{{align}}"` : '';
                    return `<td${{style}}>${{raw}}</td>`;
                }}).join('');
                return `<tr>${{cells}}</tr>`;
            }}).join('');
    
            rowCountEl_shap.textContent = `Showing ${{start + 1}}–${{end}} of ${{totalRows}}`;
            pageInfoEl_shap.textContent = `Page ${{currentPage_shap}} / ${{totalPages}}`;
    
            prevBtn_shap.disabled = currentPage_shap === 1;
            firstBtn_shap.disabled = currentPage_shap === 1;
            nextBtn_shap.disabled = currentPage_shap === totalPages;
            lastBtn_shap.disabled = currentPage_shap === totalPages;
        }}
    
        renderTableShap();
    </script>
    """

    try:
        nrows_shap = len(table_data_shap)
        computed_height_shap = max(420, min(1200, 110 + nrows_shap * 36))
    except Exception:
        computed_height_shap = 800

    components.html(html_table_shap, height=computed_height_shap, scrolling=True)


# ---------------- Player tab ----------------
elif page == "Player":
    st.markdown(
        """
        <h2 style="text-align:center; margin-top:1.2em; margin-bottom:0.6em; font-size:1.6em; letter-spacing:0.01em; color:#2a3757;">
            Player Detail
        </h2>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    params = st.experimental_get_query_params()
    qp_player = None
    qp_season_local = None
    if "player" in params and len(params["player"]) > 0:
        try:
            qp_player = unquote(params["player"][0])
        except Exception:
            qp_player = params["player"][0]
    if "season" in params and len(params["season"]) > 0:
        try:
            qp_season_local = unquote(params["season"][0])
        except Exception:
            qp_season_local = params["season"][0]

    # Player selector uses full df so tab not affected by main filters
    player_options = sorted(df["Name"].unique())
    default_index = 0
    if qp_player and qp_player in player_options:
        default_index = player_options.index(qp_player)
    
    col1, col2 = st.columns(2)
    
    with col1:
        player_select = st.selectbox(
            "Select a Player",
            player_options,
            key="player_select",
            index=default_index
        )
    
    player_season_selected = None
    
    if season_col:
        try:
            player_seasons = sorted(
                df[df["Name"] == player_select][season_col].dropna().unique()
            )
    
            if player_seasons:
                # prefer season passed via link (qp_season_local) then qp_season (global captured earlier)
                preferred_season = None
                if qp_season_local is not None:
                    preferred_season = qp_season_local
                elif qp_season is not None:
                    preferred_season = qp_season

                # find a season value in player_seasons that matches preferred_season when cast to str
                default_player_season = player_seasons[-1]
                if preferred_season is not None:
                    match = None
                    for s in player_seasons:
                        try:
                            if str(s) == str(preferred_season):
                                match = s
                                break
                        except Exception:
                            continue
                    if match is not None:
                        default_player_season = match

                idx = player_seasons.index(default_player_season) if default_player_season in player_seasons else 0
    
                with col2:
                    player_season_selected = st.selectbox(
                        "Season",
                        player_seasons,
                        index=idx,
                        key="player_season_select"
                    )
            else:
                player_season_selected = None
        except Exception:
            player_season_selected = None


    if player_season_selected is not None and season_col:
        try:
            pr_df = df[(df["Name"] == player_select) & (df[season_col] == player_season_selected)]
            if len(pr_df) > 0:
                player_row = pr_df.iloc[0]
            else:
                player_row = df[df["Name"] == player_select].iloc[0]
        except Exception:
            player_row = df[df["Name"] == player_select].iloc[0]
    else:
        player_row = df[df["Name"] == player_select].iloc[0]

    player_title = player_select
    if player_season_selected is not None:
        player_title = f"{player_select} ({player_season_selected})"

    headshot_size = 96
    logo_size = 80

    player_bio = ""
    bat_side = "R"
    if "id" in player_row and pd.notnull(player_row["id"]):
        try:
            player_id = str(int(player_row["id"]))
            mlb_bio_url = f"https://statsapi.mlb.com/api/v1/people/{player_id}"
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

    player_bio_html = f"<span style='font-size:0.98em;color:#495366;margin-top:7px;margin-bottom:0;font-weight:500;letter-spacing:0.02em;opacity:0.82;'>{player_bio}</span>" if player_bio else ""

    headshot_html = ""
    if "id" in player_row and pd.notnull(player_row["id"]):
        try:
            player_id = str(int(player_row["id"]))
            headshot_url = f"https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/w_640,q_auto:best/v1/people/{player_id}/headshot/silo/current.png"
        except Exception:
            headshot_url = "https://img.mlbstatic.com/mlb-photos/image/upload/v1/people/0/headshot/silo/current.png"
        headshot_html = f'<img src="{headshot_url}" style="height:{headshot_size}px;width:{headshot_size}px;object-fit:cover;border-radius:14px;vertical-align:middle;margin-right:18px;background:transparent;" />'
    else:
        fallback_url = "https://img.mlbstatic.com/mlb-photos/image/upload/v1/people/0/headshot/silo/current.png"
        headshot_html = f'<img src="{fallback_url}" style="height:{headshot_size}px;width:{headshot_size}px;object-fit:cover;border-radius:14px;vertical-align:middle;margin-right:18px;background:transparent;" />'

    if player_season_selected is not None:
        player_name_html = f'<span style="font-size:2.3em;font-weight:800;color:#183153;letter-spacing:0.01em;vertical-align:middle;margin:0 20px;">{player_select} <span style="font-size:0.6em;color:#6b7280;">{player_season_selected}</span></span>'
        player_title = f"{player_select} {player_season_selected}"
    else:
        player_name_html = f'<span style="font-size:2.3em;font-weight:800;color:#183153;letter-spacing:0.01em;vertical-align:middle;margin:0 20px;">{player_select}</span>'
        player_title = player_select

    team_logo_html = ""
    if "Team" in player_row and pd.notnull(player_row["Team"]):
        team_abbr = str(player_row["Team"]).strip()
        team_logo_url = image_dict.get(team_abbr, "")
        if team_logo_url:
            team_logo_html = f'<div style="margin-left:14px; display:flex; align-items:center;"><img src="{team_logo_url}" style="height:{logo_size}px;width:{logo_size}px;border-radius:8px;object-fit:contain;" /></div>'

    st.markdown(
        f"""
        <div style="display:flex;justify-content:center;align-items:center;margin-bottom:6px;margin-top:8px;">
            {headshot_html}
            <div style="display:flex;flex-direction:column;align-items:center;">
                {player_name_html}
                {player_bio_html}
            </div>
            {team_logo_html}
        </div>
        """,
        unsafe_allow_html=True
    )

    # Compute ranks within season context (player-specific if chosen)
    if season_col:
        season_context = None
        if player_season_selected is not None:
            season_context = player_season_selected
        elif 'season_selected_global' in locals() and season_selected_global is not None:
            season_context = season_selected_global

        if season_context is not None:
            try:
                df_rank = df[df[season_col] == season_context].copy()
            except Exception:
                df_rank = df.copy()
        else:
            df_rank = df.copy()
    else:
        df_rank = df.copy()

    total_players = len(df_rank) if len(df_rank) > 0 else len(df)

    if "Swing+" in df_rank.columns:
        df_rank["Swing+_rank"] = safe_rank_column(df_rank, "Swing+")
    if "HitSkillPlus" in df_rank.columns:
        df_rank["ProjSwing+_rank"] = safe_rank_column(df_rank, "HitSkillPlus")
    if "ImpactPlus" in df_rank.columns:
        df_rank["PowerIndex+_rank"] = safe_rank_column(df_rank, "ImpactPlus")

    p_idx = player_row.name
    try:
        if p_idx in df_rank.index:
            p_swing_rank = int(df_rank.loc[p_idx, "Swing+_rank"]) if "Swing+_rank" in df_rank.columns else None
            p_proj_rank = int(df_rank.loc[p_idx, "ProjSwing+_rank"]) if "ProjSwing+_rank" in df_rank.columns else None
            p_power_rank = int(df_rank.loc[p_idx, "PowerIndex+_rank"]) if "PowerIndex+_rank" in df_rank.columns else None
        else:
            sub = df_rank[df_rank["Name"] == player_select]
            if len(sub) > 0:
                p_swing_rank = int(sub["Swing+_rank"].iloc[0]) if "Swing+_rank" in sub.columns else None
                p_proj_rank = int(sub["ProjSwing+_rank"].iloc[0]) if "ProjSwing+_rank" in sub.columns else None
                p_power_rank = int(sub["PowerIndex+_rank"].iloc[0]) if "PowerIndex+_rank" in sub.columns else None
            else:
                p_swing_rank = None
                p_proj_rank = None
                p_power_rank = None
    except Exception:
        p_swing_rank = None
        p_proj_rank = None
        p_power_rank = None

    def plus_color_by_rank(rank, total, start_hex="#D32F2F", end_hex="#3B82C4"):
        if rank is None:
            return "#666666"
        if total <= 1:
            ratio = 0.0
        else:
            ratio = (rank - 1) / (total - 1)
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

    swing_color = plus_color_by_rank(p_swing_rank, total_players)
    proj_color = plus_color_by_rank(p_proj_rank, total_players)
    power_color = plus_color_by_rank(p_power_rank, total_players)

    st.markdown(
        f"""
        <div style="display: flex; justify-content: center; gap: 32px; margin-top: 0px; margin-bottom: 28px;">
          <div style="background: #fff; border-radius: 16px; box-shadow: 0 2px 12px rgba(0,0,0,0.06); padding: 24px 32px; text-align: center; min-width: 160px;">
            <div style="font-size: 2.2em; font-weight: 700; color: {swing_color};">{player_row.get('Swing+', np.nan):.2f}</div>
            <div style="font-size: 1.1em; color: #888; font-weight: 600; letter-spacing: 0.5px; margin-bottom: 4px;">Swing+</div>
            <span style="background: #FFC10733; color: #B71C1C; border-radius: 10px; font-size: 0.98em; padding: 6px 10px;">Rank {p_swing_rank if p_swing_rank is not None else 'N/A'} of {total_players}</span>
          </div>
          <div style="background: #fff; border-radius: 16px; box-shadow: 0 2px 12px rgba(0,0,0,0.06); padding: 24px 32px; text-align: center; min-width: 160px;">
            <div style="font-size: 2.2em; font-weight: 700; color: {proj_color};">{player_row.get('ProjSwing+', player_row.get('HitSkillPlus', np.nan)):.2f}</div>
            <div style="font-size: 1.1em; color: #888; font-weight: 600; letter-spacing: 0.5px; margin-bottom: 4px;">BatToBall+</div>
            <span style="background: #C8E6C933; color: #1B5E20; border-radius: 10px; font-size: 0.98em; padding: 6px 10px;">Rank {p_proj_rank if p_proj_rank is not None else 'N/A'} of {total_players}</span>
          </div>
          <div style="background: #fff; border-radius: 16px; box-shadow: 0 2px 12px rgba(0,0,0,0.06); padding: 24px 32px; text-align: center; min-width: 160px;">
            <div style="font-size: 2.2em; font-weight: 700; color: {power_color};">{player_row.get('PowerIndex+', player_row.get('ImpactPlus', np.nan)):.2f}</div>
            <div style="font-size: 1.1em; color: #888; font-weight: 600; letter-spacing: 0.5px; margin-bottom: 4px;">Impact+</div>
            <span style="background: #B3E5FC33; color: #01579B; border-radius: 10px; font-size: 0.98em; padding: 6px 10px;">Rank {p_power_rank if p_power_rank is not None else 'N/A'} of {total_players}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Build video URL
    video_url = None
    if "id" in player_row and pd.notnull(player_row["id"]):
        try:
            player_id = str(int(player_row["id"]))
        except Exception:
            player_id = None

        video_year = None
        try:
            if player_season_selected is not None:
                video_year = int(player_season_selected)
            elif 'season_selected_global' in locals() and season_selected_global is not None:
                video_year = int(season_selected_global)
            elif "year" in player_row and pd.notna(player_row["year"]):
                video_year = int(player_row["year"])
            else:
                video_year = 2025
        except Exception:
            video_year = 2025

        if player_id:
            video_url = f"https://builds.mlbstatic.com/baseballsavant.mlb.com/swing-path/splendid-splinter/cut/{player_id}-{video_year}-{bat_side}.mp4"

    DEFAULT_ONEIL_CRUZ_IDS = ['665833-2025-L', '665833-2025-R', '665833-2025-S']
    default_name = "Oneil Cruz"
    showing_default = False
    if video_url:
        showing_default = any(d in video_url for d in DEFAULT_ONEIL_CRUZ_IDS)

    if video_url:
        if showing_default:
            video_note = f"No custom video data available for this player — showing a default example ({default_name})."
        else:
            video_note = "Below is the Baseball Savant Swing Path / Attack Angle visualization for this player."

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
                <video id="player-savant-video" width="900" height="480" style="border-radius:9px; box-shadow:0 2px 12px rgba(0,0,0,0.12);" autoplay muted playsinline key="{player_id}-{video_year}-{bat_side}">
                    <source src="{video_url}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
            </div>
            """,
            unsafe_allow_html=True
        )

    mech_features_available = [f for f in mechanical_features if f in df.columns]

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
            Swing+ Feature Contributions
        </h3>
        <div style="text-align:center; color:#6b7280; margin-bottom:6px; font-size:0.95em;">
            How each mechanical feature moved the model's Swing+ prediction for this player.
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([1, 1])

    shap_pred_label = f"{shap_pred:.2f}" if (shap_pred is not None and not pd.isna(shap_pred)) else "N/A"
    swing_actual_label = f"{player_row.get('Swing+', np.nan):.2f}" if (player_row.get("Swing+") is not None and not pd.isna(player_row.get("Swing+"))) else "N/A"
    base_label = f"{shap_base:.2f}" if (shap_base is not None and not pd.isna(shap_base)) else "N/A"

    with col1:
        st.markdown(f"<div style='text-align:center;font-weight:700;color:#183153;'>Model prediction: {shap_pred_label} &nbsp; | &nbsp; Actual Swing+: {swing_actual_label}</div>", unsafe_allow_html=True)
        if not model_loaded or explainer is None or shap_df is None or len(shap_df) == 0:
            st.info("Swing+ model or SHAP explainer not available. Ensure SwingPlus.pkl is a supported model/pipeline.")
            if model_error:
                st.caption(f"Model load error: {model_error}")
        else:
            TOP_SHOW = min(8, len(shap_df))
            df_plot_top = shap_df.head(TOP_SHOW).copy()
            df_plot_top = df_plot_top.sort_values("pct_of_abs", ascending=False).reset_index(drop=True)

            y = df_plot_top["feature"].map(lambda x: FEATURE_LABELS.get(x, x)).tolist()
            x_vals = df_plot_top["shap_value"].astype(float).tolist()
            pct_vals = df_plot_top["pct_of_abs"].astype(float).tolist()
            colors = ["#D8573C" if float(v) > 0 else "#3B82C4" for v in x_vals]

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
                height=520,
                showlegend=False,
                font=dict(size=11)
            )
            st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True, "displayModeBar": False})

        with col2:
        
            st.markdown(
                f"<div style='text-align:center;font-weight:700;color:#183153; margin-bottom:1px;'>Model baseline: {base_label}</div>",
                unsafe_allow_html=True
            )
        
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
                display_df["Value"] = display_df["Value"].apply(lambda v: f"{v:.2f}" if pd.notna(v) else "NaN")
                display_df["Contribution"] = display_df["Contribution"].apply(lambda v: f"{v:.3f}")
                display_df["PctImportance"] = display_df["PctImportance"].apply(lambda v: f"{v:.0%}")
                display_df = display_df.reset_index(drop=True)
        
                st.markdown("""
                <style>
                .comp-table {
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 0.1px;
                    font-size: 0.88em;
                    background: #FFFFFF;
                    border: 2px solid #111827;
                    border-radius: 10px;
                    overflow: hidden;
                }
        
                .comp-table th {
                    background: #F3F4F6;
                    color: #374151;
                    padding: 10px 6px;
                    font-weight: 700;
                    text-align: center;
                    border-bottom: 1px solid #D1D5DB;
                }
        
                .comp-table td {
                    padding: 9px 6px;
                    text-align: center;
                    border-bottom: 1px solid #E5E7EB;
                    color: #111827;
                }
        
                .comp-table tr:last-child td {
                    border-bottom: 1px solid #E5E7EB;
                }
        
                .comp-feature {
                    text-align: left;
                    font-weight: 600;
                    color: #1F2937;
                    padding-left: 8px;
                }
                </style>
                """, unsafe_allow_html=True)
        
                # ---------- Build HTML rows safely ----------
                import html as _html
                html_rows = ""
                for _, r in display_df.iterrows():
                    feat = _html.escape(str(r["Feature"]))
                    val = _html.escape(str(r["Value"]))
                    contrib = _html.escape(str(r["Contribution"]))
                    pct = _html.escape(str(r["PctImportance"]))
                    html_rows += (
                        "<tr>"
                        f"<td class='comp-feature'>{feat}</td>"
                        f"<td>{val}</td>"
                        f"<td>{contrib}</td>"
                        f"<td>{pct}</td>"
                        "</tr>"
                    )
        
                # ---------- Final HTML ----------
                html_table = (
                    "<table class='comp-table'>"
                    "<thead>"
                    "<tr>"
                    "<th>Feature</th>"
                    "<th>Value</th>"
                    "<th>Contribution</th>"
                    "<th>Importance</th>"
                    "</tr>"
                    "</thead>"
                    f"<tbody>{html_rows}</tbody>"
                    "</table>"
                )
        
                st.markdown(html_table, unsafe_allow_html=True)

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown(
        """
        <h3 style="text-align:center; margin-top:6px; font-size:1.05em; color:#183153;">
            Year-to-Year SHAP Contributions for Mechanical Groupings
        </h3>
        <div style="text-align:center; color:#6b7280; margin-bottom:8px; font-size:0.95em;">
            Line plots show per-year SHAP contribution for each metric in the grouping for this player.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Define groupings and corresponding feature keys
    groupings = {
        "Stance": [
            "avg_stance_angle",
            "avg_batter_y_position",
            "avg_batter_x_position",
            "avg_foot_sep"
        ],
        "Timing & Direction": [
            "avg_intercept_y_vs_plate",
            "avg_intercept_y_vs_batter",
            "attack_angle",
            "attack_direction"
        ],
        "Swing Mechanics": [
            "avg_bat_speed",
            "swing_length",
            "swing_tilt"
        ]
    }

    # Gather seasons for this player (sorted)
    if season_col:
        seasons_for_player = sorted(df[df["Name"] == player_select][season_col].dropna().unique())
    else:
        seasons_for_player = []

    # If no seasons, try to use "year" column if present or fallback to unique years in df
    if not seasons_for_player:
        if season_col and season_col in df.columns:
            seasons_for_player = sorted(df[season_col].dropna().unique())
    # ensure seasons are simple types (strings/int)
    seasons_for_player = [s for s in seasons_for_player]
    # convert to list of strings for plotting labels
    seasons_labels = [str(s) for s in seasons_for_player]

    # If model/explainer not available, show informative message
    if not model_loaded or explainer is None:
        st.info("Model or SHAP explainer not available — cannot compute year-to-year SHAP contributions.")
    elif not seasons_for_player:
        st.info("No season history available for this player to plot year-to-year contributions.")
    else:
        # For each season compute shap values for player (if season row exists use that, otherwise use best available)
        per_season_shap = {}
        for s in seasons_for_player:
            # find the row for this player+season, fallback to any row for the player
            try:
                subset = df[(df["Name"] == player_select) & (df[season_col] == s)]
                if len(subset) > 0:
                    row = subset.iloc[0]
                else:
                    row = df[df["Name"] == player_select].iloc[0]
            except Exception:
                row = df[df["Name"] == player_select].iloc[0]
            # compute shap (returns Series indexed by feature names)
            try:
                shap_ser, _, _ = compute_shap(row, mech_features_available)
                if shap_ser is None:
                    shap_ser = pd.Series(0.0, index=mech_features_available)
                # align index to allow lookup by our feature keys
                per_season_shap[s] = shap_ser
            except Exception:
                per_season_shap[s] = pd.Series(0.0, index=mech_features_available)

        # Build three columns for plots
        col_a, col_b, col_c = st.columns(3)

        # Color palette
        palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

        for (group_name, feats), col in zip(groupings.items(), [col_a, col_b, col_c]):
            # Prepare data matrix: rows = seasons, cols = feats
            y_series = {f: [] for f in feats}
            for s in seasons_for_player:
                shap_ser = per_season_shap.get(s, pd.Series(0.0, index=mech_features_available))
                for f in feats:
                    # Attempt alternative keys if model uses slightly different naming
                    val = None
                    if f in shap_ser.index:
                        val = float(shap_ser.get(f, 0.0))
                    else:
                        # try lower/underscore variants
                        alt = f.replace(" ", "_")
                        if alt in shap_ser.index:
                            val = float(shap_ser.get(alt, 0.0))
                        else:
                            # fallback 0
                            val = 0.0
                    y_series[f].append(val)

            # Create plotly line figure
        fig = go.Figure()
        for i, f in enumerate(feats):
            color = palette[i % len(palette)]
            label = FEATURE_LABELS.get(f, f)
            fig.add_trace(go.Scatter(
                x=seasons_labels,
                y=y_series[f],
                mode="lines+markers",
                name=label,
                line=dict(color=color, width=2.5),
                marker=dict(size=5),
                hoverinfo="skip",  # disables hover
                showlegend=True
            ))
        
        fig.update_layout(
            title=dict(
                text=f"<b>{group_name}</b>",
                x=0.5,
                xanchor='center',
                font=dict(size=15, color="#1e293b")
            ),
            height=290,
            margin=dict(l=20, r=20, t=40, b=30),
            template="plotly_white",
            font=dict(size=11, color="#374151"),
            xaxis=dict(
                title="Season",
                titlefont=dict(size=11),
                showgrid=True,
                gridcolor="#e5e7eb"
            ),
            yaxis=dict(
                title="SHAP Contribution",
                titlefont=dict(size=11),
                showgrid=True,
                gridcolor="#e5e7eb"
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                font=dict(size=10)
            )
        )
            with col:
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
   
    # Mechanical similarity cluster
    TOP_N = 10
    mech_features_available = [f for f in mechanical_features if f in df.columns]
    if len(mech_features_available) >= 2 and name_col in df.columns:
        df_mech = df.dropna(subset=mech_features_available + [name_col]).copy()
        if season_col:
            season_ctx = player_season_selected if player_season_selected is not None else (season_selected_global if 'season_selected_global' in locals() else None)
            if season_ctx is not None:
                try:
                    df_mech = df_mech[df_mech[season_col] == season_ctx].copy()
                except Exception:
                    pass

        df_mech = df_mech.reset_index(drop=True)

        try:
            if player_select in df_mech[name_col].values and len(df_mech) > 1:
                scaler = StandardScaler()
                try:
                    X_numeric = df_mech[mech_features_available].apply(pd.to_numeric, errors='coerce')
                    col_means = X_numeric.mean().fillna(0.0)
                    X_numeric = X_numeric.fillna(col_means)
                    X_scaled = scaler.fit_transform(X_numeric)
                except Exception:
                    X_tmp = df_mech[mech_features_available].apply(pd.to_numeric, errors='coerce').fillna(0.0)
                    X_scaled = scaler.fit_transform(X_tmp)

                similarity_matrix = cosine_similarity(X_scaled)

                player_positions = df_mech.index[df_mech[name_col] == player_select].tolist()
                if not player_positions:
                    st.info("Player not present in mechanical dataset for the selected season/context.")
                else:
                    player_pos = player_positions[0]
                    sim_arr = similarity_matrix[player_pos]
                    sim_series_pos = pd.Series(sim_arr, index=df_mech.index)
                    sim_series_pos.iloc[player_pos] = np.nan
                    similar_pos = sim_series_pos.sort_values(ascending=False).dropna().head(TOP_N)
                    if similar_pos.empty:
                        st.info("No mechanically similar players found (or not enough data).")
                    else:
                        sim_rows = []
                        for pos, score in similar_pos.items():
                            try:
                                sim_row = df_mech.loc[pos]
                            except Exception:
                                continue
                            sim_name = sim_row[name_col]
                            if "id" in sim_row and pd.notnull(sim_row["id"]):
                                try:
                                    sim_id = str(int(sim_row["id"]))
                                    sim_headshot_url = f"https://img.mlbstatic.com/mlb-photos/image/upload/v1/people/{sim_id}/headshot/silo/current.png"
                                except Exception:
                                    sim_headshot_url = "https://img.mlbstatic.com/mlb-photos/image/upload/v1/people/0/headshot/silo/current.png"
                            else:
                                sim_headshot_url = "https://img.mlbstatic.com/mlb-photos/image/upload/v1/people/0/headshot/silo/current.png"
                            try:
                                sim_score = float(score)
                            except Exception:
                                sim_score = 0.0
                            sim_season = None
                            if season_col and season_col in sim_row.index:
                                try:
                                    sim_season = sim_row[season_col]
                                except Exception:
                                    sim_season = None
                            sim_rows.append({
                                "name": sim_name,
                                "headshot_url": sim_headshot_url,
                                "score": sim_score,
                                "season": sim_season,
                                "pos": int(pos)
                            })

                        st.markdown(
                            """
                            <style>
                            .sim-container { width: 100%; max-width: 1160px; margin: 12px auto 10px auto; display: flex; flex-direction: column; align-items: center; }
                            .sim-list { width: 100%; display: flex; flex-direction: column; gap: 10px; align-items: center; }
                            .sim-item { display: flex; align-items: center; background: #ffffff; border-radius: 12px; padding: 10px 14px; gap: 12px; width: 100%; border: 1px solid #eef4f8; box-shadow: 0 4px 12px rgba(16,24,40,0.04); }
                            .sim-rank { font-size: 1em; font-weight: 700; color: #183153; min-width: 36px; text-align: center; }
                            .sim-headshot-compact { height: 48px; width: 48px; border-radius: 8px; object-fit: cover; box-shadow: 0 1px 6px rgba(0,0,0,0.06); }
                            .sim-name-compact { flex: 1; font-size: 1em; color: #183153; }
                            .sim-score-compact { font-size: 0.98em; font-weight: 700; color: #333; margin-right: 12px; min-width: 72px; text-align: right; }
                            .sim-bar-mini { width: 220px; height: 10px; background: #f4f7fa; border-radius: 999px; overflow: hidden; margin-left: 8px; }
                            .sim-bar-fill { height: 100%; border-radius: 999px; transition: width 0.5s ease; }
                            .sim-compare-btn { background: #ffffff; color: #000000; padding: 8px 12px; border-radius: 10px; text-decoration: none; font-weight: 800; border: 1px solid #d1d5db; cursor: pointer; }
                            .sim-compare-btn:hover { background: #f3f4f6; transform: translateY(-1px); }
                            @media (max-width: 1100px) { .sim-container { max-width: 92%; } .sim-bar-mini { width: 160px; height: 8px; } .sim-headshot-compact { height: 40px; width: 40px; } }
                            </style>
                            """,
                            unsafe_allow_html=True
                        )

                        st.markdown(
                            f"""
                            <h3 style="text-align:center; margin-top:6px; font-size:1.08em; color:#183153; letter-spacing:0.01em;">
                                Top {TOP_N} mechanically similar players to {player_title}
                            </h3>
                            <div style="text-align:center; color:#6b7280; margin-bottom:6px; font-size:0.95em;">
                                Cosine Similarity
                            </div>
                            """,
                            unsafe_allow_html=True
                        )


                        for idx, sim in enumerate(sim_rows, 1):
                            pct = max(0.0, min(1.0, float(sim['score'])))
                            width_pct = int(round(pct * 100))
                            start_color = "#D32F2F"
                            end_color = "#FFB648"
                            sim_pct_text = f"{pct:.1%}"

                            seasonA_param = player_season_selected if player_season_selected is not None else (season_selected_global if 'season_selected_global' in locals() else None)
                            seasonB_param = sim.get("season", "")
                            href_compare = f"?playerA={quote(player_select)}&playerB={quote(sim['name'])}&page=Compare"
                            if seasonA_param:
                                href_compare += f"&seasonA={quote(str(seasonA_param))}"
                            if seasonB_param:
                                href_compare += f"&seasonB={quote(str(seasonB_param))}"

                            href_player_link = f"?player={quote(sim['name'])}&page=Player"
                            if sim.get("season", None):
                                href_player_link += f"&season={quote(str(sim['season']))}"
                            onclick_player = f"window.history.pushState(null,'','{href_player_link}'); setTimeout(()=>window.location.reload(),30); return false;"
                            onclick_compare = f"window.history.pushState(null,'','{href_compare}'); setTimeout(()=>window.location.reload(),30); return false;"

                            st.markdown(
                                f"""
                                <div class="sim-item">
                                    <div class="sim-rank">{idx}</div>
                                    <img src="{sim['headshot_url']}" class="sim-headshot-compact" alt="headshot"/>
                                    <div class="sim-name-compact">
                                        <a href="{href_player_link}" onclick="{onclick_player}" style="color:#183153;text-decoration:none;font-weight:700;">{sim['name']}</a>
                                        <div style="color:#64748b;font-size:0.86em;">{f'Season: {sim["season"]}' if sim.get("season") else ''}</div>
                                    </div>
                                    <div style="display:flex;align-items:center;gap:8px;">
                                        <div class="sim-score-compact">{sim_pct_text}</div>
                                        <div class="sim-bar-mini" aria-hidden="true">
                                            <div class="sim-bar-fill" style="width:{width_pct}%; background: linear-gradient(90deg, {start_color}, {end_color});"></div>
                                        </div>
                                        <a class="sim-compare-btn" href="{href_compare}" onclick="{onclick_compare}">Compare</a>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        st.markdown('</div></div>', unsafe_allow_html=True)

                        with st.expander("Mechanical similarity cluster (click to expand)", expanded=False):
                            try:
                                heat_positions = [player_pos] + [int(p) for p in similar_pos.index.tolist()]
                                heat_positions = list(dict.fromkeys(heat_positions))
                                if len(heat_positions) < 2:
                                    st.info("Not enough data to build cluster heatmap.")
                                else:
                                    heat_mat = similarity_matrix[np.ix_(heat_positions, heat_positions)]
                                    heat_names = []
                                    for p in heat_positions:
                                        try:
                                            heat_names.append(df_mech.iloc[int(p)][name_col])
                                        except Exception:
                                            heat_names.append(f"Player_{p}")

                                    num_players = len(heat_positions)
                                    fig_width = min(10, max(6, num_players * 0.6))
                                    fig_height = min(8, max(5, num_players * 0.5))
                                    fig_h, axh = plt.subplots(figsize=(fig_width, fig_height))

                                    sns.heatmap(
                                        heat_mat,
                                        xticklabels=heat_names,
                                        yticklabels=heat_names,
                                        cmap="coolwarm",
                                        vmin=0.0,
                                        vmax=1.0,
                                        annot=True,
                                        fmt=".2f",
                                        annot_kws={"fontsize": 9},
                                        square=True,
                                        linewidths=0.5,
                                        linecolor='gray',
                                        cbar_kws={"label": "Cosine Similarity", "shrink": 0.8},
                                        ax=axh
                                    )

                                    axh.set_title(
                                        f"Mechanical Similarity Cluster: {player_title}",
                                        fontsize=14,
                                        fontweight='bold',
                                        pad=12
                                    )

                                    axh.set_xticklabels(heat_names, rotation=45, ha='right', fontsize=9)
                                    axh.set_yticklabels(heat_names, rotation=0, fontsize=9)

                                    plt.tight_layout()
                                    st.pyplot(fig_h)
                                    plt.close(fig_h)
                            except Exception as e:
                                st.info(f"Could not render cluster heatmap: {str(e)}")
            else:
                st.info("Player not present in mechanical dataset for the selected season/context.")
        except Exception as e:
            st.info(f"Not enough mechanical data for this player/season to compute similarities: {str(e)}")

# ---------------- Compare tab ----------------
elif page == "Compare":

    mech_features_available = [f for f in mechanical_features if f in df.columns]

    st.markdown(
        """
        <h2 style="text-align:center; margin-top:1.2em; margin-bottom:0.6em; font-size:1.6em; letter-spacing:0.01em; color:#2a3757;">
            Compare Players
        </h2>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # Replace the player dropdowns block in the Compare tab with this snippet.
    player_options = sorted(df["Name"].dropna().unique())
    if not player_options:
        st.warning("No players available.")
        st.stop()
    
    # determine sensible defaults (respect query params if present)
    default_a = None
    try:
        if 'qp_player' in locals() and qp_player and qp_player in player_options:
            default_a = qp_player
        elif 'qp_player_b' in locals() and qp_player_b and qp_player_b in player_options:
            # prefer explicit playerA param, otherwise leave for later
            pass
    except Exception:
        pass
    if default_a is None:
        default_a = player_options[0]
    idx_a = player_options.index(default_a)
    
    default_b = None
    try:
        if 'qp_player_b' in locals() and qp_player_b and qp_player_b in player_options:
            default_b = qp_player_b
    except Exception:
        pass
    if default_b is None:
        # pick a different default for B when possible
        if len(player_options) > 1:
            default_b = player_options[1] if idx_a == 0 else player_options[0]
        else:
            default_b = player_options[0]
    idx_b = player_options.index(default_b)
    
    # callback to ensure player B is never the same as player A after changes
    def _ensure_player_b_not_a():
        a = st.session_state.get("compare_player_a")
        b = st.session_state.get("compare_player_b")
        if a == b:
            for opt in player_options:
                if opt != a:
                    st.session_state["compare_player_b"] = opt
                    break
    
    colA, colB = st.columns(2)
    
    with colA:
        playerA = st.selectbox("", player_options, index=idx_a, key="compare_player_a", on_change=_ensure_player_b_not_a)
        seasonsA = sorted(df[df["Name"] == playerA][season_col].dropna().unique()) if season_col else []
        seasonA = st.selectbox("Season A", seasonsA, index=len(seasonsA)-1 if seasonsA else 0)
    
    with colB:
        playerB = st.selectbox("", player_options, index=idx_b, key="compare_player_b")
        seasonsB = sorted(df[df["Name"] == playerB][season_col].dropna().unique()) if season_col else []
        seasonB = st.selectbox("Season B", seasonsB, index=len(seasonsB)-1 if seasonsB else 0)

    # -------------------------------------
    # Row extractor
    # -------------------------------------
    def get_row(name, season):
        subset = df[(df["Name"] == name) & (df[season_col] == season)]
        return subset.iloc[0] if len(subset) else df[df["Name"] == name].iloc[0]

    rowA = get_row(playerA, seasonA)
    rowB = get_row(playerB, seasonB)

    # -------------------------------------
    # Cosine similarity
    # -------------------------------------
    try:
        feats_sim = mech_features_available
        df_sim = df.dropna(subset=feats_sim)

        scaler, df_scaled = get_scaler_and_scaled_df(feats_sim, df_sim)
        vecA = scaler.transform(pd.DataFrame([rowA[feats_sim].astype(float)]))[0]
        vecB = scaler.transform(pd.DataFrame([rowB[feats_sim].astype(float)]))[0]

        cosine_sim = compute_cosine_similarity_between_rows(vecA, vecB)
        sim_pct = f"{cosine_sim*100:.1f}%"
    except Exception:
        cosine_sim = None
        sim_pct = "N/A"

    # -------------------------------------
    # Player cards
    # -------------------------------------
    col1, colSim, col2 = st.columns([1.5, 1, 1.5])

    def player_card(row, name, season):
        pid = str(int(row.get("id", 0))) if pd.notnull(row.get("id")) else "0"
        img = f"https://img.mlbstatic.com/mlb-photos/image/upload/v1/people/{pid}/headshot/silo/current.png"
        logo = image_dict.get(row.get("Team", ""), "")

        st.markdown(f"""
            <div style="text-align:center;">
                <img src="{img}" style="height:110px;width:110px;border-radius:12px;border:1px solid #D1D5DB;">
                <div style="font-size:1.05em;font-weight:800;color:#1F2937;margin-top:6px;">{name} ({season})</div>
                {'<img src="'+logo+'" style="height:38px;margin-top:4px;">' if logo else ''}
            </div>
        """, unsafe_allow_html=True)

    with col1:
        player_card(rowA, playerA, seasonA)
    with colSim:
        st.markdown(f"""
            <div style="text-align:center;background:#F3F4F6;border:1px solid #D1D5DB;
                        padding:16px;border-radius:10px;margin-top:28px;">
                <div style="font-size:1.05em;font-weight:700;color:#374151;">Similarity</div>
                <div style="font-size:1.85em;font-weight:800;color:#111827;">{sim_pct}</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        player_card(rowB, playerB, seasonB)

    # -------------------------------------
    # Stat tiles
    # -------------------------------------
    stats = ["Age", "Swing+", "HitSkillPlus", "ImpactPlus"]
    
    # Display name mapping
    stat_labels = {
        "Age": "Age",
        "Swing+": "Swing+",
        "HitSkillPlus": "BatToBall+",
        "ImpactPlus": "Impact+"
    }
    
    colA_block, _, colB_block = st.columns([1, 0.06, 1])
    
    def stat_tiles(col, row, label):
        with col:
            st.markdown(
                f"<div style='text-align:center;font-weight:700;margin-bottom:6px;'>{label}</div>",
                unsafe_allow_html=True
            )
            cols = st.columns(len(stats))
    
            for i, stat in enumerate(stats):
                v = row.get(stat, "N/A")
    
                if stat == "Age":
                    disp = f"{int(v)}" if pd.notnull(v) else "N/A"
                else:
                    try:
                        disp = f"{v:.2f}" if isinstance(v, (int, float)) else "N/A"
                    except Exception:
                        disp = "N/A"
    
                display_name = stat_labels.get(stat, stat)
    
                cols[i].markdown(f"""
                    <div style="background:#FFF;border:1px solid #D1D5DB;border-radius:10px;padding:10px;text-align:center;">
                        <div style="font-weight:700;">{disp}</div>
                        <div style="font-size:0.75em;color:#6B7280;">{display_name}</div>
                    </div>
                """, unsafe_allow_html=True)
    
    stat_tiles(colA_block, rowA, "Player A")
    stat_tiles(colB_block, rowB, "Player B")
    
    st.markdown("<hr style='margin-top:32px;margin-bottom:22px;'/>", unsafe_allow_html=True)


    # ------------------------ Mechanical comparison, table, SHAP charts ------------------------
    if len(mech_features_available) >= 2:

        feats = mech_features_available

        # build df_comp robustly using supplied seasons if present
        seasons = []
        try:
            if seasonA is not None:
                seasons.append(seasonA)
        except NameError:
            pass
        try:
            if seasonB is not None:
                seasons.append(seasonB)
        except NameError:
            pass

        if season_col and seasons:
            df_comp = df[df[season_col].isin(seasons)].dropna(subset=feats + ["Name"])
        else:
            df_comp = df.dropna(subset=feats + ["Name"])

        if df_comp.empty:
            st.warning("Not enough data for mechanical comparison.")
            st.stop()

        mean_series = df_comp[feats].mean()
        std_series = df_comp[feats].std().replace(0, 1e-9)

        valsA = rowA[feats].astype(float)
        valsB = rowB[feats].astype(float)
        zA = (valsA - mean_series) / std_series
        zB = (valsB - mean_series) / std_series
        z_diff = abs(zA - zB)

        pct_rank = df_comp[feats].rank(pct=True)

        def get_pct_for_row(pct_df, row):
            # try by exact index first
            try:
                if row.name in pct_df.index:
                    return pct_df.loc[row.name]
            except Exception:
                pass
            # try by matching name and season
            try:
                mask = (df_comp["Name"] == row["Name"])
                if season_col in df_comp.columns and season_col in row.index:
                    mask = mask & (df_comp[season_col] == row.get(season_col, None))
                subset = pct_df[mask]
                if len(subset) > 0:
                    return subset.iloc[0]
            except Exception:
                pass
            # fallback to average percentiles (or first row)
            try:
                return pct_df.iloc[0]
            except Exception:
                return pd.Series(0.5, index=pct_df.columns)

        pctA = get_pct_for_row(pct_rank, rowA)
        pctB = get_pct_for_row(pct_rank, rowB)

        # SHAP per-player
        shapA, _, _ = compute_shap(rowA, feats)
        shapB, _, _ = compute_shap(rowB, feats)
        shapA = shapA.reindex(feats).fillna(0) if shapA is not None else pd.Series(0, index=feats)
        shapB = shapB.reindex(feats).fillna(0) if shapB is not None else pd.Series(0, index=feats)

        # Importance: prefer SHAP sample mean, fallback to z-based importance, then uniform
        use_shap = False
        importance = None
        if model_loaded and explainer is not None:
            try:
                sampleX = df_comp[feats].head(200).fillna(df_comp[feats].mean())
                sample_shap = explainer(sampleX)
                if hasattr(sample_shap, "values"):
                    mean_abs_shap = abs(sample_shap.values).mean(axis=0)
                    importance = pd.Series(mean_abs_shap, index=feats)
                    use_shap = True
            except Exception:
                use_shap = False

        if not use_shap:
            # data-based fallback using zA/zB (only valid inside Compare)
            if 'zA' in locals() and 'zB' in locals():
                importance = (abs(zA) + abs(zB)).replace(0, 1e-9)
                # normalize
                total = importance.sum()
                if total == 0:
                    importance = pd.Series(1.0 / len(feats), index=feats)
                else:
                    importance = importance / total
            else:
                importance = pd.Series(1.0 / len(feats), index=feats)

        # ensure importance is a Series indexed by feats
        importance = importance.reindex(feats).fillna(0)

        # -------------------------------------------------
        # QUICK TAKEAWAYS
        # -------------------------------------------------
        st.markdown(
            """
            <h3 style="text-align:center; margin-top:6px; font-size:1.08em; color:#183153; letter-spacing:0.01em;">
                Quick Takeaways
            </h3>
            <div style="text-align:center; color:#6b7280; margin-bottom:6px; font-size:0.95em;">
                Main features that make Player A & Player B similar and different
            </div>
            """,
            unsafe_allow_html=True
        )

        weighted = (1 - (z_diff / (z_diff.max() + 1e-9))).clip(0,1) * importance
        top_sim = weighted.sort_values(ascending=False).head(3).index.tolist()
        top_diff = (z_diff * importance).sort_values(ascending=False).head(3).index.tolist()

        if cosine_sim is not None:
            st.markdown(f"- **Overall mechanical similarity:** {cosine_sim*100:.1f}%")
        for f in top_sim:
            st.markdown(f"- **Similarity driver:** {FEATURE_LABELS.get(f,f)}")
        for f in top_diff:
            st.markdown(f"- **Difference driver:** {FEATURE_LABELS.get(f,f)}")

        st.markdown(
            """
            <h3 style="text-align:center; margin-top:6px; font-size:1.08em; color:#183153; letter-spacing:0.01em;">
                Swing+ Feature Contributions
            </h3>
            <div style="text-align:center; color:#6b7280; margin-bottom:6px; font-size:0.95em;">
                How each mechanical feature moved the model's Swing+ prediction for this player.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("""
        <style>
        .comp-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 18px;
            font-size: 0.88em;
            background: #FFFFFF;
            border: 2px solid #111827;
            border-radius: 10px;
            overflow: hidden;
        }

        .comp-table th {
            background: #F3F4F6;
            color: #374151;
            padding: 10px 6px;
            font-weight: 700;
            text-align: center;
            border-bottom: 1px solid #D1D5DB;
        }

        .comp-table td {
            padding: 9px 6px;
            text-align: center;
            border-bottom: 1px solid #E5E7EB;
            color: #111827;
        }

        .comp-table tr:last-child td {
            border-bottom: 1px solid #E5E7EB;
        }

        .comp-feature {
            text-align: left;
            font-weight: 600;
            color: #1F2937;
        }
        </style>
        """, unsafe_allow_html=True)

        # ==========================================
        # BUILD TABLE
        # ==========================================
        html_rows = ""
        for f in feats:
            html_rows += (
                "<tr>"
                f"<td class='comp-feature'>{FEATURE_LABELS.get(f, f)}</td>"
                f"<td>{valsA[f]:.2f}</td>"
                f"<td>{valsB[f]:.2f}</td>"
                f"<td>{(valsA[f]-valsB[f]):.2f}</td>"
                f"<td>{z_diff[f]:.2f}</td>"
                f"<td>{pctA[f]:.0%}</td>"
                f"<td>{pctB[f]:.0%}</td>"
                f"<td>{importance[f]:.3f}</td>"
                "</tr>"
            )

        html_table = (
        f"<table class='comp-table'>"
        f"<thead>"
        f"<tr>"
        f"<th>Feature</th>"
        f"<th>{playerA} ({seasonA})</th>"
        f"<th>{playerB} ({seasonB})</th>"
        f"<th>Diff</th>"
        f"<th>Z-Diff</th>"
        f"<th>Pct A</th>"
        f"<th>Pct B</th>"
        f"<th>Importance</th>"
        f"</tr>"
        f"</thead>"
        f"<tbody>{html_rows}</tbody>"
        f"</table>"
        )

        st.markdown(html_table, unsafe_allow_html=True)

        # Title + subtitle (styled like Player tab) and SHAP plots for Compare tab
        st.markdown(
            f"""
            <h3 style="text-align:center; margin-top:6px; font-size:1.08em; color:#183153; letter-spacing:0.01em;">
                Model Contributions
            </h3>
            <div style="text-align:center; color:#6b7280; margin-bottom:6px; font-size:0.95em;">
                SHAP contributions for {playerA} ({seasonA}) and {playerB} ({seasonB})
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Ensure ordering by importance (desc) so plots align with Player tab behavior
        order = importance.sort_values(ascending=False).index.tolist()
        shapA_ord = shapA.reindex(order).fillna(0)
        shapB_ord = shapB.reindex(order).fillna(0)
        labels = [FEATURE_LABELS.get(f, f) for f in order]
        
        # importance percentages for hover/text
        imp_pct = importance.reindex(order).fillna(0)
        if imp_pct.sum() == 0:
            imp_pct_vals = [0.0] * len(imp_pct)
        else:
            imp_pct_vals = (imp_pct / imp_pct.sum()).values.astype(float)
        
        colA_shap, colB_shap = st.columns(2)
        
        with colA_shap:
            fig = go.Figure()
            vals = shapA_ord.values.astype(float)
            colors = ["#D8573C" if v > 0 else "#3B82C4" for v in vals]  # Player A: red/blue
            text_labels = [f"{v:.3f}  ({p:.0%})" for v, p in zip(vals, imp_pct_vals)]
            hover_text = [f"Contribution: {v:.3f}<br>Importance: {p:.0%}" for v, p in zip(vals, imp_pct_vals)]
            fig.add_trace(go.Bar(
                x=vals,
                y=labels,
                orientation='h',
                marker_color=colors,
                hoverinfo='text',
                hovertext=hover_text,
                text=text_labels,
                textposition='inside',
                insidetextanchor='middle'
            ))
            fig.update_layout(
                margin=dict(l=160, r=24, t=12, b=60),
                height=430,
                showlegend=False,
                xaxis_title="SHAP contribution",
                yaxis=dict(autorange="reversed"),
                font=dict(size=11)
            )
            st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True, "displayModeBar": False})
        
        with colB_shap:
            fig = go.Figure()
            vals = shapB_ord.values.astype(float)
            colors = ["#F59E0B" if v > 0 else "#60A5FA" for v in vals]  # Player B: amber/light-blue
            text_labels = [f"{v:.3f}  ({p:.0%})" for v, p in zip(vals, imp_pct_vals)]
            hover_text = [f"Contribution: {v:.3f}<br>Importance: {p:.0%}" for v, p in zip(vals, imp_pct_vals)]
            fig.add_trace(go.Bar(
                x=vals,
                y=labels,
                orientation='h',
                marker_color=colors,
                hoverinfo='text',
                hovertext=hover_text,
                text=text_labels,
                textposition='inside',
                insidetextanchor='middle'
            ))
            fig.update_layout(
                margin=dict(l=160, r=24, t=12, b=60),
                height=430,
                showlegend=False,
                xaxis_title="SHAP contribution",
                yaxis=dict(autorange="reversed"),
                font=dict(size=11)
            )
            st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True, "displayModeBar": False})
