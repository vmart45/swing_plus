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

st.set_page_config(
    page_title="Swing+ & HitSkill+ Dashboard",
    page_icon="⚾",
    layout="wide"
)

DATA_PATH = "Main.csv"
MODEL_PATH = "SwingPlus.pkl"

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

# Normalize Name to "First Last" if possible
def normalize_name(name):
    try:
        if not isinstance(name, str):
            return name
        name = name.strip()
        # If already "First Last" (no comma) just normalize whitespace
        if "," in name:
            parts = [p.strip() for p in name.split(",", 1)]
            return f"{parts[1]} {parts[0]}".strip()
        return " ".join(name.split())
    except Exception:
        return name

if "Name" in df.columns:
    df["Name"] = df["Name"].apply(normalize_name)

# Check core columns
expected_core = ["Name", "Age", "Swing+", "HitSkillPlus", "ImpactPlus"]
missing_core = [c for c in expected_core if c not in df.columns]
if missing_core:
    st.error(f"Missing required columns from data: {missing_core}")
    st.stop()

# Team logos
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

# Friendly feature labels
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

# Sidebar filters: Player name first, then age range (per user request)
st.sidebar.header("Filters")
search_name = st.sidebar.text_input("Search Player by Name")

min_age, max_age = int(df["Age"].min()), int(df["Age"].max())
age_range = st.sidebar.slider("Age Range", min_age, max_age, (min_age, max_age))

# Season filter auto-default to 2025 if present
season_col = None
for c in ["year", "Year", "season"]:
    if c in df.columns:
        season_col = c
        break

season_selected = None
if season_col:
    unique_years = sorted(df[season_col].dropna().unique())
    default_season = 2025 if 2025 in unique_years else (unique_years[-1] if unique_years else None)
    if default_season is None:
        season_selected = st.sidebar.selectbox("Season", unique_years) if unique_years else None
    else:
        season_selected = st.sidebar.selectbox("Season", unique_years, index=unique_years.index(default_season) if default_season in unique_years else 0)

# Competitive swings filter (keep, default start 100)
comp_col = None
for c in ["swings_competitive", "competitive_swings", "competitive_swings"]:
    if c in df.columns:
        comp_col = c
        break

if comp_col:
    try:
        swings_min = int(df[comp_col].min())
        swings_max = int(df[comp_col].max())
        default_low = 100 if swings_max >= 100 else swings_min
        swings_range = st.sidebar.slider("Competitive Swings", swings_min, swings_max, (default_low, swings_max))
    except Exception:
        swings_range = None
else:
    swings_range = None

# Build filtered df
df_filtered = df.copy()
if search_name:
    df_filtered = df_filtered[df_filtered["Name"].str.contains(search_name, case=False, na=False)]

df_filtered = df_filtered[(df_filtered["Age"] >= age_range[0]) & (df_filtered["Age"] <= age_range[1])]

if season_col and season_selected is not None:
    try:
        df_filtered = df_filtered[df_filtered[season_col] == season_selected]
    except Exception:
        pass

if comp_col and swings_range:
    try:
        df_filtered = df_filtered[
            (df_filtered[comp_col] >= swings_range[0]) &
            (df_filtered[comp_col] <= swings_range[1])
        ]
    except Exception:
        pass

# Note: per user request, remove the batted_ball_events filter (do not include slider),
# but keep the column in the table and round for display.

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

# Load model and SHAP explainer (optional)
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
def get_scaler_and_scaled_df(features):
    scaler = StandardScaler()
    X = df[features].astype(float)
    X_scaled = scaler.fit_transform(X)
    df_scaled = pd.DataFrame(X_scaled, columns=features, index=df.index)
    return scaler, df_scaled

def compute_cosine_similarity_between_rows(vecA, vecB):
    sim = np.dot(vecA, vecB) / (np.linalg.norm(vecA) * np.linalg.norm(vecB) + 1e-12)
    return float(sim)

def safe_rank_column(df, col):
    ranks = df[col].rank(ascending=False, method="min")
    filled = ranks.fillna(len(df) + 1).astype(int)
    return filled

# Page navigation
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

def open_compare_in_same_tab(playerA, playerB):
    try:
        st.experimental_set_query_params(playerA=playerA, playerB=playerB, page="Compare")
    except Exception:
        try:
            st.experimental_set_query_params(player=playerA, playerB=playerB, page="Compare")
        except Exception:
            pass

# ---------------- Main tab ----------------
if page == "Main":
    st.markdown("<h2 style='text-align:center; margin-top:1.2em; margin-bottom:0.6em; font-size:1.6em; color:#2a3757;'>Player Metrics Table</h2>", unsafe_allow_html=True)

    # Build display columns with plus stats moved directly after counts (PA, Competitive Swings, Batted Ball Events)
    all_stats = []

    # Always show basic columns first
    all_stats.extend(["Name", "Team"])
    # season if present
    if "year" in df_filtered.columns:
        all_stats.append("year")
    # Counts: pa, competitive swings, batted ball events (round these later)
    for c in ["pa", comp_col if comp_col else None, "batted_ball_events"]:
        if c and c in df_filtered.columns and c not in all_stats:
            all_stats.append(c)

    # Move plus stats to front after counts
    for c in ["Swing+", "HitSkillPlus", "ImpactPlus"]:
        if c in df_filtered.columns and c not in all_stats:
            all_stats.append(c)

    # Then add remaining mechanical and other useful columns, excluding removed ones
    remaining = [
        "swing_length", "avg_bat_speed", "swing_tilt", "attack_angle", "attack_direction",
        "avg_intercept_y_vs_plate", "avg_intercept_y_vs_batter", "avg_batter_y_position", "avg_batter_x_position",
        "avg_foot_sep", "avg_stance_angle"
    ]
    for c in remaining:
        if c in df_filtered.columns and c not in all_stats:
            all_stats.append(c)

    # Remove disallowed/removed columns entirely from display if present
    # User requested removal of: bip, batter_run_value, est_ba, est_slg, est_woba, xwOBA_pred, xba_pred, xslg_pred, side (BT)
    removed_cols = ["bip", "batter_run_value", "est_ba", "est_slg", "est_woba", "xwOBA_pred", "xba_pred", "xslg_pred", "side"]
    # Ensure removed columns are not included
    display_cols = [c for c in all_stats if c not in removed_cols]

    display_df = df_filtered[display_cols].copy()

    # Round counts to 0 decimals (pa, competitive swings, batted ball events)
    for c in ["pa", comp_col if comp_col else None, "batted_ball_events"]:
        if c and c in display_df.columns:
            try:
                display_df[c] = display_df[c].round(0).astype("Int64")
            except Exception:
                try:
                    display_df[c] = display_df[c].round(0).astype(int)
                except Exception:
                    pass

    # Round age to 0 decimals
    if "Age" in display_df.columns:
        try:
            display_df["Age"] = display_df["Age"].round(0).astype("Int64")
        except Exception:
            try:
                display_df["Age"] = display_df["Age"].round(0).astype(int)
            except Exception:
                pass

    # Rename friendly labels
    rename_map = {}
    if "year" in display_df.columns:
        rename_map["year"] = "Season"
    if "pa" in display_df.columns:
        rename_map["pa"] = "PA"
    if comp_col and comp_col in display_df.columns:
        rename_map[comp_col] = "Competitive Swings"
    if "batted_ball_events" in display_df.columns:
        rename_map["batted_ball_events"] = "Batted Ball Events"
    # Plus shorthand
    if "Swing+" in display_df.columns:
        rename_map["Swing+"] = "Swing+"
    if "HitSkillPlus" in display_df.columns:
        rename_map["HitSkillPlus"] = "HitSkill+"
    if "ImpactPlus" in display_df.columns:
        rename_map["ImpactPlus"] = "Impact+"
    # mechanical labels
    for k, v in FEATURE_LABELS.items():
        if k in display_df.columns:
            rename_map[k] = v

    # Apply sort by Swing+ if present
    sort_col = "Swing+" if "Swing+" in display_df.columns else display_df.columns[0]
    styled = (
        display_df
        .rename(columns=rename_map)
        .sort_values(sort_col, ascending=False)
        .reset_index(drop=True)
    )

    # Restore coloring on plus stats
    plus_labels = []
    for p in ["Swing+", "HitSkillPlus", "ImpactPlus"]:
        if p in display_df.columns:
            plus_labels.append(rename_map.get(p, p))

    # Use pandas Styler to add background gradient only to plus stats
    try:
        styler = styled.style.format(precision=2)
        if plus_labels:
            styler = styler.background_gradient(subset=plus_labels, cmap=elite_cmap)
        st.dataframe(styler, use_container_width=True, hide_index=True)
    except Exception:
        st.dataframe(styled, use_container_width=True, hide_index=True)

    # Leaderboards (keep Swing+ and HitSkill+ top lists)
    st.markdown("<h2 style='text-align:center; margin-top:1.2em; margin-bottom:0.6em; font-size:1.6em; color:#2a3757;'>Top 10 Leaderboards</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div style="text-align:center; font-size:1.15em; font-weight:600; margin-bottom:0.6em; color:#385684;">Top 10 by Swing+</div>', unsafe_allow_html=True)
        if "Swing+" in df_filtered.columns:
            top_swing = df_filtered.sort_values("Swing+", ascending=False).head(10).reset_index(drop=True)
            leaderboard_cols = [c for c in ["Name", "Team", "Age", "Swing+", "HitSkillPlus", "ImpactPlus"] if c in top_swing.columns]
            top_swing_display = top_swing.copy()
            for col in ["Swing+", "HitSkillPlus", "ImpactPlus"]:
                if col in top_swing_display.columns:
                    top_swing_display[col] = top_swing_display[col].apply(lambda v: f"{v:.2f}" if pd.notna(v) else v)
            if "Age" in top_swing_display.columns:
                try:
                    top_swing_display["Age"] = top_swing_display["Age"].round(0).astype("Int64")
                except Exception:
                    pass
            # Rename columns for display and apply gradient anchored to global values so top-10 shows dark red for highest scores
            top_swing_renamed = top_swing_display.rename(columns=rename_map)
            display_cols_renamed = [rename_map.get(c, c) for c in leaderboard_cols]
            swing_label = rename_map.get("Swing+", "Swing+")
            try:
                vmin = float(df_filtered["Swing+"].min())
                vmax = float(df_filtered["Swing+"].max())
                st.dataframe(
                    top_swing_renamed[display_cols_renamed]
                    .style.background_gradient(subset=[swing_label], cmap=elite_cmap, vmin=vmin, vmax=vmax),
                    use_container_width=True,
                    hide_index=True
                )
            except Exception:
                st.dataframe(top_swing_renamed[display_cols_renamed], use_container_width=True, hide_index=True)
        else:
            st.info("Swing+ not present in dataset; leaderboard unavailable.")

    with col2:
        st.markdown('<div style="text-align:center; font-size:1.15em; font-weight:600; margin-bottom:0.6em; color:#385684;">Top 10 by HitSkill+</div>', unsafe_allow_html=True)
        if "HitSkillPlus" in df_filtered.columns:
            top_hit = df_filtered.sort_values("HitSkillPlus", ascending=False).head(10).reset_index(drop=True)
            leaderboard_cols = [c for c in ["Name", "Team", "Age", "HitSkillPlus", "Swing+", "ImpactPlus"] if c in top_hit.columns]
            top_hit_display = top_hit.copy()
            for col in ["Swing+", "HitSkillPlus", "ImpactPlus"]:
                if col in top_hit_display.columns:
                    top_hit_display[col] = top_hit_display[col].apply(lambda v: f"{v:.2f}" if pd.notna(v) else v)
            if "Age" in top_hit_display.columns:
                try:
                    top_hit_display["Age"] = top_hit_display["Age"].round(0).astype("Int64")
                except Exception:
                    pass
            # Rename and apply gradient anchored to global HitSkillPlus range
            top_hit_renamed = top_hit_display.rename(columns=rename_map)
            display_cols_hit_renamed = [rename_map.get(c, c) for c in leaderboard_cols]
            hit_label = rename_map.get("HitSkillPlus", "HitSkillPlus")
            try:
                vmin_h = float(df_filtered["HitSkillPlus"].min())
                vmax_h = float(df_filtered["HitSkillPlus"].max())
                st.dataframe(
                    top_hit_renamed[display_cols_hit_renamed]
                    .style.background_gradient(subset=[hit_label], cmap=elite_cmap, vmin=vmin_h, vmax=vmax_h),
                    use_container_width=True,
                    hide_index=True
                )
            except Exception:
                st.dataframe(top_hit_renamed[display_cols_hit_renamed], use_container_width=True, hide_index=True)
        else:
            st.info("HitSkillPlus not present in dataset; leaderboard unavailable.")

# ---------------- Player tab ----------------
elif page == "Player":
    st.markdown("<h2 style='text-align:center; margin-top:1.2em; margin-bottom:0.6em; font-size:1.6em; color:#2a3757;'>Player Detail</h2>", unsafe_allow_html=True)

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

    player_select = st.selectbox("Select a Player", player_options, key="player_select", index=default_index)
    player_row = df[df["Name"] == player_select].iloc[0]

    headshot_size = 96
    logo_size = 80

    headshot_html = ""
    if "id" in player_row and pd.notnull(player_row["id"]):
        try:
            player_id = str(int(player_row["id"]))
            headshot_url = f"https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/w_640,q_auto:best/v1/people/{player_id}/headshot/silo/current.png"
        except Exception:
            headshot_url = "https://img.mlbstatic.com/mlb-photos/image/upload/v1/people/0/headshot/silo/current.png"
        headshot_html = f'<img src="{headshot_url}" style="height:{headshot_size}px;width:{headshot_size}px;object-fit:cover;border-radius:14px;margin-right:18px;" alt="headshot"/>'
    else:
        fallback_url = "https://img.mlbstatic.com/mlb-photos/image/upload/v1/people/0/headshot/silo/current.png"
        headshot_html = f'<img src="{fallback_url}" style="height:{headshot_size}px;width:{headshot_size}px;object-fit:cover;border-radius:14px;margin-right:18px;" alt="headshot"/>'

    player_name_html = f'<span style="font-size:2.3em;font-weight:800;color:#183153;margin:0 20px;">{player_select}</span>'

    team_logo_html = ""
    if "Team" in player_row and pd.notnull(player_row["Team"]):
        team_abbr = str(player_row["Team"]).strip()
        team_logo_url = image_dict.get(team_abbr, "")
        if team_logo_url:
            team_logo_html = f'<div style="margin-left:14px; display:flex; align-items:center;"><img src="{team_logo_url}" style="height:{logo_size}px;width:{logo_size}px;border-radius:8px;object-fit:cover;"></div>'

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
    # Use safe ranking helper to avoid IntCastingNaNError
    if "Swing+" in df.columns:
        df["Swing+_rank"] = safe_rank_column(df, "Swing+")
    if "HitSkillPlus" in df.columns:
        df["HitSkillPlus_rank"] = safe_rank_column(df, "HitSkillPlus")
    if "ImpactPlus" in df.columns:
        df["ImpactPlus_rank"] = safe_rank_column(df, "ImpactPlus")

    p_swing_rank = int(df.loc[df["Name"] == player_select, "Swing+_rank"].iloc[0]) if "Swing+_rank" in df.columns else None
    p_hit_rank = int(df.loc[df["Name"] == player_select, "HitSkillPlus_rank"].iloc[0]) if "HitSkillPlus_rank" in df.columns else None
    p_impact_rank = int(df.loc[df["Name"] == player_select, "ImpactPlus_rank"].iloc[0]) if "ImpactPlus_rank" in df.columns else None

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
    hit_color = plus_color_by_rank(p_hit_rank, total_players)
    impact_color = plus_color_by_rank(p_impact_rank, total_players)

    st.markdown(
        f"""
        <div style="display: flex; justify-content: center; gap: 32px; margin-top: 0px; margin-bottom: 28px;">
          <div style="background: #fff; border-radius: 16px; padding: 24px 32px; text-align: center; min-width: 160px;">
            <div style="font-size: 2.2em; font-weight: 700; color: {swing_color};">{player_row['Swing+']:.2f}</div>
            <div style="font-size: 1.1em; color: #888; font-weight: 600; margin-bottom: 4px;">Swing+</div>
            <span style="background: #FFC10733; color: #B71C1C; border-radius: 10px; font-size: 0.98em; padding: 2px 10px;">Rank {p_swing_rank} of {total_players}</span>
          </div>
          <div style="background: #fff; border-radius: 16px; padding: 24px 32px; text-align: center; min-width: 160px;">
            <div style="font-size: 2.2em; font-weight: 700; color: {hit_color};">{player_row['HitSkillPlus']:.2f}</div>
            <div style="font-size: 1.1em; color: #888; font-weight: 600; margin-bottom: 4px;">HitSkill+</div>
            <span style="background: #C8E6C933; color: #1B5E20; border-radius: 10px; font-size: 0.98em; padding: 2px 10px;">Rank {p_hit_rank} of {total_players}</span>
          </div>
          <div style="background: #fff; border-radius: 16px; padding: 24px 32px; text-align: center; min-width: 160px;">
            <div style="font-size: 2.2em; font-weight: 700; color: {impact_color};">{player_row['ImpactPlus']:.2f}</div>
            <div style="font-size: 1.1em; color: #888; font-weight: 600; margin-bottom: 4px;">Impact+</div>
            <span style="background: #B3E5FC33; color: #01579B; border-radius: 10px; font-size: 0.98em; padding: 2px 10px;">Rank {p_impact_rank} of {total_players}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Rest of Player tab (SHAP, similarity, etc.) remains as before but avoiding removed columns where applicable

    mech_features_available = [f for f in mechanical_features if f in df.columns]

    shap_df = None
    shap_base = None
    shap_pred = None
    shap_values_arr = None

    if model_loaded and explainer is not None and len(mech_features_available) >= 2:
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
            shap_df = pd.DataFrame({
                "feature": X_player.columns.tolist(),
                "raw": [float(X_player.iloc[0][f]) if pd.notna(X_player.iloc[0][f]) else np.nan for f in X_player.columns],
                "shap_value": shap_values_arr
            })
            shap_df["abs_shap"] = np.abs(shap_df["shap_value"])
            total_abs = shap_df["abs_shap"].sum() if shap_df["abs_shap"].sum() != 0 else 1.0
            shap_df["pct_of_abs"] = shap_df["abs_shap"] / total_abs
            shap_df = shap_df.sort_values("abs_shap", ascending=False).reset_index(drop=True)
        except Exception:
            shap_df = None

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center; margin-top:6px; font-size:1.08em; color:#183153;'>Swing+ Feature Contributions (SHAP)</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    shap_pred_label = f"{shap_pred:.2f}" if (shap_pred is not None and not pd.isna(shap_pred)) else "N/A"
    swing_actual_label = f"{player_row['Swing+']:.2f}" if (player_row.get("Swing+") is not None and not pd.isna(player_row.get("Swing+"))) else "N/A"
    base_label = f"{shap_base:.2f}" if (shap_base is not None and not pd.isna(shap_base)) else "N/A"

    with col1:
        st.markdown(f"<div style='text-align:center;font-weight:700;color:#183153;'>Model prediction: {shap_pred_label} &nbsp; | &nbsp; Actual Swing+: {swing_actual_label}</div>", unsafe_allow_html=True)
        if not model_loaded or explainer is None or shap_df is None or len(shap_df) == 0:
            st.info("Swing+ model or SHAP explainer not available.")
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
            fig.add_trace(go.Bar(x=x_vals, y=y, orientation='h', marker_color=colors, hoverinfo='text', hovertext=[f"Contribution: {v:.3f}<br>Importance: {p:.0%}" for v, p in zip(x_vals, pct_vals)], text=text_labels, textposition='auto'))
            fig.update_layout(margin=dict(l=160, r=24, t=12, b=60), xaxis_title="SHAP contribution to Swing+ (signed)", yaxis=dict(autorange="reversed"), height=420, showlegend=False, font=dict(size=12))
            st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True, "displayModeBar": False})

    with col2:
        st.markdown(f"<div style='text-align:center;font-weight:700;color:#183153;'>Model baseline: {base_label}</div>", unsafe_allow_html=True)
        if shap_df is None or len(shap_df) == 0:
            st.write("No SHAP data to show.")
        else:
            display_df = shap_df.copy()
            display_df["feature_label"] = display_df["feature"].map(lambda x: FEATURE_LABELS.get(x, x))
            display_df = display_df.sort_values("abs_shap", ascending=False).head(12)
            display_df = display_df[["feature_label", "raw", "shap_value", "pct_of_abs"]].rename(columns={"feature_label": "Feature", "raw": "Value", "shap_value": "Contribution", "pct_of_abs": "PctImportance"})
            display_df["Value"] = display_df["Value"].apply(lambda v: f"{v:.2f}" if pd.notna(v) else "NaN")
            display_df["Contribution"] = display_df["Contribution"].apply(lambda v: f"{v:.3f}")
            display_df["PctImportance"] = display_df["PctImportance"].apply(lambda v: f"{v:.0%}")
            display_df = display_df.reset_index(drop=True)
            st.dataframe(display_df, use_container_width=True, hide_index=True)

# ---------------- Compare tab ----------------
elif page == "Compare":
    st.markdown('<h2 style="text-align:center; margin-top:10px; margin-bottom:6px; font-size:1.4em; color:#183153;">Compare Players</h2>', unsafe_allow_html=True)

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
                if "id" in rowA and pd.notnull(rowA["id"]):
                    try:
                        pid = str(int(rowA["id"]))
                        imgA = f"https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/w_640,q_auto:best/v1/people/{pid}/headshot/silo/current.png"
                    except Exception:
                        imgA = "https://img.mlbstatic.com/mlb-photos/image/upload/v1/people/0/headshot/silo/current.png"
                else:
                    imgA = "https://img.mlbstatic.com/mlb-photos/image/upload/v1/people/0/headshot/silo/current.png"
                logo_html_a = f'<div style="margin-top:8px;"><img src="{logoA}" style="height:40px;width:40px;border-radius:6px;"></div>' if logoA else ""
                st.markdown(f'<div style="text-align:center;"><img src="{imgA}" style="height:84px;width:84px;border-radius:12px;"><div style="font-weight:800;margin-top:6px;color:#183153;">{playerA}</div>{logo_html_a}</div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div style="text-align:center;padding:8px;border-radius:10px;"><div style="font-size:1.25em;font-weight:800;color:#0b6efd;">Similarity</div><div style="font-size:1.6em;font-weight:800;color:#183153;margin-top:4px;">{sim_pct}</div></div>', unsafe_allow_html=True)
            with col3:
                teamB = rowB["Team"] if "Team" in rowB and pd.notnull(rowB["Team"]) else ""
                logoB = image_dict.get(teamB, "")
                if "id" in rowB and pd.notnull(rowB["id"]):
                    try:
                        pid = str(int(rowB["id"]))
                        imgB = f"https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/w_640,q_auto:best/v1/people/{pid}/headshot/silo/current.png"
                    except Exception:
                        imgB = "https://img.mlbstatic.com/mlb-photos/image/upload/v1/people/0/headshot/silo/current.png"
                else:
                    imgB = "https://img.mlbstatic.com/mlb-photos/image/upload/v1/people/0/headshot/silo/current.png"
                logo_html_b = f'<div style="margin-top:8px;"><img src="{logoB}" style="height:40px;width:40px;border-radius:6px;"></div>' if logoB else ""
                st.markdown(f'<div style="text-align:center;"><img src="{imgB}" style="height:84px;width:84px;border-radius:12px;"><div style="font-weight:800;margin-top:6px;color:#183153;">{playerB}</div>{logo_html_b}</div>', unsafe_allow_html=True)

            st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

            stats = ["Age", "Swing+", "HitSkillPlus", "ImpactPlus"]
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

            # Compare SHAP: ensure shap vectors are aligned with feats and filled with zeros if missing
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

                if shapA is not None:
                    try:
                        shapA = shapA.reindex(feats).fillna(0)
                    except Exception:
                        shapA = pd.Series(0, index=feats)
                if shapB is not None:
                    try:
                        shapB = shapB.reindex(feats).fillna(0)
                    except Exception:
                        shapB = pd.Series(0, index=feats)

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
                    order = importance.sort_values(ascending=False).index.tolist()
                    labels = [FEATURE_LABELS.get(f, f) for f in order]
                    shapA_ord = shapA.reindex(order).fillna(0)
                    shapB_ord = shapB.reindex(order).fillna(0)

                    col_shap_a, col_shap_b = st.columns([1, 1])
                    with col_shap_a:
                        vals = shapA_ord.values.astype(float)
                        colors = ["#D8573C" if v > 0 else "#3B82C4" for v in vals]
                        text_labels = [f"{v:.3f}" for v in vals]
                        figA = go.Figure()
                        figA.add_trace(go.Bar(x=vals, y=labels, orientation='h', marker_color=colors, hoverinfo='text', hovertext=[f"Contribution: {v:.3f}" for v in vals], text=text_labels, textposition='auto'))
                        figA.update_layout(margin=dict(l=160, r=24, t=28, b=60), xaxis_title="SHAP contribution to Swing+ (signed)", yaxis=dict(autorange="reversed"), height=420, showlegend=False, title=f"{playerA} contributions")
                        st.plotly_chart(figA, use_container_width=True, config={"displayModeBar": False})

                    with col_shap_b:
                        vals = shapB_ord.values.astype(float)
                        colors = ["#F59E0B" if v > 0 else "#60A5FA" for v in vals]
                        text_labels = [f"{v:.3f}" for v in vals]
                        figB = go.Figure()
                        figB.add_trace(go.Bar(x=vals, y=labels, orientation='h', marker_color=colors, hoverinfo='text', hovertext=[f"Contribution: {v:.3f}" for v in vals], text=text_labels, textposition='auto'))
                        figB.update_layout(margin=dict(l=160, r=24, t=28, b=60), xaxis_title="SHAP contribution to Swing+ (signed)", yaxis=dict(autorange="reversed"), height=420, showlegend=False, title=f"{playerB} contributions")
                        st.plotly_chart(figB, use_container_width=True, config={"displayModeBar": False})

                st.markdown("### Percentiles (radar)")
                try:
                    pctA_vals = pctA[feats].values
                    pctB_vals = pctB[feats].values
                    labels_radar = [FEATURE_LABELS.get(f, f) for f in feats]
                    fig_r = go.Figure()
                    fig_r.add_trace(go.Scatterpolar(r=pctA_vals, theta=labels_radar, fill='toself', name=playerA, marker_color="#FF7A1A", fillcolor="rgba(255,122,26,0.25)"))
                    fig_r.add_trace(go.Scatterpolar(r=pctB_vals, theta=labels_radar, fill='toself', name=playerB, marker_color="#0b6efd", fillcolor="rgba(11,110,253,0.15)"))
                    fig_r.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), showlegend=True, height=450)
                    st.plotly_chart(fig_r, use_container_width=True, config={"displayModeBar": False})
                except Exception:
                    st.info("Radar chart not available due to data issues.")

                st.markdown("### Feature distribution")
                sel_feat_map = {FEATURE_LABELS.get(f, f): f for f in feats}
                sel_feat_label = st.selectbox("Choose feature for distribution", list(sel_feat_map.keys()), index=0, key="dist_select")
                sel_feat = sel_feat_map.get(sel_feat_label, feats[0])
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
