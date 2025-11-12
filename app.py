# Full Swing+ Streamlit app - updated to use Main.csv and load SwingPlus/HitSkill/ImpactPlus models
# NOTE: This is the full complete script file as requested.

import os
import sys
import math
import json
import time
from functools import lru_cache
from urllib.parse import quote, unquote

import pandas as pd
import numpy as np

import streamlit as st
import joblib
import pickle
import shap

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

import requests
from io import BytesIO
from PIL import Image

import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------- App config --------------------
st.set_page_config(
    page_title="Swing+ Dashboard",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- File & model names --------------------
# Prefer Main.csv per user's dataset; fallback options included
PREFERRED_DATA_FILES = [
    "Main.csv",
    "main.csv",
    "ProjSwingPlus_Output_with_team.csv",
    "data.csv",
    "players.csv"
]

# Model files requested by user
MODEL_FILES = {
    "Swing+": "SwingPlus.pkl",
    "HitSkill+": "HitSkill.pkl",
    "ImpactPlus": "ImpactPlus.pkl"
}

# -------------------- Utilities --------------------
def find_data_file():
    for fname in PREFERRED_DATA_FILES:
        if os.path.exists(fname):
            return fname
    # if nothing found, try to find any csv in CWD
    csvs = [f for f in os.listdir(".") if f.lower().endswith(".csv")]
    if csvs:
        return csvs[0]
    return None

@st.cache_data(ttl=3600)
def load_csv(path):
    # Attempts several read strategies to be robust to weird CSV encodings
    try:
        df = pd.read_csv(path)
        return df
    except Exception:
        try:
            df = pd.read_csv(path, engine="python")
            return df
        except Exception as e:
            raise RuntimeError(f"Failed to read CSV {path}: {e}")

def safe_float(x, default=np.nan):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default

def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*[int(max(0, min(255, round(x)))) for x in rgb])

def gradient_between(start_hex, end_hex, ratio):
    sr, sg, sb = hex_to_rgb(start_hex)
    er, eg, eb = hex_to_rgb(end_hex)
    rr = sr + (er - sr) * ratio
    rg = sg + (eg - sg) * ratio
    rb = sb + (eb - sb) * ratio
    return rgb_to_hex((rr, rg, rb))

# -------------------- Load dataset --------------------
DATA_PATH = find_data_file()
if DATA_PATH is None:
    st.error("No CSV data file found. Place Main.csv (or ProjSwingPlus_Output_with_team.csv) in the app directory.")
    st.stop()

df_raw = load_csv(DATA_PATH)

# Work on a copy
df = df_raw.copy()

# -------------------- Column normalization --------------------
# Map common column names to the canonical names we use in the UI
def normalize_columns(df):
    col_map = {}
    for c in df.columns:
        c_stripped = c.strip()
        lower = c_stripped.lower()
        # canonicalize typical fields
        if lower in ("name", "player", "player_name"):
            col_map[c] = "Name"
        elif lower in ("team", "team_abbr"):
            col_map[c] = "Team"
        elif lower in ("id", "playerid", "player_id"):
            col_map[c] = "id"
        elif lower in ("year", "season"):
            col_map[c] = "year"
        elif lower in ("age",):
            col_map[c] = "Age"
        elif lower in ("swing+", "swing_plus", "swingplus", "swing_plus_score"):
            col_map[c] = "Swing+"
        elif lower in ("hitskill+", "hitskill", "hitskill_plus"):
            col_map[c] = "HitSkill+"
        elif lower in ("impact+", "impactplus", "impact_plus"):
            col_map[c] = "ImpactPlus"
        else:
            # normalize whitespace only
            if c != c_stripped:
                col_map[c] = c_stripped
    if col_map:
        df = df.rename(columns=col_map)
    return df

df = normalize_columns(df)

# Ensure Name exists
if "Name" not in df.columns:
    # try few alternatives
    for alt in ["name", "player", "player_name"]:
        if alt in df.columns:
            df = df.rename(columns={alt: "Name"})
            break

# Drop unnamed cols that are artifacts
unnamed = [c for c in df.columns if c.startswith("Unnamed")]
if unnamed:
    df = df.drop(columns=unnamed, errors="ignore")

# -------------------- Feature detection --------------------
KNOWN_FEATURE_PATTERNS = [
    "avg_", "attack", "swing", "bat", "speed", "tilt", "intercept", "batter_", "length", "direction", "tilt"
]

def detect_mechanical_features(df):
    mech = []
    for c in df.columns:
        if c in ("Name", "Team", "id", "year") or c in MODEL_FILES.keys():
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            low = c.lower()
            if any(p in low for p in KNOWN_FEATURE_PATTERNS):
                mech.append(c)
    # fallback picks if none found
    fallback = [
        "avg_bat_speed", "swing_length", "attack_angle", "swing_tilt", "attack_direction",
        "avg_intercept_y_vs_plate", "avg_intercept_y_vs_batter", "avg_batter_y_position", "avg_batter_x_position"
    ]
    for f in fallback:
        if f in df.columns and f not in mech:
            mech.append(f)
    # keep order stable and unique
    seen = set()
    mech_unique = []
    for m in mech:
        if m not in seen:
            mech_unique.append(m)
            seen.add(m)
    return mech_unique

mechanical_features = detect_mechanical_features(df)

# friendly feature labels if present
FEATURE_LABELS = {
    "avg_bat_speed": "Avg Bat Speed (mph)",
    "swing_length": "Swing Length (m)",
    "attack_angle": "Attack Angle (°)",
    "swing_tilt": "Swing Tilt (°)",
    "attack_direction": "Attack Direction",
    "avg_intercept_y_vs_plate": "Intercept Y vs Plate",
    "avg_intercept_y_vs_batter": "Intercept Y vs Batter",
    "avg_batter_y_position": "Batter Y Pos",
    "avg_batter_x_position": "Batter X Pos",
}

# -------------------- Team logos / headshots --------------------
mlb_teams = [
    {"team": "AZ", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/ari.png"},
    {"team": "ATL", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/atl.png"},
    {"team": "BAL", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/bal.png"},
    {"team": "BOS", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/bos.png"},
    {"team": "CHC", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/chc.png"},
    {"team": "CWS", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/chw.png"},
    {"team": "CIN", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/cin.png"},
    {"team": "CLE", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/cle.png"},
    {"team": "COL", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/col.png"},
    {"team": "DET", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/det.png"},
    {"team": "HOU", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/hou.png"},
    {"team": "KC", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/kc.png"},
    {"team": "LAA", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/laa.png"},
    {"team": "LAD", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/lad.png"},
    {"team": "MIA", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/mia.png"},
    {"team": "MIL", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/mil.png"},
    {"team": "MIN", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/min.png"},
    {"team": "NYM", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/nym.png"},
    {"team": "NYY", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/nyy.png"},
    {"team": "OAK", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/oak.png"},
    {"team": "PHI", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/phi.png"},
    {"team": "PIT", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/pit.png"},
    {"team": "SD", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/sd.png"},
    {"team": "SF", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/sf.png"},
    {"team": "SEA", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/sea.png"},
    {"team": "STL", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/stl.png"},
    {"team": "TB", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/tb.png"},
    {"team": "TEX", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/tex.png"},
    {"team": "TOR", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/tor.png"},
    {"team": "WSH", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/wsh.png"}
]

team_logo_map = pd.DataFrame(mlb_teams).set_index("team")["logo_url"].to_dict()

# -------------------- Load models and SHAP explainers --------------------
models = {}
explainers = {}
model_errors = {}

def load_model_file(path):
    # try joblib then pickle
    try:
        return joblib.load(path)
    except Exception:
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            raise e

for label, fname in MODEL_FILES.items():
    if os.path.exists(fname):
        try:
            mdl = load_model_file(fname)
            models[label] = mdl
            # create SHAP explainer with fallbacks
            try:
                expl = shap.Explainer(mdl)
            except Exception:
                try:
                    expl = shap.TreeExplainer(mdl)
                except Exception:
                    expl = None
            explainers[label] = expl
        except Exception as e:
            models[label] = None
            explainers[label] = None
            model_errors[label] = str(e)
    else:
        models[label] = None
        explainers[label] = None
        model_errors[label] = f"File not found: {fname}"

# -------------------- Model input preparation --------------------
def infer_model_expected_features(model_obj, fallback):
    expected = None
    try:
        if model_obj is None:
            expected = list(fallback)
        else:
            if hasattr(model_obj, "feature_names_in_"):
                expected = list(model_obj.feature_names_in_)
            elif hasattr(model_obj, "feature_name_"):
                expected = list(model_obj.feature_name_)
            elif hasattr(model_obj, "get_booster") and hasattr(model_obj.get_booster(), "feature_names"):
                expected = list(model_obj.get_booster().feature_names)
            elif hasattr(model_obj, "booster_") and hasattr(model_obj.booster_, "feature_names"):
                expected = list(model_obj.booster_.feature_names)
            else:
                expected = list(fallback)
    except Exception:
        expected = list(fallback)
    if expected is None or len(expected) == 0:
        expected = list(fallback)
    return expected

def prepare_model_input(player_row, model_obj, df_reference, fallback_features):
    expected = infer_model_expected_features(model_obj, fallback_features)
    row = {}
    for feat in expected:
        if feat in player_row and pd.notna(player_row[feat]):
            row[feat] = player_row[feat]
        else:
            # tolerant lookup variants
            for alt in (feat, feat.replace(" ", "_"), feat.lower(), feat.upper()):
                if alt in player_row and pd.notna(player_row[alt]):
                    row[feat] = player_row[alt]
                    break
            else:
                # fallback to mean from df_reference if possible
                if feat in df_reference.columns:
                    try:
                        row[feat] = float(df_reference[feat].mean())
                    except Exception:
                        row[feat] = 0.0
                else:
                    row[feat] = 0.0
    X = pd.DataFrame([row], columns=expected)
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0).astype(float)
    return X

# -------------------- SHAP compute wrapper --------------------
@st.cache_data(ttl=300)
def compute_shap_for_player(player_row_dict, model_key):
    mdl = models.get(model_key)
    expl = explainers.get(model_key)
    if mdl is None or expl is None:
        return None, None, None
    # prepare input
    Xp = prepare_model_input(player_row_dict, mdl, df, mechanical_features)
    try:
        pred = float(mdl.predict(Xp)[0])
    except Exception:
        try:
            pred = float(mdl.predict(Xp.values.reshape(1, -1))[0])
        except Exception:
            pred = None
    try:
        shap_vals = expl(Xp)
    except Exception:
        try:
            shap_vals = expl(Xp.values)
        except Exception:
            shap_vals = None
    if shap_vals is None:
        return None, pred, None
    if hasattr(shap_vals, "values"):
        vals = np.array(shap_vals.values).flatten()
        base = float(shap_vals.base_values) if np.size(shap_vals.base_values) == 1 else None
    else:
        vals = np.array(shap_vals).flatten()
        base = None
    shap_ser = pd.Series(vals, index=Xp.columns)
    return shap_ser, pred, base

# -------------------- Cosine similarity utilities --------------------
@st.cache_data(ttl=600)
def get_scaled_mech_df(mech_feats):
    X = df[mech_feats].astype(float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    df_scaled = pd.DataFrame(Xs, columns=mech_feats, index=df.index)
    return scaler, df_scaled

def compute_similar_players(player_name, mech_feats, top_n=10):
    # require player in df and features present
    if player_name not in df["Name"].values:
        return []
    df_mech = df.dropna(subset=mech_feats + ["Name"]).reset_index(drop=True)
    if df_mech.shape[0] <= 1:
        return []
    scaler = StandardScaler()
    Xs = scaler.fit_transform(df_mech[mech_feats])
    sim = cosine_similarity(Xs)
    sim_df = pd.DataFrame(sim, index=df_mech["Name"], columns=df_mech["Name"])
    s = sim_df.loc[player_name].sort_values(ascending=False).iloc[1:top_n+1]
    return list(zip(s.index.tolist(), s.values.tolist()))

# -------------------- Query params to allow linking --------------------
params = st.experimental_get_query_params()
qp_player = None
qp_player_b = None
qp_page = None

if "player" in params and params["player"]:
    try:
        qp_player = unquote(params["player"][0])
    except Exception:
        qp_player = params["player"][0]

if "playerA" in params and params["playerA"]:
    try:
        qp_player = unquote(params["playerA"][0])
    except Exception:
        qp_player = params["playerA"][0]

if "playerB" in params and params["playerB"]:
    try:
        qp_player_b = unquote(params["playerB"][0])
    except Exception:
        qp_player_b = params["playerB"][0]

if "page" in params and params["page"]:
    try:
        qp_page = unquote(params["page"][0])
    except Exception:
        qp_page = params["page"][0]

# -------------------- Page navigation --------------------
page_options = ["Main", "Player", "Compare", "Glossary"]
default_page = 0
if qp_page and qp_page in page_options:
    default_page = page_options.index(qp_page)
elif qp_player:
    default_page = page_options.index("Player")
elif qp_player_b:
    default_page = page_options.index("Compare")

st.markdown("<div style='display:flex;justify-content:center;margin-bottom:6px;'>", unsafe_allow_html=True)
page = st.radio("", page_options, index=default_page, horizontal=True)
st.markdown("</div>", unsafe_allow_html=True)

# -------------------- Sidebar filters --------------------
st.sidebar.header("Filters")

# Year filter if present
if "year" in df.columns:
    years = sorted(df["year"].dropna().unique().astype(int).tolist())
    if years:
        default_year = years[-1]
        selected_years = st.sidebar.multiselect("Year(s)", years, default=[default_year])
        if selected_years:
            df = df[df["year"].isin(selected_years)]

# Age filter
if "Age" in df.columns:
    amin = int(df["Age"].min())
    amax = int(df["Age"].max())
    age_range = st.sidebar.slider("Age Range", amin, amax, (amin, amax))
    df = df[(df["Age"] >= age_range[0]) & (df["Age"] <= age_range[1])]

# Search name
search_name = st.sidebar.text_input("Search Player by Name")
if search_name and "Name" in df.columns:
    df = df[df["Name"].str.contains(search_name, case=False, na=False)]

# Optional swings/bbe filters if present
if "competitive_swings" in df.columns:
    min_cs = int(df["competitive_swings"].min())
    max_cs = int(df["competitive_swings"].max())
    cs_range = st.sidebar.slider("Competitive Swings", min_cs, max_cs, (min_cs, max_cs))
    df = df[(df["competitive_swings"] >= cs_range[0]) & (df["competitive_swings"] <= cs_range[1])]

if "batted_ball_events" in df.columns:
    min_bbe = int(df["batted_ball_events"].min())
    max_bbe = int(df["batted_ball_events"].max())
    bbe_range = st.sidebar.slider("Batted Ball Events", min_bbe, max_bbe, (min_bbe, max_bbe))
    df = df[(df["batted_ball_events"] >= bbe_range[0]) & (df["batted_ball_events"] <= bbe_range[1])]

# -------------------- Main page --------------------
if page == "Main":
    st.markdown("<h1 style='text-align:center;margin-bottom:0.2rem;color:#183153;'>Swing+ Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:center;color:#6b7280;margin-bottom:1rem;'>Browse players, leaderboards, and model predictions</div>", unsafe_allow_html=True)

    # Build display columns dynamically
    display_cols = ["Name"]
    if "Team" in df.columns:
        display_cols.append("Team")
    if "Age" in df.columns:
        display_cols.append("Age")

    # include dataset metrics if present
    for metric in MODEL_FILES.keys():
        if metric in df.columns:
            display_cols.append(metric)

    # add some mechanical features for quick glance
    display_cols += [c for c in mechanical_features[:8] if c in df.columns]

    # friendly rename map
    rename_map = {}
    for f, lab in FEATURE_LABELS.items():
        if f in df.columns:
            rename_map[f] = lab

    # If est_woba/xwOBA present, map to nicer names
    if "est_woba" in df.columns and "xwOBA" not in df.columns:
        rename_map["est_woba"] = "xwOBA"

    if "xwOBA_pred" in df.columns and "Predicted xwOBA" not in df.columns:
        rename_map["xwOBA_pred"] = "Predicted xwOBA"

    try:
        df_display = df[display_cols].rename(columns=rename_map).sort_values(by=[c for c in ["Swing+"] if c in df.columns], ascending=False).reset_index(drop=True)
        st.dataframe(df_display.style.format(precision=2), use_container_width=True, hide_index=True)
    except Exception:
        st.dataframe(df[display_cols].head(200), use_container_width=True)

    # Leaderboards
    st.markdown("## Top Leaderboards")
    col1, col2 = st.columns(2)

    with col1:
        if "Swing+" in df.columns:
            st.write("Top 10 by Swing+")
            top = df.sort_values("Swing+", ascending=False).head(10)
            cols = [c for c in ["Name", "Team", "Age", "Swing+"] if c in top.columns]
            st.dataframe(top[cols].style.background_gradient(subset=["Swing+"], cmap="Reds"), use_container_width=True, hide_index=True)
        else:
            st.info("Swing+ metric not present in dataset.")

    with col2:
        # choose secondary metric
        secondary = None
        for cand in ["HitSkill+", "ImpactPlus", "xwOBA_pred", "est_woba"]:
            if cand in df.columns:
                secondary = cand
                break
        if secondary:
            st.write(f"Top 10 by {secondary}")
            top2 = df.sort_values(secondary, ascending=False).head(10)
            cols2 = [c for c in ["Name", "Team", "Age", secondary] if c in top2.columns]
            st.dataframe(top2[cols2].style.background_gradient(subset=[secondary], cmap="Reds"), use_container_width=True, hide_index=True)
        else:
            st.info("No secondary leaderboard metric found in dataset.")


# -------------------- Player page --------------------
elif page == "Player":
    st.markdown("<h2 style='text-align:center;color:#183153;'>Player Detail</h2>", unsafe_allow_html=True)

    if "Name" not in df.columns:
        st.error("Dataset missing 'Name' column.")
        st.stop()

    player_options = sorted(df["Name"].unique().tolist())
    default_index = 0
    if qp_player and qp_player in player_options:
        default_index = player_options.index(qp_player)

    player_select = st.selectbox("Select a Player", player_options, index=default_index, key="player_select")
    player_rows = df[df["Name"] == player_select]
    if player_rows.empty:
        st.warning("Player not found in dataset.")
        st.stop()

    # if multiple year rows, pick the most recent if year present
    if "year" in player_rows.columns:
        player_row = player_rows.sort_values("year", ascending=False).iloc[0]
    else:
        player_row = player_rows.iloc[0]

    # Header with headshot & team logo
    headshot_html = ""
    if "id" in player_row and pd.notna(player_row["id"]):
        try:
            pid = str(int(player_row["id"]))
            headshot_url = f"https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/w_640,q_auto:best/v1/people/{pid}/headshot/silo/current.png"
            headshot_html = f'<img src="{headshot_url}" style="height:96px;width:96px;border-radius:14px;object-fit:cover;margin-right:18px;" alt="headshot"/>'
        except Exception:
            headshot_html = ""
    else:
        fallback = "https://img.mlbstatic.com/mlb-photos/image/upload/v1/people/0/headshot/silo/current.png"
        headshot_html = f'<img src="{fallback}" style="height:96px;width:96px;border-radius:14px;object-fit:cover;margin-right:18px;" alt="headshot"/>'

    team_logo_html = ""
    if "Team" in player_row and pd.notna(player_row["Team"]):
        t = str(player_row["Team"]).strip()
        t_url = team_logo_map.get(t, "")
        if t_url:
            team_logo_html = f'<img src="{t_url}" style="height:80px;width:80px;border-radius:8px;margin-left:12px;" alt="team"/>'

    st.markdown(
        f"""
        <div style="display:flex;justify-content:center;align-items:center;margin-bottom:8px;">
            {headshot_html}
            <div style="display:flex;flex-direction:column;align-items:center;">
                <div style="font-size:28px;font-weight:800;color:#183153;">{player_select}</div>
                <div style="color:#6b7280;margin-top:6px;">{player_row.get('Team','')}</div>
            </div>
            {team_logo_html}
        </div>
        """,
        unsafe_allow_html=True
    )

    # Top metric cards for each model
    st.markdown("<div style='display:flex;gap:18px;justify-content:center;margin-bottom:14px;'>", unsafe_allow_html=True)
    for metric_label in MODEL_FILES.keys():
        val = player_row.get(metric_label, None)
        val_text = f"{val:.2f}" if (val is not None and not pd.isna(val)) else "N/A"
        mdl = models.get(metric_label)
        pred_text = "N/A"
        if mdl is not None:
            try:
                Xp = prepare_model_input(player_row, mdl, df, mechanical_features)
                try:
                    pred_val = float(mdl.predict(Xp)[0])
                except Exception:
                    pred_val = float(mdl.predict(Xp.values.reshape(1, -1))[0])
                pred_text = f"{pred_val:.2f}"
            except Exception:
                pred_text = "Err"
        st.markdown(
            f"""
            <div style="background:#fff;padding:14px;border-radius:14px;min-width:150px;text-align:center;box-shadow:0 4px 18px rgba(0,0,0,0.04);">
                <div style="font-size:20px;font-weight:800;color:#183153;">{pred_text}</div>
                <div style="color:#8892a6;margin-top:6px;font-weight:700;">{metric_label} (model)</div>
                <div style="color:#6b7280;margin-top:8px;">Actual: {val_text}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # Display SHAP feature contributions
    st.markdown("### Feature contributions (SHAP)")
    model_choice = st.selectbox("Choose model to explain", list(MODEL_FILES.keys()))
    shap_ser, shap_pred, shap_base = compute_shap_for_player(player_row.to_dict(), model_choice)
    if shap_ser is None:
        st.info(f"SHAP not available for {model_choice}. Model loaded: {'Yes' if models.get(model_choice) is not None else 'No'}.")
        if model_errors.get(model_choice):
            st.caption(f"Model load error: {model_errors.get(model_choice)}")
    else:
        shap_df = pd.DataFrame({"feature": shap_ser.index, "shap_value": shap_ser.values})
        shap_df["abs_shap"] = shap_df["shap_value"].abs()
        shap_df = shap_df.sort_values("abs_shap", ascending=False).head(12).reset_index(drop=True)
        shap_df["feature_label"] = shap_df["feature"].map(lambda x: FEATURE_LABELS.get(x, x))
        # build horizontal bar chart
        y = shap_df["feature_label"].tolist()[::-1]
        x_vals = shap_df["shap_value"].tolist()[::-1]
        colors = ["#D8573C" if v > 0 else "#3B82C4" for v in x_vals]
        fig = go.Figure(go.Bar(x=x_vals, y=y, orientation='h', marker_color=colors, text=[f"{v:.3f}" for v in x_vals], textposition='inside'))
        fig.update_layout(margin=dict(l=180, r=20, t=10, b=60), xaxis_title=f"SHAP contribution ({model_choice})", height=420, yaxis=dict(autorange="reversed"), showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True, "displayModeBar": False})

        # show table
        table_df = shap_df[["feature_label", "shap_value", "abs_shap"]].rename(columns={"feature_label": "Feature", "shap_value": "Contribution", "abs_shap": "AbsImportance"})
        table_df["Contribution"] = table_df["Contribution"].apply(lambda v: f"{v:.3f}")
        table_df["AbsImportance"] = table_df["AbsImportance"].apply(lambda v: f"{v:.3f}")
        st.dataframe(table_df, use_container_width=True, hide_index=True)

    # Mechanically similar players
    st.markdown("### Mechanically similar players")
    mech_feats = [f for f in mechanical_features if f in df.columns]
    if len(mech_feats) >= 2:
        similar = compute_similar_players(player_select, mech_feats, top_n=10)
        if similar:
            for idx, (name, score) in enumerate(similar, start=1):
                # player headshot lookup if id available
                img_url = ""
                r = df[df["Name"] == name].iloc[0]
                if "id" in r and pd.notna(r["id"]):
                    try:
                        pid = str(int(r["id"]))
                        img_url = f"https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/w_640,q_auto:best/v1/people/{pid}/headshot/silo/current.png"
                    except Exception:
                        img_url = ""
                score_pct = f"{score:.1%}"
                href = f"?playerA={quote(player_select)}&playerB={quote(name)}&page=Compare"
                st.markdown(
                    f"""
                    <div style="display:flex;align-items:center;gap:12px;padding:10px;border-radius:10px;border:1px solid #eef4f8;margin-bottom:6px;">
                        <div style="font-weight:800;color:#183153;width:28px;text-align:center;">{idx}</div>
                        <img src="{img_url}" style="height:48px;width:48px;border-radius:8px;object-fit:cover;"/>
                        <div style="flex:1;font-weight:700;color:#183153;"><a href="?player={quote(name)}" style="color:#183153;text-decoration:none;">{name}</a></div>
                        <div style="min-width:72px;text-align:right;font-weight:700;">{score_pct}</div>
                        <div><a href="{href}" style="font-weight:800;padding:8px 10px;border-radius:8px;border:1px solid #d1d5db;background:#fff;text-decoration:none;color:#111;">Compare</a></div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with st.expander("Show similarity heatmap"):
                # build small heatmap of the top cluster including selected
                names = [player_select] + [n for n, s in similar]
                df_mech = df.dropna(subset=mech_feats + ["Name"]).reset_index(drop=True)
                idxs = [df_mech[df_mech["Name"] == n].index[0] for n in names]
                scaler = StandardScaler()
                Xs = scaler.fit_transform(df_mech[mech_feats])
                mat = cosine_similarity(Xs)
                cluster = mat[np.ix_(idxs, idxs)]
                fig_h, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cluster, xticklabels=names, yticklabels=names, annot=True, cmap="RdYlBu_r", ax=ax)
                ax.set_title("Mechanical similarity")
                st.pyplot(fig_h)
        else:
            st.info("Not enough mechanical data to compute similarity for this player.")
    else:
        st.info("Not enough detected mechanical features to compute similarity.")

# -------------------- Compare page --------------------
elif page == "Compare":
    st.markdown("<h2 style='text-align:center;color:#183153;'>Compare Players</h2>", unsafe_allow_html=True)
    if "Name" not in df.columns:
        st.error("Dataset missing 'Name'.")
        st.stop()

    all_players = sorted(df["Name"].unique().tolist())
    params = st.experimental_get_query_params()
    pA = None
    pB = None
    if "playerA" in params and params["playerA"]:
        pA = unquote(params["playerA"][0])
    if "playerB" in params and params["playerB"]:
        pB = unquote(params["playerB"][0])

    col1, col2 = st.columns(2)
    with col1:
        selA = st.selectbox("Player A", all_players, index=all_players.index(pA) if pA in all_players else 0)
    with col2:
        selB = st.selectbox("Player B", all_players, index=all_players.index(pB) if pB in all_players else 1 if len(all_players) > 1 else 0)

    if selA == selB:
        st.warning("Choose two different players.")
    else:
        rowA = df[df["Name"] == selA].sort_values("year", ascending=False).iloc[0]
        rowB = df[df["Name"] == selB].sort_values("year", ascending=False).iloc[0]

        # side-by-side card for each model + actual columns
        cols = st.columns(len(MODEL_FILES))
        for i, metric_label in enumerate(MODEL_FILES.keys()):
            with cols[i]:
                valA = rowA.get(metric_label, None)
                valB = rowB.get(metric_label, None)
                valA_text = f"{valA:.2f}" if (valA is not None and not pd.isna(valA)) else "N/A"
                valB_text = f"{valB:.2f}" if (valB is not None and not pd.isna(valB)) else "N/A"
                mdl = models.get(metric_label)
                predA = predB = "N/A"
                if mdl is not None:
                    try:
                        XA = prepare_model_input(rowA, mdl, df, mechanical_features)
                        XB = prepare_model_input(rowB, mdl, df, mechanical_features)
                        try:
                            pA_val = float(mdl.predict(XA)[0])
                        except Exception:
                            pA_val = float(mdl.predict(XA.values.reshape(1, -1))[0])
                        try:
                            pB_val = float(mdl.predict(XB)[0])
                        except Exception:
                            pB_val = float(mdl.predict(XB.values.reshape(1, -1))[0])
                        predA = f"{pA_val:.2f}"
                        predB = f"{pB_val:.2f}"
                    except Exception:
                        predA = predB = "Err"
                st.markdown(
                    f"""
                    <div style="background:#fff;padding:12px;border-radius:12px;text-align:center;box-shadow:0 4px 18px rgba(0,0,0,0.04);">
                        <div style="font-weight:800;color:#183153;font-size:18px;">{metric_label}</div>
                        <div style="display:flex;justify-content:space-between;margin-top:10px;">
                            <div style="flex:1;text-align:center;">A: <strong>{predA}</strong><div style="color:#6b7280;margin-top:6px;">Actual: {valA_text}</div></div>
                            <div style="flex:1;text-align:center;border-left:1px solid #f0f0f0;">B: <strong>{predB}</strong><div style="color:#6b7280;margin-top:6px;">Actual: {valB_text}</div></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True
                )

        st.markdown("---")
        st.subheader("Mechanical feature radar / comparison")
        # plot a few mechanical features side-by-side if available
        mech_feats = [f for f in mechanical_features if f in df.columns][:8]
        if len(mech_feats) >= 3:
            valuesA = [safe_float(rowA.get(f, np.nan), np.nan) for f in mech_feats]
            valuesB = [safe_float(rowB.get(f, np.nan), np.nan) for f in mech_feats]
            # normalize to 0-1 for radar plotting by percentiles across dataset for each feat
            norm_valsA = []
            norm_valsB = []
            mins = []
            maxs = []
            for f in mech_feats:
                col = df[f].astype(float)
                mn = np.nanmin(col)
                mx = np.nanmax(col)
                mins.append(mn)
                maxs.append(mx)
            for v, mn, mx in zip(valuesA, mins, maxs):
                if np.isnan(v) or mx - mn == 0:
                    norm_valsA.append(0.5)
                else:
                    norm_valsA.append((v - mn) / (mx - mn))
            for v, mn, mx in zip(valuesB, mins, maxs):
                if np.isnan(v) or mx - mn == 0:
                    norm_valsB.append(0.5)
                else:
                    norm_valsB.append((v - mn) / (mx - mn))
            # radar plot via plotly
            categories = [FEATURE_LABELS.get(f, f) for f in mech_feats]
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=norm_valsA + [norm_valsA[0]], theta=categories + [categories[0]], fill='toself', name=selA))
            fig.add_trace(go.Scatterpolar(r=norm_valsB + [norm_valsB[0]], theta=categories + [categories[0]], fill='toself', name=selB))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), showlegend=True, height=520)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough mechanical features to show radar comparison.")

# -------------------- Glossary page --------------------
elif page == "Glossary":
    st.markdown("<h2 style='text-align:center;color:#183153;'>Glossary & Detected Features</h2>", unsafe_allow_html=True)
    st.write("Detected mechanical features in the dataset (automatically detected):")
    st.write(mechanical_features)
    st.write("Friendly feature labels mapping (used in charts/tables when available):")
    st.write(FEATURE_LABELS)
    st.write("Model load status:")
    for mk in MODEL_FILES.keys():
        status = "Loaded" if models.get(mk) is not None else f"Not loaded ({model_errors.get(mk,'missing')})"
        st.write(f"- {mk}: {status}")

# -------------------- Footer info --------------------
st.markdown("---")
st.caption(f"Data file: {DATA_PATH} — rows: {len(df_raw)} (raw), {len(df)} (after filters). Mechanical features detected: {len(mechanical_features)}")

# Show quick debug panel (collapsed) with model load errors for debugging
with st.expander("Debug: model load details", expanded=False):
    st.write("Model load errors (if any):")
    st.write(model_errors)
    st.write("Available models objects keys:")
    st.write({k: (type(v).__name__ if v is not None else None) for k, v in models.items()})
    st.write("Explainer objects keys:")
    st.write({k: (type(v).__name__ if v is not None else None) for k, v in explainers.items()})

# End of script
