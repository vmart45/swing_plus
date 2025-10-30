import pandas as pd
import streamlit as st
import os

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

DATA_PATH = "ProjSwingPlus_Output_with_team.csv"

if not os.path.exists(DATA_PATH):
    st.error(f"❌ Could not find `{DATA_PATH}` in the app directory.")
    st.stop()

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

df = load_data(DATA_PATH)

core_cols = ["Name", "Age", "Swing+", "PowerIndex+", "ProjSwing+"]
extra_cols = ["avg_bat_speed", "swing_length", "attack_angle", "swing_tilt"]
if "id" in df.columns:
    core_cols = ["id"] + core_cols
if "Team" in df.columns and "Team" not in core_cols:
    core_cols.insert(1, "Team")
required_cols = core_cols + [c for c in extra_cols if c in df.columns]

missing = [c for c in core_cols if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

st.sidebar.header("Filters")

min_age, max_age = int(df["Age"].min()), int(df["Age"].max())
age_range = st.sidebar.slider("Age Range", min_age, max_age, (min_age, 25))

df_filtered = df[(df["Age"] >= age_range[0]) & (df["Age"] <= age_range[1])]

search_name = st.sidebar.text_input("Search Player by Name")
if search_name:
    df_filtered = df_filtered[df_filtered["Name"].str.contains(search_name, case=False, na=False)]

main_cmap = "RdYlBu_r"
elite_cmap = "Reds"

st.subheader("📊 Player Metrics Table")

display_cols = [c for c in ["Name", "Team", "Age", "Swing+", "ProjSwing+", "PowerIndex+"] + extra_cols if c in df_filtered.columns]

rename_map = {
    "Team": "Team",
    "Swing+": "Swing+",
    "ProjSwing+": "ProjSwing+",
    "PowerIndex+": "PowerIndex+",
    "avg_bat_speed": "Avg Bat Speed (mph)",
    "swing_length": "Swing Length (m)",
    "attack_angle": "Attack Angle (°)",
    "swing_tilt": "Swing Tilt (°)"
}

styled_df = (
    df_filtered[display_cols]
    .rename(columns=rename_map)
    .sort_values("Swing+", ascending=False)
    .reset_index(drop=True)
    .style.background_gradient(
        subset=["Swing+", "ProjSwing+", "PowerIndex+"], cmap=main_cmap
    )
    .format(precision=1)
)

st.dataframe(styled_df, use_container_width=True, hide_index=True)

st.subheader("🏆 Top 10 Leaderboards")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Top 10 by Swing+**")
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
    st.markdown("**Top 10 by ProjSwing+**")
    top_proj = df_filtered.sort_values("ProjSwing+", ascending=False).head(10).reset_index(drop=True)
    leaderboard_cols = [c for c in ["Name", "Team", "Age", "ProjSwing+", "Swing+", "PowerIndex+"] if c in top_proj.columns]
    st.dataframe(
        top_proj[leaderboard_cols]
        .style.background_gradient(subset=["ProjSwing+"], cmap=elite_cmap)
        .format(precision=1),
        use_container_width=True,
        hide_index=True
    )

st.subheader("🔍 Player Detail View")

player_select = st.selectbox("Select a Player", sorted(df_filtered["Name"].unique()))
player_row = df[df["Name"] == player_select].iloc[0]

st.markdown(
    f"""
    <h2 style="text-align:center; margin-bottom:0.5em;">{player_select}</h2>
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
    <div style="display:flex; justify-content:center; gap:60px; margin-top:10px; margin-bottom:20px;">
        <div style="text-align:center;">
            <div style="width:70px; height:70px; border-radius:50%; background:linear-gradient(135deg,#FFECB3,#F44336 80%); display:flex; align-items:center; justify-content:center; margin:0 auto;">
                <span style="font-size:2em; font-weight:600; color:#222;">{p_swing_rank}</span>
            </div>
            <div style="margin-top:6px; font-weight:600;">Swing+</div>
            <div style="color:gray; font-size:13px;">{round(player_row['Swing+'],1)}</div>
        </div>
        <div style="text-align:center;">
            <div style="width:70px; height:70px; border-radius:50%; background:linear-gradient(135deg,#C8E6C9,#388E3C 80%); display:flex; align-items:center; justify-content:center; margin:0 auto;">
                <span style="font-size:2em; font-weight:600; color:#222;">{p_proj_rank}</span>
            </div>
            <div style="margin-top:6px; font-weight:600;">ProjSwing+</div>
            <div style="color:gray; font-size:13px;">{round(player_row['ProjSwing+'],1)}</div>
        </div>
        <div style="text-align:center;">
            <div style="width:70px; height:70px; border-radius:50%; background:linear-gradient(135deg,#B3E5FC,#1976D2 80%); display:flex; align-items:center; justify-content:center; margin:0 auto;">
                <span style="font-size:2em; font-weight:600; color:#222;">{p_power_rank}</span>
            </div>
            <div style="margin-top:6px; font-weight:600;">PowerIndex+</div>
            <div style="color:gray; font-size:13px;">{round(player_row['PowerIndex+'],1)}</div>
        </div>
    </div>
    <div style="text-align:center; font-size:12px; color:#888; margin-top:-10px;">
      Rank out of {total_players} players
    </div>
    """,
    unsafe_allow_html=True
)

if set(extra_cols).issubset(df.columns):
    st.markdown("**Swing Mechanics**")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Bat Speed", f"{round(player_row['avg_bat_speed'], 1)} mph")
    col2.metric("Swing Length", round(player_row["swing_length"], 2))
    col3.metric("Attack Angle", round(player_row["attack_angle"], 1))
    col4.metric("Swing Tilt", round(player_row["swing_tilt"], 1))
