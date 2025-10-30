import pandas as pd
import streamlit as st
import plotly.express as px

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
This dashboard visualizes **Swing+**, **ProjSwing+**, **PowerIndex+**, and **GapPotential**  
from your `ProjSwingPlus_Output.csv`.  
Use the filters below to explore players and compare across metrics.
""")

# =============================
# LOAD DATA
# =============================
uploaded_file = st.file_uploader("Upload your ProjSwingPlus_Output.csv", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Ensure key columns exist
    required_cols = ["Name", "Age", "Swing+", "PowerIndex+", "ProjSwing+", "GapPotential"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()

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
    # TABLE DISPLAY
    # =============================
    st.subheader("📊 Player Metrics Table")
    st.dataframe(
        df_filtered[["Name", "Age", "Swing+", "PowerIndex+", "ProjSwing+", "GapPotential"]]
        .sort_values("ProjSwing+", ascending=False)
        .style.background_gradient(subset=["ProjSwing+"], cmap="YlOrBr")
        .format(precision=1)
    )

    # =============================
    # VISUAL COMPARISONS
    # =============================
    st.subheader("📈 Swing+ vs ProjSwing+ Scatter")
    fig = px.scatter(
        df_filtered,
        x="Swing+",
        y="ProjSwing+",
        color="PowerIndex+",
        color_continuous_scale="YlOrBr",
        hover_name="Name",
        size="GapPotential",
        title="Swing+ vs ProjSwing+ (Colored by PowerIndex+)",
        template="plotly_white"
    )
    fig.add_hline(y=100, line_dash="dash", line_color="gray")
    fig.add_vline(x=100, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)

    # =============================
    # LEADERBOARDS
    # =============================
    st.subheader("🏆 Top 10 Leaderboards")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Top 10 by ProjSwing+**")
        top_proj = df_filtered.sort_values("ProjSwing+", ascending=False).head(10)
        st.dataframe(top_proj[["Name", "Age", "ProjSwing+", "Swing+", "GapPotential"]]
                     .style.background_gradient(subset=["ProjSwing+"], cmap="YlOrBr")
                     .format(precision=1))

    with col2:
        st.markdown("**Top 10 by PowerIndex+**")
        top_power = df_filtered.sort_values("PowerIndex+", ascending=False).head(10)
        st.dataframe(top_power[["Name", "Age", "PowerIndex+", "Swing+", "ProjSwing+"]]
                     .style.background_gradient(subset=["PowerIndex+"], cmap="YlOrBr")
                     .format(precision=1))

    # =============================
    # PLAYER DETAIL SECTION
    # =============================
    st.subheader("🔍 Player Detail View")
    player_select = st.selectbox("Select a Player", sorted(df_filtered["Name"].unique()))
    player_data = df[df["Name"] == player_select].iloc[0]

    colA, colB, colC, colD = st.columns(4)
    colA.metric("Swing+", round(player_data["Swing+"], 1))
    colB.metric("PowerIndex+", round(player_data["PowerIndex+"], 1))
    colC.metric("ProjSwing+", round(player_data["ProjSwing+"], 1))
    colD.metric("GapPotential", round(player_data["GapPotential"], 1))

else:
    st.info("👆 Upload `ProjSwingPlus_Output.csv` to begin.")
