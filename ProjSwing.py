import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# ==============================
# CONFIG
# ==============================
MECH_DATA = "bat_tracking_merged.csv"
SWINGPLUS_DATA = "SwingPlus_with_age.csv"
SAVE_PLOT = True
OUTPUT_FILE = "ProjSwingPlus_Output.csv"
PLOT_FILE = "Swing_vs_ProjSwingPlus.png"
AGE_CUTOFF = 25

# ==============================
# LOAD DATA
# ==============================
mech = pd.read_csv(MECH_DATA)
swing = pd.read_csv(SWINGPLUS_DATA)

# ==============================
# NORMALIZE PLAYER NAMES
# ==============================
if "last_name, first_name" in mech.columns:
    mech = mech.rename(columns={"last_name, first_name": "Name"})

mech["Name"] = mech["Name"].apply(
    lambda x: " ".join([p.strip() for p in str(x).split(",")[::-1]]) if "," in str(x) else str(x)
)

# ==============================
# MERGE
# ==============================
common_names = set(mech["Name"]) & set(swing["Name"])
print(f"✅ Found {len(common_names)} matching player names between datasets.")
if len(common_names) == 0:
    raise ValueError("❌ No matching names found — check name formatting.")

df = mech.merge(swing, on="Name", how="inner")
print(f"✅ Merge successful. Combined rows: {len(df)}")

# ==============================
# CLEAN COLUMN NAMES
# ==============================
cols_to_drop = [c for c in df.columns if c.endswith("_y")]
df = df.drop(columns=cols_to_drop)
df.columns = [c.replace("_x", "") for c in df.columns]

# ==============================
# VALIDATE REQUIRED COLUMNS
# ==============================
required_cols = [
    "Swing+",
    "avg_bat_speed",
    "swing_length",
    "attack_angle",
    "swing_tilt",
    "attack_direction"
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    print("⚠️ Available columns:")
    print(list(df.columns))
    raise ValueError(f"Missing required columns after merge: {missing}")

df = df.dropna(subset=required_cols)
print(f"✅ Rows after dropping missing data: {len(df)}")

# ==============================
# SCALE & CALCULATE POWER INDEX
# ==============================
power_features = [
    "avg_bat_speed",
    "swing_length",
    "attack_angle",
    "swing_tilt",
    "attack_direction"
]

scaler = StandardScaler()
scaled_values = scaler.fit_transform(df[power_features])
scaled_df = pd.DataFrame(scaled_values, columns=power_features, index=df.index)

df["PowerIndex"] = (
    0.5 * scaled_df["avg_bat_speed"] +
    0.2 * scaled_df["swing_length"] +
    0.15 * scaled_df["attack_angle"] +
    0.1 * scaled_df["swing_tilt"] +
    0.05 * scaled_df["attack_direction"]
)

df["PowerIndex+"] = 100 + ((df["PowerIndex"] - df["PowerIndex"].mean()) / df["PowerIndex"].std()) * 10

# ==============================
# CREATE PROJSWING+ & GAP POTENTIAL
# ==============================
df["ProjSwing+"] = 0.7 * df["Swing+"] + 0.3 * df["PowerIndex+"]
df["GapPotential"] = df["ProjSwing+"] - df["Swing+"]

# ==============================
# SUMMARY OUTPUT
# ==============================
summary_cols = ["Name", "Age", "Swing+", "PowerIndex+", "ProjSwing+", "GapPotential"]
summary_df = df[summary_cols].sort_values("ProjSwing+", ascending=False)

print("\n=== Top 10 Projectable Hitters (ProjSwing+) ===")
print(summary_df.head(10).to_string(index=False))

# ==============================
# VISUALIZATION (SAVE ONLY)
# ==============================
plt.figure(figsize=(9, 7))
sns.scatterplot(
    data=df,
    x="Swing+",
    y="ProjSwing+",
    hue="PowerIndex+",
    palette="coolwarm",
    s=80,
    edgecolor="black"
)
plt.title("Swing+ vs ProjSwing+ (Colored by Power Potential)", fontsize=15, weight="bold")
plt.xlabel("Current Swing+ (Efficiency)")
plt.ylabel("ProjSwing+ (Scalable Potential)")
plt.axhline(100, color="gray", linestyle="--", lw=1)
plt.axvline(100, color="gray", linestyle="--", lw=1)
plt.tight_layout()

# ==============================
# CREATE UNDER-25 LEADERBOARDS
# ==============================
under25 = df[df["Age"] < 25].copy()

power_tbl = (
    under25[["Name", "Age", "Swing+", "PowerIndex+", "ProjSwing+"]]
    .sort_values("PowerIndex+", ascending=False)
    .head(10)
)

proj_tbl = (
    under25[["Name", "Age", "Swing+", "PowerIndex+", "ProjSwing+"]]
    .sort_values("ProjSwing+", ascending=False)
    .head(10)
)


if SAVE_PLOT:
    plt.savefig(PLOT_FILE, dpi=300)
    plt.close()  # ✅ don't display in PyCharm
    print(f"✅ Plot saved as: {PLOT_FILE}")


from great_tables import GT

def gt_leaderboard(df_tbl, title, highlight_col, filename):
    # Keep only top 10 and round decimals
    df_tbl = df_tbl.head(10).reset_index(drop=True)
    df_tbl = df_tbl.round(1)

    # Build GT table
    gt = (
        GT(df_tbl)
        .tab_header(title=title)
        .fmt_number(columns=["Swing+", "PowerIndex+", "ProjSwing+"], decimals=1)
        .data_color(
            columns=[highlight_col],
            palette="YlOrBr",
            reverse=False
        )
        .opt_table_font("Arial")
        .opt_align_table_header("center")
        .opt_row_striping()
        .tab_options(
            table_font_size="12pt",
            heading_title_font_size="14pt",
            heading_title_font_weight="bold",
            data_row_padding="4px",
            table_background_color="white",
            column_labels_background_color="#F3F3F3",
            column_labels_font_weight="bold"
        )
    )

    # Disable table lines if supported (keeps backward compatibility)
    if hasattr(gt, "opt_table_lines"):
        gt = gt.opt_table_lines("none")

    gt.save(filename)
    print(f"✅ Saved polished leaderboard: {filename}")

# === Create and save GT tables ===
gt_leaderboard(proj_tbl, "Top Under-25 ProjSwing+ Hitters", "ProjSwing+", "Under25_ProjSwingPlus.png")
gt_leaderboard(power_tbl, "Top Under-25 PowerIndex+ Hitters", "PowerIndex+", "Under25_PowerIndexPlus.png")

