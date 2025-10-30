import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt

# === CONFIG ===
DATA_PATH = "bat_tracking_merged.csv"   # File with biomechanical features
TARGET_PLAYER = "James Wood"            # Player to find comps for
TOP_N = 10                              # Number of similar players to return
SAVE_HEATMAP = True                     # Toggle heatmap output

# === LOAD DATA ===
df = pd.read_csv(DATA_PATH)

# === Identify name column ===
if "Player" in df.columns:
    name_col = "Player"
elif "Name" in df.columns:
    name_col = "Name"
elif "last_name, first_name" in df.columns:
    df["Name"] = df["last_name, first_name"].apply(
        lambda x: " ".join([p.strip() for p in x.split(",")[::-1]])
    )
    name_col = "Name"
else:
    raise ValueError("No player name column found.")

# === Select mechanical features ===
features = [
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

# Drop rows with missing values
df = df.dropna(subset=features + [name_col]).reset_index(drop=True)

# === Standardize features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# === Compute cosine similarity matrix ===
similarity_matrix = cosine_similarity(X_scaled)
similarity_df = pd.DataFrame(
    similarity_matrix, index=df[name_col], columns=df[name_col]
)

# === Retrieve comps for target ===
if TARGET_PLAYER not in similarity_df.index:
    raise ValueError(f"{TARGET_PLAYER} not found in dataset.")

similar_players = (
    similarity_df.loc[TARGET_PLAYER]
    .sort_values(ascending=False)
    .iloc[1 : TOP_N + 1]  # exclude self
)

print(f"\nTop {TOP_N} mechanically similar players to {TARGET_PLAYER}:")
print(similar_players)

# === Optional: save similarity heatmap ===
if SAVE_HEATMAP:
    top_names = [TARGET_PLAYER] + list(similar_players.index)
    heatmap_data = similarity_df.loc[top_names, top_names]

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        cbar_kws={"label": "Cosine Similarity"},
    )
    plt.title(f"Mechanical Similarity Cluster: {TARGET_PLAYER}", fontsize=14, weight="bold")
    plt.tight_layout()
    plt.savefig(f"{TARGET_PLAYER.replace(' ', '_')}_Mechanical_Similarity.png", dpi=300)
    plt.close()
    print(f"✅ Saved: {TARGET_PLAYER.replace(' ', '_')}_Mechanical_Similarity.png")

print("\nAnalysis complete — mechanical comps calculated using cosine similarity.")
