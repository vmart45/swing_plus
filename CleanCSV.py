import pandas as pd
from pybaseball import playerid_reverse_lookup, batting_stats, pitching_stats

# ============================================================
# 1. Load BatTracking (SwingPlus, HitSkillPlus, ImpactPlus) dataset
# ============================================================
df = pd.read_csv("BatTracking_Metrics_AllModels.csv")

# ============================================================
# Fix year column before anything else
# ============================================================
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df = df.dropna(subset=["year"])
df["year"] = df["year"].astype(int)

# Fix id column the same way
df["id"] = pd.to_numeric(df["id"], errors="coerce")
df = df.dropna(subset=["id"])
df["id"] = df["id"].astype(int)

# ============================================================
# 2. MLBAM → FanGraphs ID mapping
# ============================================================
mlbam_ids = df["id"].unique().tolist()

id_map = playerid_reverse_lookup(mlbam_ids, key_type="mlbam").rename(columns={
    "key_mlbam": "id",
    "key_fangraphs": "fangraphsID",
    "name_first": "first_name",
    "name_last": "last_name"
})[["id", "fangraphsID", "first_name", "last_name"]]

df = df.merge(id_map, on="id", how="left")

# ============================================================
# 3. Pull FanGraphs data (batters + pitchers)
# ============================================================
years = sorted(df["year"].unique())

batter_data = pd.concat([batting_stats(y, qual=0) for y in years], ignore_index=True)
batter_data = batter_data.rename(columns={"IDfg": "fangraphsID", "Season": "year"})[
    ["fangraphsID", "year", "Team", "Age"]
]

pitcher_data = pd.concat([pitching_stats(y, qual=0) for y in years], ignore_index=True)
pitcher_data = pitcher_data.rename(columns={"IDfg": "fangraphsID", "Season": "year"})[
    ["fangraphsID", "year", "Team", "Age"]
]

merged_fg = pd.concat([batter_data, pitcher_data], ignore_index=True)
merged_fg = merged_fg.drop_duplicates(subset=["fangraphsID", "year"])

# ============================================================
# 4. Merge FanGraphs stats to main dataset
# ============================================================
df["fangraphsID"] = df["fangraphsID"].astype(str)
merged_fg["fangraphsID"] = merged_fg["fangraphsID"].astype(str)

df = df.merge(merged_fg, on=["fangraphsID", "year"], how="left", suffixes=("", "_fg"))
print(f"FG merge filled: {df['Team'].notna().sum()} rows")

# ============================================================
# 5. Fallback: use FanGraphs_2023_2025_Merged.csv
# ============================================================
fg_csv = pd.read_csv("FanGraphs_2023_2025_Merged.csv")

fg_csv = fg_csv[["IDfg", "Season", "Name", "Team", "Age"]].copy()
fg_csv["Season"] = pd.to_numeric(fg_csv["Season"], errors="coerce")
fg_csv = fg_csv.dropna(subset=["Season"])
fg_csv["Season"] = fg_csv["Season"].astype(int)

# Convert "Last, First" to "First Last"
def last_first_to_first_last(x):
    if pd.isna(x):
        return None
    x = x.strip()
    if "," in x:
        last, first = x.split(",", 1)
        return f"{first.strip()} {last.strip()}"
    return x

df["Name_for_match"] = df["name"].apply(last_first_to_first_last)

fg_csv["Name_for_match"] = fg_csv["Name"].astype(str).str.strip()
fg_csv = fg_csv.rename(columns={"Season": "year"})

# Merge fallback
df = df.merge(
    fg_csv[["Name_for_match", "year", "Team", "Age"]],
    on=["Name_for_match", "year"],
    how="left",
    suffixes=("", "_fgcsv")
)

# Fill Team & Age if missing
df["Team"] = df["Team"].fillna(df["Team_fgcsv"])
df["Age"] = df["Age"].fillna(df["Age_fgcsv"])

df.drop(columns=["Team_fgcsv", "Age_fgcsv", "Name_for_match"], inplace=True)

print(f"Total Team filled after fallback: {df['Team'].notna().sum()}")

# ============================================================
# 6. Fill forward/backward by player
# ============================================================
df.sort_values(["id", "year"], inplace=True)
df["Team"] = df.groupby("id")["Team"].ffill().bfill()
df["Age"] = df.groupby("id")["Age"].ffill().bfill()

# ============================================================
# 7. Drop blank rows (safety)
# ============================================================
df = df.dropna(subset=["Team", "Age", "name"])

# ============================================================
# 8. Drop unnecessary ID/name columns
# ============================================================
cols_to_drop = ["fangraphsID", "first_name", "last_name"]
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# ============================================================
# 9. Clean Team column: replace --- / - - - / '' with TOT
# ============================================================
df["Team"] = df["Team"].replace(
    to_replace=["---", "- - -", "--", "-", ""],
    value="TOT"
)

# ============================================================
# 10. Save final cleaned dataset
# ============================================================
output_path = "SwingPlus_with_team_age_cleaned.csv"
df.to_csv(output_path, index=False)

print(f"\nSaved cleaned file: {output_path}")
print(df[["id", "year", "name", "Team", "Age"]].head(10))
