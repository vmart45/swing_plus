import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score
import shap
import matplotlib.pyplot as plt
import joblib

# === Load merged dataset ===
df = pd.read_csv("bat_tracking_merged.csv")

# === Clean bat speed (if not already done) ===
if 'avg_bat_speed_y' in df.columns:
    df = df.drop(columns=['avg_bat_speed_y'], errors='ignore')
    df = df.rename(columns={'avg_bat_speed_x': 'avg_bat_speed'})

# === Clean name format ('Last, First' → 'First Last') ===
if 'last_name, first_name' in df.columns:
    df['Name'] = df['last_name, first_name'].apply(
        lambda x: ' '.join([part.strip() for part in x.split(',')[::-1]]) if isinstance(x, str) else x
    )
    name_col = 'Name'
else:
    # Fallback if not present
    name_col = 'name' if 'name' in df.columns else (
        'name_x' if 'name_x' in df.columns else 'name_y'
    )

# === Select features and target ===
features = [
    'avg_bat_speed',
    'swing_tilt',
    'attack_angle',
    'attack_direction',
    'avg_intercept_y_vs_plate',
    'avg_intercept_y_vs_batter',
    'avg_batter_y_position',
    'avg_batter_x_position',
    'swing_length'
]

target = 'est_woba'

# Drop missing values
df = df.dropna(subset=features + [target])

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# === LightGBM Model ===
model = LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=5, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "swingplus_model.pkl")


# === Model Performance ===
preds = model.predict(X_test)
r2 = r2_score(y_test, preds)
print(f"R² on test set: {r2:.3f}")

# === Feature Importance ===
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop Features:")
print(importance_df)

# === Compute Swing+ ===
df['xwOBA_pred'] = model.predict(df[features])
mean_pred = df['xwOBA_pred'].mean()
std_pred = df['xwOBA_pred'].std()
df['Swing+'] = 100 + ((df['xwOBA_pred'] - mean_pred) / std_pred) * 10

# === Export results ===
output_cols = ['id', name_col, 'est_woba', 'xwOBA_pred', 'Swing+']
df_out = df[output_cols].sort_values('Swing+', ascending=False)
df_out.to_csv("SwingPlus_results.csv", index=False)
print(f"\nSaved Swing+ results to SwingPlus_results.csv using name column: {name_col}")

# === SHAP summary plot ===
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig("SwingPlus_SHAP.png", dpi=300)
plt.close()
print("\nSaved SHAP summary plot as SwingPlus_SHAP.png")

# === Scatterplot: Swing+ vs est_woba ===
plt.figure(figsize=(10, 7))
plt.scatter(df['Swing+'], df['est_woba'], alpha=0.7, edgecolor='k')

# Fit and plot trendline
z = np.polyfit(df['Swing+'], df['est_woba'], 1)
p = np.poly1d(z)
plt.plot(df['Swing+'], p(df['Swing+']), "r--", label='Trendline')

plt.title("Swing+ vs Expected wOBA", fontsize=16, weight='bold')
plt.xlabel("Swing+", fontsize=14)
plt.ylabel("Expected wOBA", fontsize=14)
plt.grid(alpha=0.3)
plt.legend()

# Label top and bottom performers by Swing+
top_outliers = df.nlargest(3, 'Swing+')
bottom_outliers = df.nsmallest(3, 'Swing+')
for _, row in pd.concat([top_outliers, bottom_outliers]).iterrows():
    plt.annotate(row[name_col],
                 (row['Swing+'], row['est_woba']),
                 textcoords="offset points", xytext=(5,5),
                 ha='left', fontsize=9, weight='bold')

plt.tight_layout()
plt.savefig("SwingPlus_vs_estwOBA.png", dpi=300)
plt.close()
print("\nSaved scatterplot: SwingPlus_vs_estwOBA.png")

# === Scatterplot with outliers (residual-based) ===
plt.figure(figsize=(10, 7))
plt.scatter(df['Swing+'], df['est_woba'], alpha=0.7, edgecolor='k')

# Trendline again
plt.plot(df['Swing+'], p(df['Swing+']), "r--", label='Trendline')

# Compute residuals
df['residual'] = df['est_woba'] - p(df['Swing+'])

# Label top/bottom 5 residuals
top_resid = df.nlargest(5, 'residual')
bottom_resid = df.nsmallest(5, 'residual')

for _, row in top_resid.iterrows():
    plt.annotate(row[name_col],
                 (row['Swing+'], row['est_woba']),
                 textcoords="offset points", xytext=(5,8),
                 ha='left', fontsize=9, weight='bold', color='green')

for _, row in bottom_resid.iterrows():
    plt.annotate(row[name_col],
                 (row['Swing+'], row['est_woba']),
                 textcoords="offset points", xytext=(5,-10),
                 ha='left', fontsize=9, weight='bold', color='red')

plt.title("Swing+ vs Expected wOBA (Outliers Highlighted)", fontsize=16, weight='bold')
plt.xlabel("Swing+", fontsize=14)
plt.ylabel("Expected wOBA", fontsize=14)
plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig("SwingPlus_vs_estwOBA_outliers.png", dpi=300)
plt.close()
print("\nSaved scatterplot with outliers: SwingPlus_vs_estwOBA_outliers.png")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Assuming df already includes columns: Swing+, est_woba, and Name

plt.figure(figsize=(10, 7))
plt.scatter(df['Swing+'], df['est_woba'], alpha=0.7, edgecolor='k')

# Fit and plot trendline
z = np.polyfit(df['Swing+'], df['est_woba'], 1)
p = np.poly1d(z)
plt.plot(df['Swing+'], p(df['Swing+']), "r--", label='Trendline')

# Compute residuals (vertical distance from the trendline)
df['residual'] = df['est_woba'] - p(df['Swing+'])
abs_resid = df['residual'].abs()

# Select top 10 farthest points from the trendline
outliers = df.loc[abs_resid.nlargest(10).index]

# Annotate those outliers
for _, row in outliers.iterrows():
    plt.annotate(row['Name'],
                 (row['Swing+'], row['est_woba']),
                 textcoords="offset points",
                 xytext=(6, 4),
                 ha='left',
                 fontsize=9,
                 weight='bold',
                 color='darkred')

plt.title("Swing+ vs Expected wOBA — Largest Deviations", fontsize=16, weight='bold')
plt.xlabel("Swing+", fontsize=14)
plt.ylabel("Expected wOBA", fontsize=14)
plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig("SwingPlus_vs_estwOBA_resid_outliers.png", dpi=300)
plt.close()

print("\nSaved: SwingPlus_vs_estwOBA_resid_outliers.png")

import requests
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# === 1️⃣ Pull player ages using MLBAM currentAge ===
def get_player_age(mlbam_id):
    """Fetch currentAge from MLB Stats API given a player ID."""
    try:
        url = f"https://statsapi.mlb.com/api/v1/people/{int(mlbam_id)}"
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            data = r.json()
            people = data.get("people", [])
            if people and "currentAge" in people[0]:
                return people[0]["currentAge"]
    except Exception:
        pass
    return np.nan

print("Fetching player ages from MLBAM (using currentAge)...")
df["Age"] = df["id"].apply(get_player_age)
print("✅ Done fetching ages.")

# Save updated dataset
df.to_csv("SwingPlus_with_age.csv", index=False)
print("Saved SwingPlus_with_age.csv")

# === 2️⃣ Plot: Age vs Swing+ ===
plt.figure(figsize=(10, 7))
plt.scatter(df["Age"], df["Swing+"], alpha=0.7, edgecolor="k")

# Fit & plot trendline
z = np.polyfit(df["Age"].dropna(), df["Swing+"].dropna(), 1)
p = np.poly1d(z)
plt.plot(df["Age"], p(df["Age"]), "r--", label="Trendline")

plt.title("Age vs Swing+", fontsize=16, weight="bold")
plt.xlabel("Age", fontsize=14)
plt.ylabel("Swing+", fontsize=14)
plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig("Age_vs_SwingPlus.png", dpi=300)
plt.close()
print("\nSaved: Age_vs_SwingPlus.png")

# === 3️⃣ Bar Chart: Top Swing+ under age 25 ===
young = df[df["Age"] < 25].sort_values("Swing+", ascending=False)

plt.figure(figsize=(10, 7))
plt.barh(young["Name"].head(15)[::-1], young["Swing+"].head(15)[::-1], color="royalblue")
plt.title("Top Swing+ Players Under Age 25", fontsize=16, weight="bold")
plt.xlabel("Swing+", fontsize=14)
plt.ylabel("Player", fontsize=14)
plt.tight_layout()
plt.savefig("Top_SwingPlus_Under25.png", dpi=300)
plt.close()
print("\nSaved: Top_SwingPlus_Under25.png")
