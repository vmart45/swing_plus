import pandas as pd
import numpy as np
import joblib
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score
import shap
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# Load data
# ============================================================
df = pd.read_csv("bat_merged.csv")

# ============================================================
# Shared features
# ============================================================
features = [
    'avg_bat_speed',
    'swing_tilt',
    'attack_angle',
    'attack_direction',
    'avg_intercept_y_vs_plate',
    'avg_intercept_y_vs_batter',
    'avg_batter_y_position',
    'avg_batter_x_position',
    'swing_length',
    'avg_foot_sep',
    'avg_stance_angle'
]

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=features)

# ============================================================
# Detect BIP column
# ============================================================
bip_col = None
for c in ["BIP", "bip", "BallsInPlay"]:
    if c in df.columns:
        bip_col = c
        break

if bip_col is None:
    raise ValueError("Missing BIP column.")

if "pa" not in df.columns:
    raise ValueError("Missing 'pa' column for SwingPlus model.")


# ============================================================
# Train model utility
# ============================================================
def train_metric(df, features, target, weight_col, pred_name, model_file):
    print(f"\n==================== Training: {pred_name} ====================")

    df2 = df.dropna(subset=features + [target, weight_col]).copy()
    df2["sample_weight"] = np.sqrt(np.maximum(df2[weight_col], 0))

    X = df2[features]
    y = df2[target]
    w = df2["sample_weight"]

    model = LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.01,
        max_depth=-1,
        num_leaves=31,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        random_state=42
    )

    model.fit(X, y, sample_weight=w)
    joblib.dump(model, model_file)

    preds = model.predict(X)

    r2_unweighted = r2_score(y, preds)
    r2_weighted = r2_score(y, preds, sample_weight=w)

    print(f"R² (unweighted): {r2_unweighted:.4f}")
    print(f"R² (weighted):   {r2_weighted:.4f}")

    df2[pred_name] = preds
    return df2[[pred_name]], model, X, y


# ============================================================
# Model 1 — Swing+ (est_woba → xwoba_pred)
# ============================================================
df_sw, model_sw, X_sw, y_sw = train_metric(
    df=df,
    features=features,
    target="est_woba",
    weight_col="pa",
    pred_name="xwoba_pred",
    model_file="SwingPlus.pkl"
)

# ============================================================
# Model 2 — HitSkill+ (est_ba → xba_pred)
# ============================================================
df_ct, model_ct, X_ct, y_ct = train_metric(
    df=df,
    features=features,
    target="est_ba",
    weight_col=bip_col,
    pred_name="xba_pred",
    model_file="HitSkillPlus.pkl"
)

# ============================================================
# Model 3 — Impact+ (est_slg → xslg_pred)
# ============================================================
df_pw, model_pw, X_pw, y_pw = train_metric(
    df=df,
    features=features,
    target="est_slg",
    weight_col=bip_col,
    pred_name="xslg_pred",
    model_file="ImpactPlus.pkl"
)

# ============================================================
# Merge predictions into df
# ============================================================
df_out = df.copy()
df_out = df_out.join(df_sw, how="left")
df_out = df_out.join(df_ct, how="left")
df_out = df_out.join(df_pw, how="left")


# ============================================================
# Create + metrics (100 mean, ±10 per SD)
# ============================================================
def create_plus_metric(df, pred_col, plus_col):
    m = df[pred_col].mean()
    s = df[pred_col].std()
    df[plus_col] = 100 + ((df[pred_col] - m) / s) * 10

create_plus_metric(df_out, "xwoba_pred", "SwingPlus")
create_plus_metric(df_out, "xba_pred", "HitSkillPlus")
create_plus_metric(df_out, "xslg_pred", "ImpactPlus")


# ============================================================
# FAST SHAP PLOTS (TreeExplainer + Sampling)
# ============================================================
def shap_plot(model, X, name):
    explainer = shap.TreeExplainer(model)

    # sample up to 500 rows for speed
    Xs = X.sample(min(500, len(X)), random_state=42)

    shap_values = explainer.shap_values(Xs)

    shap.summary_plot(shap_values, Xs, show=False)
    plt.tight_layout()
    plt.savefig(f"{name}_SHAP.png", dpi=300)
    plt.close()

    print(f"Saved SHAP plot: {name}_SHAP.png")


shap_plot(model_sw, X_sw, "SwingPlus")
shap_plot(model_ct, X_ct, "HitSkillPlus")
shap_plot(model_pw, X_pw, "ImpactPlus")


# ============================================================
# Safe scatter plot (no SVD crash)
# ============================================================
def scatter_plot(df, pred, target, title):
    tmp = df[[pred, target]].dropna()
    tmp = tmp[np.isfinite(tmp[pred]) & np.isfinite(tmp[target])]

    plt.figure(figsize=(10,7))
    plt.scatter(tmp[pred], tmp[target], alpha=0.7, edgecolor='k')

    if len(tmp) > 2:
        z = np.polyfit(tmp[pred], tmp[target], 1)
        p = np.poly1d(z)
        plt.plot(tmp[pred], p(tmp[pred]), "r--")

    plt.title(title, fontsize=16, weight="bold")
    plt.xlabel(pred, fontsize=14)
    plt.ylabel(target, fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png", dpi=300)
    plt.close()

    print(f"Saved scatterplot: {title}.png")


scatter_plot(df_out, "xwoba_pred", "est_woba", "SwingPlus_vs_est_woba")
scatter_plot(df_out, "xba_pred", "est_ba", "HitSkillPlus_vs_est_ba")
scatter_plot(df_out, "xslg_pred", "est_slg", "ImpactPlus_vs_est_slg")


# ============================================================
# Save final output
# ============================================================
df_out.to_csv("BatTracking_Metrics_AllModels.csv", index=False)
print("\nSaved BatTracking_Metrics_AllModels.csv")

print(df_out[[
    "xwoba_pred", "SwingPlus",
    "xba_pred", "HitSkillPlus",
    "xslg_pred", "ImpactPlus"
]].head(10))
