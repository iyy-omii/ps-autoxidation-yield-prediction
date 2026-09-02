import os
import copy
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from ngboost import NGBRegressor
from ngboost.distns import Normal


df = pd.read_excel("Polystyrene_autoxidation.xlsx", sheet_name=0, header=0)

df = df.dropna().reset_index(drop=True)

target_col = "Yield (%)"
feature_cols = [c for c in df.columns if c != target_col]

df["condition_id"] = df.groupby(feature_cols).ngroup()

grouped = df.groupby("condition_id", sort=False)
condition_summary = []
for cid, grp in grouped:
    condition_summary.append({
        "condition_id": cid,
        **{col: grp[col].iloc[0] for col in feature_cols},
        "n_repeats": len(grp),
        "row_indices": grp.index.tolist(),
        "Yield (%)": grp[target_col].tolist(),
    })

df_conditions = pd.DataFrame(condition_summary)

print("=" * 80)
print(f"n_rows: {len(df)}")
print(f"n_unique_conditions: {df['condition_id'].nunique()}")
print(
    f"repeats_per_condition: min={df_conditions['n_repeats'].min()}, "
    f"max={df_conditions['n_repeats'].max()}, "
    f"mean={df_conditions['n_repeats'].mean():.2f}"
)
print("=" * 80)

print(df.head())
print(df_conditions.head())


X = df[feature_cols].copy()
y = df[target_col].copy()
groups = df["condition_id"].copy()

input_dim = X.shape[1]
unique_groups = groups.unique()

print(f"n_features: {input_dim}")
print(f"n_unique_groups: {len(unique_groups)}")


class HeteroMLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden1=64, hidden2=64, dropout=0.1):
        super().__init__()
        d_in, h1, h2 = input_dim, hidden1, hidden2
        self.trunk = nn.Sequential(
            nn.Linear(d_in, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(h2, 1)
        self.log_var_head = nn.Linear(h2, 1)

    def forward(self, x):
        z = self.trunk(x)
        mu = self.mu_head(z)
        log_var = self.log_var_head(z).clamp(-5.0, 5.0)
        return mu, log_var


class HeteroMLPMuOnly(nn.Module):
    def __init__(self, m: "HeteroMLPRegressor"):
        super().__init__()
        self.inner = m

    def forward(self, x):
        mu, _ = self.inner(x)
        return mu


def heteroscedastic_loss(mu, log_var, y):
    var = torch.exp(log_var)
    return (log_var + (y - mu) ** 2 / var).mean()


def fit_stage1_mu_mse(model, X, y, epochs, lr, weight_decay):
    for p in model.log_var_head.parameters():
        p.requires_grad = False
    trainable = list(model.trunk.parameters()) + list(model.mu_head.parameters())
    opt = torch.optim.Adam(trainable, lr=lr, weight_decay=weight_decay)
    X_t = torch.FloatTensor(X)
    y_t = torch.FloatTensor(y).reshape(-1, 1)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        mu, _ = model(X_t)
        loss = F.mse_loss(mu, y_t)
        loss.backward()
        opt.step()


def fit_stage2_hetero(model, X, y, epochs, lr, weight_decay):
    for p in model.trunk.parameters():
        p.requires_grad = False
    for p in model.mu_head.parameters():
        p.requires_grad = False
    for p in model.log_var_head.parameters():
        p.requires_grad = True
    opt = torch.optim.Adam(model.log_var_head.parameters(), lr=lr, weight_decay=weight_decay)
    X_t = torch.FloatTensor(X)
    y_t = torch.FloatTensor(y).reshape(-1, 1)
    for _ in range(epochs):
        model.eval()
        opt.zero_grad()
        mu, log_var = model(X_t)
        loss = heteroscedastic_loss(mu, log_var, y_t)
        loss.backward()
        opt.step()


def fit_hetero_two_stage(model, X, y, epochs_stage1, epochs_stage2, lr, weight_decay):
    fit_stage1_mu_mse(model, X, y, epochs_stage1, lr, weight_decay)
    fit_stage2_hetero(model, X, y, epochs_stage2, lr, weight_decay)


def predict_hetero_mu_logvar(model, X):
    model.eval()
    with torch.no_grad():
        mu, log_var = model(torch.FloatTensor(X))
    return mu.numpy().flatten(), log_var.numpy().flatten()


NGB_WEIGHT = 0.5
MLP_WEIGHT = 0.5
NGB_VAR_WEIGHT = 0.5
MLP_VAR_WEIGHT = 0.5

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

ngb_params = {
    "Dist": Normal,
    "n_estimators": 200,
    "learning_rate": 8e-2,
    "Base": DecisionTreeRegressor(max_depth=3),
    "random_state": RANDOM_SEED,
    "verbose": 0,
}

mlp_params = {
    "hidden1": 32,
    "hidden2": 32,
    "dropout": 0.1,
    "epochs_stage1": 80,
    "epochs_stage2": 20,
    "lr": 5e-2,
    "weight_decay": 1e-4,
}

print(f"Ensemble = {NGB_WEIGHT} * NGBoost + {MLP_WEIGHT} * MLP (mu)")


TRAIN_RATIO = 0.1
TEST_RATIO = 1.0 - TRAIN_RATIO

if not np.isclose(TRAIN_RATIO + TEST_RATIO, 1.0):
    raise ValueError("TRAIN_RATIO + TEST_RATIO must equal 1.0")
if not (0.0 < TRAIN_RATIO < 1.0):
    raise ValueError("TRAIN_RATIO must be between 0 and 1")

MAX_SPLIT_RETRIES = 30
MIN_TRAIN_ROWS = 5
MIN_TEST_ROWS = 5

print(f"train:test = {int(TRAIN_RATIO*100)}:{int(TEST_RATIO*100)}")


ratio_label = f"train_{int(TRAIN_RATIO*100):02d}_test_{int(TEST_RATIO*100):02d}"


def split_groups_by_ratio(groups, train_ratio, seed):
    unique_group_ids = groups.unique()
    group_train_ids, group_test_ids = train_test_split(
        unique_group_ids,
        train_size=train_ratio,
        random_state=seed,
        shuffle=True,
    )
    return np.array(group_train_ids), np.array(group_test_ids)


def sanitize_split(X_part, y_part, group_part, split_name):
    valid_mask = y_part.notna() & X_part.notna().all(axis=1)
    dropped = int((~valid_mask).sum())
    if dropped > 0:
        print(f"{split_name}: dropped {dropped} row(s) with missing values")
    X_clean = X_part.loc[valid_mask].reset_index(drop=True)
    y_clean = y_part.loc[valid_mask].reset_index(drop=True)
    group_clean = group_part.loc[valid_mask].reset_index(drop=True)
    return X_clean, y_clean, group_clean


split_ok = False
for attempt in range(1, MAX_SPLIT_RETRIES + 1):
    split_seed = RANDOM_SEED + attempt
    train_group_ids, test_group_ids = split_groups_by_ratio(
        groups=groups, train_ratio=TRAIN_RATIO, seed=split_seed
    )

    train_mask = groups.isin(train_group_ids)
    test_mask = groups.isin(test_group_ids)

    X_train = X.loc[train_mask].reset_index(drop=True)
    X_test = X.loc[test_mask].reset_index(drop=True)
    y_train = y.loc[train_mask].reset_index(drop=True)
    y_test = y.loc[test_mask].reset_index(drop=True)

    group_train = groups.loc[train_mask].reset_index(drop=True)
    group_test = groups.loc[test_mask].reset_index(drop=True)

    X_train, y_train, group_train = sanitize_split(X_train, y_train, group_train, "train")
    X_test, y_test, group_test = sanitize_split(X_test, y_test, group_test, "test")

    overlap = set(group_train.unique()) & set(group_test.unique())
    if len(overlap) != 0:
        continue
    if len(X_train) < MIN_TRAIN_ROWS or len(X_test) < MIN_TEST_ROWS:
        continue

    split_ok = True
    break

if not split_ok:
    raise ValueError(
        f"Could not draw a valid split for TRAIN_RATIO={TRAIN_RATIO} in {MAX_SPLIT_RETRIES} attempts."
    )

print(f"n_train={len(X_train)} n_test={len(X_test)} (seed={split_seed})")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
X_test_scaled = scaler.transform(X_test).astype(np.float32)

y_train_np = y_train.values.astype(np.float32)
y_test_np = y_test.values.astype(np.float32)

ngb_model = NGBRegressor(**ngb_params)
ngb_model.fit(
    np.asarray(X_train_scaled, dtype=np.float64),
    np.asarray(y_train_np, dtype=np.float64),
)

pd_test = ngb_model.pred_dist(np.asarray(X_test_scaled, dtype=np.float64))
pred_ngb = np.asarray(pd_test.loc, dtype=np.float32).reshape(-1)
sigma_ngb_test = np.asarray(pd_test.scale, dtype=np.float32).reshape(-1)
pred_var_ngb = np.maximum(sigma_ngb_test**2, 1e-8)

mse_ngb = mean_squared_error(y_test_np, pred_ngb)
mae_ngb = mean_absolute_error(y_test_np, pred_ngb)

mlp_model = HeteroMLPRegressor(
    input_dim=input_dim,
    hidden1=mlp_params["hidden1"],
    hidden2=mlp_params["hidden2"],
    dropout=mlp_params["dropout"],
)

fit_hetero_two_stage(
    mlp_model,
    X_train_scaled,
    y_train_np,
    epochs_stage1=mlp_params["epochs_stage1"],
    epochs_stage2=mlp_params["epochs_stage2"],
    lr=mlp_params["lr"],
    weight_decay=mlp_params["weight_decay"],
)

pred_mlp_mu, log_var_mlp_test = predict_hetero_mu_logvar(mlp_model, X_test_scaled)
pred_mlp = pred_mlp_mu

mse_mlp = mean_squared_error(y_test_np, pred_mlp)
mae_mlp = mean_absolute_error(y_test_np, pred_mlp)

sigma_mlp = np.exp(0.5 * log_var_mlp_test)
pred_var_mlp = np.maximum(sigma_mlp**2, 1e-8)
sigma_ngb_var = sigma_ngb_test
sigma_ensemble = np.sqrt(
    MLP_VAR_WEIGHT * (sigma_mlp**2) + NGB_VAR_WEIGHT * pred_var_ngb
)

pred_ensemble = NGB_WEIGHT * pred_ngb + MLP_WEIGHT * pred_mlp

mse_ensemble = mean_squared_error(y_test_np, pred_ensemble)
mae_ensemble = mean_absolute_error(y_test_np, pred_ensemble)

ci_lower_95 = pred_ensemble - 1.96 * sigma_ensemble
ci_upper_95 = pred_ensemble + 1.96 * sigma_ensemble

result_df = X_test.copy()
result_df[target_col] = y_test_np
result_df["pred_ngb"] = pred_ngb
result_df["pred_mlp"] = pred_mlp
result_df["pred_ensemble"] = pred_ensemble
result_df["abs_error_ngb"] = np.abs(y_test_np - pred_ngb)
result_df["abs_error_mlp"] = np.abs(y_test_np - pred_mlp)
result_df["abs_error_ensemble"] = np.abs(y_test_np - pred_ensemble)
result_df["split_label"] = ratio_label
result_df["sample_id"] = X.loc[test_mask].index.values[: len(result_df)]
result_df["sigma_mlp"] = sigma_mlp
result_df["sigma_ngb_var"] = sigma_ngb_var
result_df["sigma_ensemble"] = sigma_ensemble
result_df["pred_var_mlp"] = pred_var_mlp
result_df["pred_var_ngb"] = pred_var_ngb
result_df["log_var_mlp"] = log_var_mlp_test
result_df["ci_lower_95"] = ci_lower_95
result_df["ci_upper_95"] = ci_upper_95

run_summary = {
    "split_label": ratio_label,
    "train_ratio_target": TRAIN_RATIO,
    "test_ratio_target": TEST_RATIO,
    "n_train_rows": len(X_train),
    "n_test_rows": len(X_test),
    "train_ratio": len(X_train) / len(X),
    "test_ratio": len(X_test) / len(X),
    "ngb_mse": mse_ngb,
    "ngb_mae": mae_ngb,
    "mlp_mse": mse_mlp,
    "mlp_mae": mae_mlp,
    "ensemble_mse": mse_ensemble,
    "ensemble_mae": mae_ensemble,
}
run_summary_df = pd.DataFrame([run_summary])

print()
print("=" * 60)
print(f"Ratio {ratio_label} -- Figure 6 / Table S2 row:")
print(f"  Ensemble: MAE {mae_ensemble:.4f} | MSE {mse_ensemble:.4f}")
print(f"  NGBoost : MAE {mae_ngb:.4f}")
print(f"  MLP     : MAE {mae_mlp:.4f}")
print("=" * 60)


print(run_summary_df)


_cols = [
    target_col,
    "pred_ensemble",
    "sigma_mlp",
    "sigma_ngb_var",
    "sigma_ensemble",
    "pred_var_mlp",
    "pred_var_ngb",
    "ci_lower_95",
    "ci_upper_95",
]
_cols = [c for c in _cols if c in result_df.columns]
print(result_df[_cols].head(30))
print("mean sigma_ensemble:", result_df["sigma_ensemble"].mean())


save_dir = f"ratio_run_{ratio_label}"
os.makedirs(save_dir, exist_ok=True)

excel_path = os.path.join(save_dir, "ratio_run_results.xlsx")

_export_summary = run_summary_df.copy()
_export_pred = result_df.copy()
numeric_cols = _export_summary.select_dtypes(include=[np.number]).columns
_export_summary[numeric_cols] = _export_summary[numeric_cols].round(4)
numeric_cols = _export_pred.select_dtypes(include=[np.number]).columns
_export_pred[numeric_cols] = _export_pred[numeric_cols].round(4)

with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    _export_summary.to_excel(writer, sheet_name="run_summary", index=False)
    _export_pred.to_excel(writer, sheet_name="test_rows_full", index=False)

print(f"Excel written: {excel_path}")


checkpoint_dir = os.path.join(save_dir, "checkpoint")
os.makedirs(checkpoint_dir, exist_ok=True)

joblib.dump(scaler, os.path.join(checkpoint_dir, "scaler.pkl"))
joblib.dump(ngb_model, os.path.join(checkpoint_dir, "ngb_model.pkl"))
torch.save(mlp_model.state_dict(), os.path.join(checkpoint_dir, "mlp_model.pt"))
joblib.dump(mlp_params, os.path.join(checkpoint_dir, "mlp_params.pkl"))

joblib.dump(
    {
        "split_label": ratio_label,
        "NGB_WEIGHT": NGB_WEIGHT,
        "MLP_WEIGHT": MLP_WEIGHT,
        "NGB_VAR_WEIGHT": NGB_VAR_WEIGHT,
        "MLP_VAR_WEIGHT": MLP_VAR_WEIGHT,
        "feature_cols": list(feature_cols),
        "target_col": target_col,
        "input_dim": input_dim,
        "ngb_params": copy.deepcopy(ngb_params),
    },
    os.path.join(checkpoint_dir, "inference_meta.pkl"),
)

print(f"checkpoint saved: {checkpoint_dir}")


scaler = joblib.load(os.path.join(checkpoint_dir, "scaler.pkl"))
mlp_params = joblib.load(os.path.join(checkpoint_dir, "mlp_params.pkl"))
meta = joblib.load(os.path.join(checkpoint_dir, "inference_meta.pkl"))

NGB_WEIGHT = meta["NGB_WEIGHT"]
MLP_WEIGHT = meta["MLP_WEIGHT"]
NGB_VAR_WEIGHT = meta["NGB_VAR_WEIGHT"]
MLP_VAR_WEIGHT = meta["MLP_VAR_WEIGHT"]
input_dim = meta["input_dim"]
feature_cols = meta["feature_cols"]

ngb_model = joblib.load(os.path.join(checkpoint_dir, "ngb_model.pkl"))

mlp_model = HeteroMLPRegressor(
    input_dim=input_dim,
    hidden1=mlp_params["hidden1"],
    hidden2=mlp_params["hidden2"],
    dropout=mlp_params["dropout"],
)
mlp_model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "mlp_model.pt")))
