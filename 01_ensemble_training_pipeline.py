import os
import copy
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

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

print(f"n_rows: {len(df)}")
print(f"n_unique_conditions: {df['condition_id'].nunique()}")
print(
    f"repeats_per_condition: min={df_conditions['n_repeats'].min()}, "
    f"max={df_conditions['n_repeats'].max()}, "
    f"mean={df_conditions['n_repeats'].mean():.2f}"
)

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
    lr_logvar = lr * 0.01
    opt = torch.optim.Adam(model.log_var_head.parameters(), lr=lr_logvar, weight_decay=weight_decay)
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

FOLD_A_SAMPLE_IDS = [2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 20, 21, 22, 25, 26, 27, 28, 29, 30,
    31, 41, 42, 43, 44, 45, 46, 47, 50, 51, 54, 55, 56, 59, 60, 66, 67, 68, 69, 70, 71, 75,
    76, 77, 78, 86, 87, 93, 94, 97, 98, 99, 100, 101, 105, 106, 107, 108, 109, 110, 112,
    114, 116, 118, 119, 120, 121, 123, 129, 131]
FOLD_B_SAMPLE_IDS = [0, 1, 5, 6, 14, 15, 16, 17, 18, 19, 23, 24, 32, 33, 34, 35, 36, 37, 38,
    39, 40, 48, 49, 52, 53, 57, 58, 61, 62, 63, 64, 65, 72, 73, 74, 79, 80, 81, 82, 83, 84,
    85, 88, 89, 90, 91, 92, 95, 96, 102, 103, 104, 111, 113, 115, 117, 122, 124, 125, 126,
    127, 128, 130, 132, 133, 134, 135, 136]

assert set(FOLD_A_SAMPLE_IDS) & set(FOLD_B_SAMPLE_IDS) == set()
assert set(FOLD_A_SAMPLE_IDS) | set(FOLD_B_SAMPLE_IDS) == set(range(len(X)))

COMBS = [
    {"comb_idx": 1, "fold_label": "fold_A", "train_ids": FOLD_B_SAMPLE_IDS, "test_ids": FOLD_A_SAMPLE_IDS},
    {"comb_idx": 2, "fold_label": "fold_B", "train_ids": FOLD_A_SAMPLE_IDS, "test_ids": FOLD_B_SAMPLE_IDS},
]

print(f"Fold A: {len(FOLD_A_SAMPLE_IDS)} samples | Fold B: {len(FOLD_B_SAMPLE_IDS)} samples")

iter_comb_summary = []
iter_comb_result_dfs = []
iter_comb_basefold_metrics = []
iter_comb_models = []

for comb in COMBS:
    comb_idx = comb["comb_idx"]
    train_ids = np.array(comb["train_ids"])
    test_ids = np.array(comb["test_ids"])

    train_mask = X.index.isin(train_ids)
    test_mask = X.index.isin(test_ids)

    X_train = X.loc[train_mask].reset_index(drop=True)
    X_test = X.loc[test_mask].reset_index(drop=True)
    y_train = y.loc[train_mask].reset_index(drop=True)
    y_test = y.loc[test_mask].reset_index(drop=True)

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

    base_fold_metrics = [{
        "comb_idx": comb_idx,
        "test_fold_indices": comb["fold_label"],
        "base_fold": 0 if comb_idx == 1 else 1,
        "n_rows": len(X_test),
        "ngb_mse": mse_ngb,
        "ngb_mae": mae_ngb,
        "mlp_mse": mse_mlp,
        "mlp_mae": mae_mlp,
        "ensemble_mse": mse_ensemble,
        "ensemble_mae": mae_ensemble,
    }]

    iter_comb_models.append({
        "comb_idx": comb_idx,
        "test_fold_indices": comb["fold_label"],
        "scaler": copy.deepcopy(scaler),
        "ngb_model": copy.deepcopy(ngb_model),
        "mlp_state_dict": copy.deepcopy(mlp_model.state_dict()),
        "mlp_params": copy.deepcopy(mlp_params),
    })

    result_df = X_test.copy()
    result_df[target_col] = y_test_np
    result_df["pred_ngb"] = pred_ngb
    result_df["pred_mlp"] = pred_mlp
    result_df["pred_ensemble"] = pred_ensemble
    result_df["abs_error_ngb"] = np.abs(y_test_np - pred_ngb)
    result_df["abs_error_mlp"] = np.abs(y_test_np - pred_mlp)
    result_df["abs_error_ensemble"] = np.abs(y_test_np - pred_ensemble)
    result_df["comb_idx"] = comb_idx
    result_df["test_fold_indices"] = comb["fold_label"]
    result_df["sample_id"] = X.loc[test_mask].index.values

    result_df["sigma_mlp"] = sigma_mlp
    result_df["sigma_ngb_var"] = sigma_ngb_var
    result_df["sigma_ensemble"] = sigma_ensemble
    result_df["pred_var_ensemble"] = np.maximum(sigma_ensemble**2, 1e-8)
    result_df["pred_var_mlp"] = pred_var_mlp
    result_df["pred_var_ngb"] = pred_var_ngb
    result_df["log_var_mlp"] = log_var_mlp_test
    result_df["ci_lower_95"] = ci_lower_95
    result_df["ci_upper_95"] = ci_upper_95

    comb_summary_row = {
        "comb_idx": comb_idx,
        "test_fold_indices": comb["fold_label"],
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

    iter_comb_summary.append(comb_summary_row)
    iter_comb_result_dfs.append(result_df.copy())
    iter_comb_basefold_metrics.append(pd.DataFrame(base_fold_metrics))

    print(f"[comb {comb_idx}] ensemble MAE={mae_ensemble:.4f} | ngb MAE={mae_ngb:.4f} | mlp MAE={mae_mlp:.4f}")

comb_summary_df = pd.DataFrame(iter_comb_summary)
all_comb_predictions_df = pd.concat(iter_comb_result_dfs, axis=0).reset_index(drop=True)
all_basefold_metrics_df = pd.concat(iter_comb_basefold_metrics, axis=0).reset_index(drop=True)
all_comb_models = iter_comb_models

mean_ensemble_mae = comb_summary_df["ensemble_mae"].mean()
std_ensemble_mae = comb_summary_df["ensemble_mae"].std()
mean_ensemble_mse = comb_summary_df["ensemble_mse"].mean()
std_ensemble_mse = comb_summary_df["ensemble_mse"].std()
mean_ngb_mae = comb_summary_df["ngb_mae"].mean()
std_ngb_mae = comb_summary_df["ngb_mae"].std()
mean_mlp_mae = comb_summary_df["mlp_mae"].mean()
std_mlp_mae = comb_summary_df["mlp_mae"].std()

print(f"Ensemble : MAE {mean_ensemble_mae:.4f} +/- {std_ensemble_mae:.4f} | MSE {mean_ensemble_mse:.4f} +/- {std_ensemble_mse:.4f}")
print(f"NGBoost  : MAE {mean_ngb_mae:.4f} +/- {std_ngb_mae:.4f}")
print(f"MLP      : MAE {mean_mlp_mae:.4f} +/- {std_mlp_mae:.4f}")
print(comb_summary_df)

group_cols = ["sample_id"] + feature_cols + [target_col]

avg_pred_df = (
    all_comb_predictions_df
    .groupby(group_cols, as_index=False)
    .agg(
        pred_ngb_mean=("pred_ngb", "mean"),
        pred_mlp_mean=("pred_mlp", "mean"),
        pred_ensemble_mean=("pred_ensemble", "mean"),
        sigma_ensemble_mean=("sigma_ensemble", "mean"),
        ci_lower_95_mean=("ci_lower_95", "mean"),
        ci_upper_95_mean=("ci_upper_95", "mean"),
        n_test_appearances=("pred_ensemble", "size")
    )
)

avg_pred_df["abs_error_ngb_mean"] = np.abs(avg_pred_df[target_col] - avg_pred_df["pred_ngb_mean"])
avg_pred_df["abs_error_mlp_mean"] = np.abs(avg_pred_df[target_col] - avg_pred_df["pred_mlp_mean"])
avg_pred_df["abs_error_ensemble_mean"] = np.abs(avg_pred_df[target_col] - avg_pred_df["pred_ensemble_mean"])

avg_mse = mean_squared_error(avg_pred_df[target_col], avg_pred_df["pred_ensemble_mean"])
avg_mae = mean_absolute_error(avg_pred_df[target_col], avg_pred_df["pred_ensemble_mean"])

avg_metrics_df = pd.DataFrame([{
    "metric_type": "averaged_over_combinations",
    "ensemble_mse": avg_mse,
    "ensemble_mae": avg_mae,
    "n_unique_samples": len(avg_pred_df),
    "mean_test_appearances": avg_pred_df["n_test_appearances"].mean(),
}])

save_dir = "reported_run"
os.makedirs(save_dir, exist_ok=True)

excel_path = os.path.join(save_dir, "reported_run_all_combinations.xlsx")

best_comb_summary_df = comb_summary_df.copy()
best_all_preds_df = all_comb_predictions_df.copy()
best_basefold_df = all_basefold_metrics_df.copy()

numeric_cols = best_comb_summary_df.select_dtypes(include=[np.number]).columns
best_comb_summary_df[numeric_cols] = best_comb_summary_df[numeric_cols].round(4)

numeric_cols = best_all_preds_df.select_dtypes(include=[np.number]).columns
best_all_preds_df[numeric_cols] = best_all_preds_df[numeric_cols].round(4)

numeric_cols = best_basefold_df.select_dtypes(include=[np.number]).columns
best_basefold_df[numeric_cols] = best_basefold_df[numeric_cols].round(4)

numeric_cols = avg_pred_df.select_dtypes(include=[np.number]).columns
avg_pred_df[numeric_cols] = avg_pred_df[numeric_cols].round(4)

numeric_cols = avg_metrics_df.select_dtypes(include=[np.number]).columns
avg_metrics_df[numeric_cols] = avg_metrics_df[numeric_cols].round(4)

with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    best_comb_summary_df.to_excel(writer, sheet_name="comb_summary", index=False)
    best_all_preds_df.to_excel(writer, sheet_name="all_predictions", index=False)
    best_basefold_df.to_excel(writer, sheet_name="basefold_metrics", index=False)

    for comb_idx in sorted(best_comb_summary_df["comb_idx"].unique()):
        comb_df = best_all_preds_df[best_all_preds_df["comb_idx"] == comb_idx].reset_index(drop=True)
        numeric_cols = comb_df.select_dtypes(include=[np.number]).columns
        comb_df[numeric_cols] = comb_df[numeric_cols].round(4)
        writer_sheet_name = f"comb_{comb_idx:02d}"
        comb_df.to_excel(writer, sheet_name=writer_sheet_name, index=False)

    avg_pred_df.to_excel(writer, sheet_name="averaged_predictions", index=False)
    avg_metrics_df.to_excel(writer, sheet_name="averaged_metrics", index=False)

print(f"Excel written: {excel_path}")

for _comb_idx in sorted(best_comb_summary_df["comb_idx"].unique()):
    _comb_pred = best_all_preds_df[best_all_preds_df["comb_idx"] == _comb_idx].reset_index(drop=True)
    _comb_sum = best_comb_summary_df[best_comb_summary_df["comb_idx"] == _comb_idx].reset_index(drop=True)
    _comb_bf = best_basefold_df[best_basefold_df["comb_idx"] == _comb_idx].reset_index(drop=True)
    _tf_raw = str(_comb_pred["test_fold_indices"].iloc[0])
    _split_path = os.path.join(save_dir, f"split_comb{_comb_idx:02d}_{_tf_raw}.xlsx")

    _meta = pd.DataFrame(
        {
            "key": [
                "target_col",
                "comb_idx",
                "test_fold_indices",
                "n_train_rows",
                "n_test_rows",
                "train_ratio",
                "test_ratio",
            ],
            "value": [
                target_col,
                int(_comb_idx),
                _tf_raw,
                int(_comb_sum["n_train_rows"].iloc[0]) if len(_comb_sum) else None,
                int(_comb_sum["n_test_rows"].iloc[0]) if len(_comb_sum) else None,
                float(_comb_sum["train_ratio"].iloc[0]) if len(_comb_sum) else None,
                float(_comb_sum["test_ratio"].iloc[0]) if len(_comb_sum) else None,
            ],
        }
    )
    _feat_cols_df = pd.DataFrame({"feature_col": list(feature_cols)})

    with pd.ExcelWriter(_split_path, engine="openpyxl") as _sw:
        _comb_sum.to_excel(_sw, sheet_name="comb_summary_metrics", index=False)
        _comb_bf.to_excel(_sw, sheet_name="basefold_metrics", index=False)
        _comb_pred.to_excel(_sw, sheet_name="test_rows_full", index=False)
        _meta.to_excel(_sw, sheet_name="meta", index=False)
        _feat_cols_df.to_excel(_sw, sheet_name="feature_columns", index=False)

    print(f"split file: {_split_path}")

print(f"Ensemble MAE (mean over the two folds): {mean_ensemble_mae:.4f}")
print(f"Averaged prediction MAE: {avg_mae:.4f}")
print(f"Averaged prediction MSE: {avg_mse:.4f}")

checkpoint_root = os.path.join(save_dir, "checkpoint")
os.makedirs(checkpoint_root, exist_ok=True)

for comb_model in all_comb_models:
    comb_idx = int(comb_model["comb_idx"])
    comb_dir = os.path.join(checkpoint_root, f"comb_{comb_idx:02d}")
    os.makedirs(comb_dir, exist_ok=True)

    joblib.dump(comb_model["scaler"], os.path.join(comb_dir, "scaler.pkl"))
    joblib.dump(comb_model["ngb_model"], os.path.join(comb_dir, "ngb_model.pkl"))
    torch.save(comb_model["mlp_state_dict"], os.path.join(comb_dir, "mlp_model.pt"))
    joblib.dump(comb_model["mlp_params"], os.path.join(comb_dir, "mlp_params.pkl"))

    joblib.dump(
        {
            "comb_idx": comb_idx,
            "test_fold_indices": comb_model.get("test_fold_indices"),
            "NGB_WEIGHT": NGB_WEIGHT,
            "MLP_WEIGHT": MLP_WEIGHT,
            "NGB_VAR_WEIGHT": NGB_VAR_WEIGHT,
            "MLP_VAR_WEIGHT": MLP_VAR_WEIGHT,
            "feature_cols": list(feature_cols),
            "target_col": target_col,
            "input_dim": input_dim,
            "ngb_params": copy.deepcopy(ngb_params),
        },
        os.path.join(comb_dir, "inference_meta.pkl"),
    )

    print(f"checkpoint saved: {comb_dir}")

COMB_IDX = int(comb_summary_df.loc[comb_summary_df["ensemble_mae"].idxmin(), "comb_idx"])
checkpoint_dir = os.path.join(save_dir, "checkpoint", f"comb_{COMB_IDX:02d}")

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
