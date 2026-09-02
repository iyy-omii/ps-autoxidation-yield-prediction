import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 160)

DATA_PATH = "Polystyrene_autoxidation.xlsx"
FILE_FOLD0 = "reported_run/split_comb01_fold_A.xlsx"
FILE_FOLD1 = "reported_run/split_comb02_fold_B.xlsx"

df = pd.read_excel(DATA_PATH).dropna().reset_index(drop=True)
FEATURE_COLS = ["PS amount (g)", "benzoic acid amount (g)", "Mn(OAc)2 (wt%)", "NaBr (wt%)",
                "Reaction time (h)", "temperature (degree C)"]
TARGET_COL = "Yield (%)"

f0 = pd.read_excel(FILE_FOLD0, sheet_name="test_rows_full")
f1 = pd.read_excel(FILE_FOLD1, sheet_name="test_rows_full")

fold0_test_idx = f0["sample_id"].values
fold1_test_idx = f1["sample_id"].values

assert len(set(fold0_test_idx) & set(fold1_test_idx)) == 0
assert len(fold0_test_idx) + len(fold1_test_idx) == len(df) == 137
print(f"fold0_test={len(fold0_test_idx)} fold1_test={len(fold1_test_idx)} total={len(df)}")

X_all = df[FEATURE_COLS].values
y_all = df[TARGET_COL].values

SPLITS = [
    ("comb1_test_fold0", fold1_test_idx, fold0_test_idx),
    ("comb2_test_fold1", fold0_test_idx, fold1_test_idx),
]


def picp_mpiw(y, mu, var, z=1.96):
    sigma = np.sqrt(np.maximum(var, 1e-8))
    lower, upper = mu - z * sigma, mu + z * sigma
    covered = (y >= lower) & (y <= upper)
    return covered.mean(), (upper - lower).mean()


def gaussian_nll(y, mu, var):
    var = np.maximum(var, 1e-8)
    return np.mean(0.5 * np.log(2 * np.pi * var) + (y - mu) ** 2 / (2 * var))


def gaussian_crps(y, mu, var):
    sigma = np.sqrt(np.maximum(var, 1e-8))
    z = (y - mu) / sigma
    return np.mean(sigma * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi)))


class HeteroscedasticMLP(nn.Module):
    def __init__(self, in_dim, hidden=32, dropout_p=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.drop1 = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(hidden, hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.mu_head = nn.Linear(hidden, 1)
        self.logvar_head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = F.relu(self.bn1(self.fc1(x)))
        h = self.drop1(h)
        h = F.relu(self.bn2(self.fc2(h)))
        mu = self.mu_head(h).squeeze(-1)
        logvar = torch.clamp(self.logvar_head(h).squeeze(-1), -5, 5)
        return mu, logvar

    def backbone_params(self):
        return list(self.fc1.parameters()) + list(self.bn1.parameters()) + \
               list(self.fc2.parameters()) + list(self.bn2.parameters()) + \
               list(self.mu_head.parameters())


def train_heteroscedastic_mlp(Xtr, ytr, in_dim, seed=0):
    torch.manual_seed(seed)
    model = HeteroscedasticMLP(in_dim)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)

    for p in model.logvar_head.parameters():
        p.requires_grad_(False)
    opt1 = torch.optim.Adam(model.backbone_params(), lr=5e-2, weight_decay=1e-4)
    model.train()
    for epoch in range(80):
        opt1.zero_grad()
        mu, _ = model(Xtr_t)
        loss = F.mse_loss(mu, ytr_t)
        loss.backward()
        opt1.step()

    for p in model.backbone_params():
        p.requires_grad_(False)
    for p in model.logvar_head.parameters():
        p.requires_grad_(True)
    opt2 = torch.optim.Adam(model.logvar_head.parameters(), lr=5e-4, weight_decay=1e-4)
    model.eval()
    for epoch in range(20):
        opt2.zero_grad()
        mu, logvar = model(Xtr_t)
        var = torch.exp(logvar)
        nll = 0.5 * (logvar + (ytr_t - mu) ** 2 / var)
        loss = nll.mean()
        loss.backward()
        opt2.step()
    return model


def predict_heteroscedastic_mlp(model, X):
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32)
        mu, logvar = model(X_t)
        return mu.numpy(), np.exp(logvar.numpy())


class MCDropoutMLP(nn.Module):
    def __init__(self, in_dim, hidden=32, dropout_p=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.drop1 = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(hidden, hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.drop2 = nn.Dropout(dropout_p)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        h = F.relu(self.bn1(self.fc1(x)))
        h = self.drop1(h)
        h = F.relu(self.bn2(self.fc2(h)))
        h = self.drop2(h)
        return self.out(h).squeeze(-1)


def train_mc_dropout(Xtr, ytr, in_dim, seed=0, epochs=300):
    torch.manual_seed(seed)
    model = MCDropoutMLP(in_dim)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
    model.train()
    for epoch in range(epochs):
        opt.zero_grad()
        pred = model(Xtr_t)
        loss = F.mse_loss(pred, ytr_t)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        train_pred = model(Xtr_t).numpy()
    residual_var = np.mean((ytr - train_pred) ** 2)
    return model, residual_var


def predict_mc_dropout(model, X, residual_var, T=100):
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
    X_t = torch.tensor(X, dtype=torch.float32)
    preds = []
    with torch.no_grad():
        for _ in range(T):
            preds.append(model(X_t).numpy())
    preds = np.stack(preds, axis=0)
    mu = preds.mean(axis=0)
    epistemic_var = preds.var(axis=0)
    total_var = epistemic_var + residual_var
    return mu, total_var


N_DEEP_ENSEMBLE_MEMBERS = 5


def train_deep_ensemble(Xtr, ytr, in_dim, base_seed=0):
    return [train_heteroscedastic_mlp(Xtr, ytr, in_dim, seed=base_seed * 1000 + m)
            for m in range(N_DEEP_ENSEMBLE_MEMBERS)]


def predict_deep_ensemble(members, X):
    mus, vars_ = [], []
    for model in members:
        mu, var = predict_heteroscedastic_mlp(model, X)
        mus.append(mu)
        vars_.append(var)
    mus = np.stack(mus, axis=0)
    vars_ = np.stack(vars_, axis=0)
    mu_star = mus.mean(axis=0)
    var_star = (vars_ + mus ** 2).mean(axis=0) - mu_star ** 2
    return mu_star, var_star


RIDGE_ALPHA = 1.0
LASSO_ALPHA = 0.01

MODEL_NAMES = ["Linear", "Ridge", "Lasso", "RandomForest", "GPR", "MC-Dropout BNN", "Deep Ensemble"]
records = []

for split_name, train_idx, test_idx in SPLITS:
    Xtr, Xte = X_all[train_idx], X_all[test_idx]
    ytr, yte = y_all[train_idx], y_all[test_idx]
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
    in_dim = Xtr_s.shape[1]

    preds = {}

    linreg = LinearRegression()
    linreg.fit(Xtr_s, ytr)
    preds["Linear"] = (linreg.predict(Xte_s), None)

    ridge = Ridge(alpha=RIDGE_ALPHA, random_state=SEED)
    ridge.fit(Xtr_s, ytr)
    preds["Ridge"] = (ridge.predict(Xte_s), None)

    lasso = Lasso(alpha=LASSO_ALPHA, random_state=SEED, max_iter=10000)
    lasso.fit(Xtr_s, ytr)
    preds["Lasso"] = (lasso.predict(Xte_s), None)

    rf = RandomForestRegressor(n_estimators=300, random_state=SEED)
    rf.fit(Xtr_s, ytr)
    preds["RandomForest"] = (rf.predict(Xte_s), None)

    kernel = ConstantKernel(1.0) * RBF(length_scale=np.ones(in_dim)) + WhiteKernel(1.0)
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=3,
                                    random_state=SEED)
    gpr.fit(Xtr_s, ytr)
    mu_gpr, std_gpr = gpr.predict(Xte_s, return_std=True)
    preds["GPR"] = (mu_gpr, std_gpr ** 2)

    mc_model, resid_var = train_mc_dropout(Xtr_s, ytr, in_dim, seed=SEED)
    mu_mc, var_mc = predict_mc_dropout(mc_model, Xte_s, resid_var, T=100)
    preds["MC-Dropout BNN"] = (mu_mc, var_mc)

    members = train_deep_ensemble(Xtr_s, ytr, in_dim, base_seed=SEED)
    mu_de, var_de = predict_deep_ensemble(members, Xte_s)
    preds["Deep Ensemble"] = (mu_de, var_de)

    for name, (mu, var) in preds.items():
        row = {
            "split": split_name, "model": name, "n_test": len(test_idx),
            "mae": mean_absolute_error(yte, mu),
            "rmse": np.sqrt(mean_squared_error(yte, mu)),
        }
        if var is not None:
            picp, mpiw = picp_mpiw(yte, mu, var)
            row.update({"picp": picp, "mpiw": mpiw,
                        "nll": gaussian_nll(yte, mu, var),
                        "crps": gaussian_crps(yte, mu, var)})
        records.append(row)

    print(f"{split_name} done train={len(train_idx)} test={len(test_idx)}")

results_df = pd.DataFrame(records)

final_table = results_df.groupby("model").agg(
    mae_mean=("mae", "mean"), mae_std=("mae", "std"),
    rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
    picp_mean=("picp", "mean"), mpiw_mean=("mpiw", "mean"),
    nll_mean=("nll", "mean"), crps_mean=("crps", "mean"),
).reindex(MODEL_NAMES)

print(final_table.round(3).to_string())

final_table.round(3).to_excel("baseline_comparison_summary.xlsx")
results_df.to_excel("baseline_comparison_fold_results.xlsx", index=False)
