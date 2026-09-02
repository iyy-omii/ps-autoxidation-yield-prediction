import pandas as pd
import numpy as np

pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 160)

FILE_FOLD0 = "./reported_run/split_comb01_fold_A.xlsx"
FILE_FOLD1 = "./reported_run/split_comb02_fold_B.xlsx"
SHEET_NAME = "test_rows_full"

CONDITION_COLS = [
    "PS amount (g)",
    "benzoic acid amount (g)",
    "Mn(OAc)2 (wt%)",
    "NaBr (wt%)",
    "Reaction time (h)",
    "temperature (degree C)",
]

df_fold0 = pd.read_excel(FILE_FOLD0, sheet_name=SHEET_NAME)
df_fold1 = pd.read_excel(FILE_FOLD1, sheet_name=SHEET_NAME)

print(f"[fold0] {FILE_FOLD0} -> {df_fold0.shape[0]} rows x {df_fold0.shape[1]} cols")
print(f"[fold1] {FILE_FOLD1} -> {df_fold1.shape[0]} rows x {df_fold1.shape[1]} cols")
print(list(df_fold0.columns))

assert list(df_fold0.columns) == list(df_fold1.columns)

df_all = pd.concat([df_fold0, df_fold1], ignore_index=True)
print(f"n_test_rows_total: {len(df_all)}")

ids_fold0 = set(df_fold0["sample_id"])
ids_fold1 = set(df_fold1["sample_id"])
overlap = ids_fold0 & ids_fold1

print(f"n_unique_fold0: {len(ids_fold0)}")
print(f"n_unique_fold1: {len(ids_fold1)}")
print(f"n_overlap: {len(overlap)}")
print(f"n_total: {len(ids_fold0) + len(ids_fold1)}")

assert len(overlap) == 0
assert len(ids_fold0) + len(ids_fold1) == 137

df_all["_cond_key"] = list(
    zip(*[df_all[c].round(4) for c in CONDITION_COLS])
)

group_sizes = df_all.groupby("_cond_key").size()
print(group_sizes.value_counts().sort_index())

triplicate_keys = group_sizes[group_sizes == 3].index.tolist()
n_triplicate = len(triplicate_keys)

print(f"n_triplicate_conditions: {n_triplicate}")
for k in triplicate_keys:
    print(" ", k)

TABLE2_ORDER = [
    (0.45, 1.00, 14.0, 4.0, 2, 165),
    (0.45, 1.00, 11.0, 4.0, 2, 165),
    (0.45, 1.00,  7.1, 6.2, 2, 165),
    (0.45, 1.00,  7.1, 4.0, 2, 165),
    (0.45, 1.00,  7.1, 2.0, 2, 180),
    (0.45, 1.00,  7.1, 2.0, 2, 165),
    (0.45, 1.00,  7.1, 1.0, 2, 165),
    (0.45, 1.00,  7.1, 0.5, 2, 165),
    (0.45, 1.00,  7.1, 0.0, 2, 165),
    (0.45, 1.00,  3.6, 4.0, 2, 165),
    (0.45, 1.00,  1.8, 4.0, 2, 165),
    (0.45, 0.70,  7.1, 2.0, 2, 165),
    (0.45, 0.35,  7.1, 2.0, 2, 165),
]

def find_matching_key(target, keys, tol=1e-3):
    for k in keys:
        if all(abs(k[i] - target[i]) < tol for i in range(len(target))):
            return k
    raise ValueError(f"no matching condition: {target}")

ordered_keys = [find_matching_key(t, triplicate_keys) for t in TABLE2_ORDER]
print(f"matched_all_13: {len(ordered_keys) == 13 == n_triplicate}")

rows_summary = []

for key in ordered_keys:
    sub = df_all[df_all["_cond_key"] == key].sort_values("Yield (%)")

    for col in ["pred_ngb", "pred_mlp", "pred_ensemble", "pred_var_ensemble"]:
        n_unique = sub[col].round(6).nunique()
        assert n_unique == 1, f"{key} {col} mismatch across rows"

    yields = sub["Yield (%)"].tolist()
    pred_ngb = sub["pred_ngb"].iloc[0]
    pred_mlp = sub["pred_mlp"].iloc[0]
    pred_ensemble = sub["pred_ensemble"].iloc[0]
    var_old = sub["pred_var_ensemble"].iloc[0]

    rows_summary.append({
        "condition": key,
        "yield_1": yields[0], "yield_2": yields[1], "yield_3": yields[2],
        "observed_mean": np.mean(yields),
        "pred_ngb": pred_ngb,
        "pred_mlp": pred_mlp,
        "pred_ensemble": pred_ensemble,
        "var_data_old": var_old,
    })

summary = pd.DataFrame(rows_summary)
print(summary)

summary["sigma_old"] = np.sqrt(summary["var_data_old"])
summary["ci_old_lower"] = summary["pred_ensemble"] - 1.96 * summary["sigma_old"]
summary["ci_old_upper"] = summary["pred_ensemble"] + 1.96 * summary["sigma_old"]

def count_coverage(row, lower_col, upper_col):
    ys = [row["yield_1"], row["yield_2"], row["yield_3"]]
    return sum(1 for y in ys if row[lower_col] <= y <= row[upper_col])

summary["coverage_old"] = summary.apply(count_coverage, axis=1, lower_col="ci_old_lower", upper_col="ci_old_upper")

display_cols = ["condition", "yield_1", "yield_2", "yield_3", "pred_ensemble",
                 "var_data_old", "sigma_old", "ci_old_lower", "ci_old_upper", "coverage_old"]
print(summary[display_cols].round(3).to_string(index=False))

total_old_covered = summary["coverage_old"].sum()
total_n = len(summary) * 3
picp_old = total_old_covered / total_n
print(f"picp_old: {total_old_covered}/{total_n} = {picp_old:.4%}")

def model_disagreement_var_methodA(mu_ngb, mu_mlp):
    return np.var([mu_ngb, mu_mlp], ddof=0)

def model_disagreement_var_methodB(mu_ngb, mu_mlp):
    values = np.array([mu_ngb, mu_mlp])
    return np.mean(values**2) - np.mean(values)**2

def model_disagreement_var_shortcut(mu_ngb, mu_mlp):
    return ((mu_ngb - mu_mlp) ** 2) / 4

check_rows = []
for _, row in summary.iterrows():
    a = model_disagreement_var_methodA(row["pred_ngb"], row["pred_mlp"])
    b = model_disagreement_var_methodB(row["pred_ngb"], row["pred_mlp"])
    c = model_disagreement_var_shortcut(row["pred_ngb"], row["pred_mlp"])
    check_rows.append({"condition": row["condition"], "method_A": a, "method_B": b, "method_C": c})

check_df = pd.DataFrame(check_rows)
print(check_df.round(6).to_string(index=False))

all_match = np.allclose(check_df["method_A"], check_df["method_B"]) and \
            np.allclose(check_df["method_A"], check_df["method_C"])
print(f"methods_match: {all_match}")

wrong_example = summary.iloc[0]
wrong_var_ddof1 = np.var([wrong_example["pred_ngb"], wrong_example["pred_mlp"]], ddof=1)
correct_var_ddof0 = np.var([wrong_example["pred_ngb"], wrong_example["pred_mlp"]], ddof=0)

print(f"pred_ngb={wrong_example['pred_ngb']:.4f} pred_mlp={wrong_example['pred_mlp']:.4f}")
print(f"var_ddof1={wrong_var_ddof1:.4f} var_ddof0={correct_var_ddof0:.4f}")
print(f"ratio={wrong_var_ddof1 / correct_var_ddof0:.2f}")

summary["var_model_disagreement"] = summary.apply(
    lambda r: model_disagreement_var_shortcut(r["pred_ngb"], r["pred_mlp"]), axis=1
)
print(summary[["condition", "pred_ngb", "pred_mlp", "var_model_disagreement"]].round(4).to_string(index=False))

summary["var_total_new"] = summary["var_data_old"] + summary["var_model_disagreement"]
summary["sigma_new"] = np.sqrt(summary["var_total_new"])
summary["ci_new_lower"] = summary["pred_ensemble"] - 1.96 * summary["sigma_new"]
summary["ci_new_upper"] = summary["pred_ensemble"] + 1.96 * summary["sigma_new"]
summary["coverage_new"] = summary.apply(count_coverage, axis=1, lower_col="ci_new_lower", upper_col="ci_new_upper")

display_cols_new = ["condition", "yield_1", "yield_2", "yield_3", "pred_ensemble",
                     "var_data_old", "var_model_disagreement", "var_total_new",
                     "sigma_new", "ci_new_lower", "ci_new_upper", "coverage_new"]
print(summary[display_cols_new].round(3).to_string(index=False))

compare = summary[["condition", "coverage_old", "coverage_new"]].copy()
compare["ci_old_width"] = summary["ci_old_upper"] - summary["ci_old_lower"]
compare["ci_new_width"] = summary["ci_new_upper"] - summary["ci_new_lower"]
compare["width_increase"] = compare["ci_new_width"] - compare["ci_old_width"]
compare["coverage_changed"] = compare["coverage_old"] != compare["coverage_new"]
print(compare.round(2).to_string(index=False))

n_improved = (compare["coverage_new"] > compare["coverage_old"]).sum()
n_same = (compare["coverage_new"] == compare["coverage_old"]).sum()
n_worse = (compare["coverage_new"] < compare["coverage_old"]).sum()
print(f"improved={n_improved} same={n_same} worse={n_worse} (of 13)")

total_old_covered = summary["coverage_old"].sum()
total_new_covered = summary["coverage_new"].sum()
total_n = len(summary) * 3

picp_old = total_old_covered / total_n
picp_new = total_new_covered / total_n

print(f"n_obs={total_n}")
print(f"picp_old={total_old_covered}/{total_n}={picp_old:.2%}")
print(f"picp_new={total_new_covered}/{total_n}={picp_new:.2%}")
print(f"picp_delta=+{(picp_new - picp_old):.2%}p")

mpiw_old = (summary["ci_old_upper"] - summary["ci_old_lower"]).mean()
mpiw_new = (summary["ci_new_upper"] - summary["ci_new_lower"]).mean()

print(f"mpiw_old={mpiw_old:.3f}")
print(f"mpiw_new={mpiw_new:.3f}")
print(f"ratio={mpiw_new / mpiw_old:.2f}")

final_table = summary.drop(columns=["condition"]).copy()
cond_df = pd.DataFrame(summary["condition"].tolist(), columns=CONDITION_COLS)
final_table = pd.concat([cond_df, final_table], axis=1)

output_name = "table2_model_disagreement_variance_verified.xlsx"
final_table.to_excel(output_name, index=False)
print(f"written: {output_name}")
print(final_table.round(3))
