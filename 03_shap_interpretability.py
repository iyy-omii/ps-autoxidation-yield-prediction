import joblib
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import shap
from pathlib import Path

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

try:
    from matplotlib_inline.backend_inline import set_matplotlib_formats

    set_matplotlib_formats("retina")
except Exception:
    pass


def _setup_inter_font() -> None:
    font_dirs = [Path.home() / "Library" / "Fonts", Path("/Library/Fonts")]
    for font_dir in font_dirs:
        if not font_dir.exists():
            continue
        for path in font_dir.glob("Inter*.ttf"):
            fm.fontManager.addfont(str(path))


_setup_inter_font()
_inter_path = Path.home() / "Library" / "Fonts" / "Inter-Regular.ttf"
print("[font] Inter available:", _inter_path.exists(), "->", _inter_path)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "DejaVu Sans", "Arial"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Inter",
        "mathtext.it": "Inter:italic",
        "mathtext.bf": "Inter:bold",
        "font.size": 17,
        "axes.labelsize": 18,
        "axes.titlesize": 19,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "text.color": "black",
        "axes.labelcolor": "black",
        "axes.edgecolor": "black",
        "axes.titlecolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.format": "png",
        "savefig.dpi": 600,
    }
)


def _enforce_black_plot_style(fig, color: str = "black") -> None:
    for ax in fig.axes:
        ax.tick_params(axis="both", colors=color, labelcolor=color)
        ax.xaxis.label.set_color(color)
        ax.yaxis.label.set_color(color)
        ax.title.set_color(color)
        ax.xaxis.label.set_fontfamily("Inter")
        ax.yaxis.label.set_fontfamily("Inter")
        ax.title.set_fontfamily("Inter")
        for spine in ax.spines.values():
            spine.set_color(color)
        for text in ax.get_xticklabels() + ax.get_yticklabels():
            text.set_color(color)
            if "$" not in text.get_text():
                text.set_fontfamily("Inter")
    for text in fig.texts:
        text.set_color(color)
        if "$" not in text.get_text():
            text.set_fontfamily("Inter")
    for legend in fig.legends:
        for text in legend.get_texts():
            text.set_color(color)
            if "$" not in text.get_text():
                text.set_fontfamily("Inter")


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
    def __init__(self, m: HeteroMLPRegressor):
        super().__init__()
        self.inner = m

    def forward(self, x):
        mu, _ = self.inner(x)
        return mu


def _list_combs(best_dir: Path) -> list[tuple[int, Path, float]]:
    """All split_combXX_*.xlsx entries under best_dir, each with its comb_idx and ensemble_mae.
    Both folds are processed -- SHAP results are reported for the lower- and higher-error fold
    alike (see Section 3.3)."""
    combs = []
    for p in sorted(best_dir.glob("split_comb*_*.xlsx")):
        if p.name.startswith("~$"):
            continue
        m = pd.read_excel(p, sheet_name="comb_summary_metrics")
        mae = float(m["ensemble_mae"].iloc[0])
        comb_idx = int(m["comb_idx"].iloc[0])
        combs.append((comb_idx, p, mae))
    return sorted(combs, key=lambda t: t[0])


def _drop_ps(
    feature_cols: list,
    shap_vals: np.ndarray,
    x_matrix: np.ndarray,
    ps_name: str = "PS amount (g)",
):
    keep_idx = [i for i, c in enumerate(feature_cols) if c != ps_name]
    names = [feature_cols[i] for i in keep_idx]
    return (
        shap_vals[:, keep_idx],
        x_matrix[:, keep_idx],
        names,
        keep_idx,
    )


def _fix_mn_subscript(text: str) -> str:
    return text.replace("Mn(OAc)₂", "Mn(OAc)$_2$").replace("Mn(OAc)2", "Mn(OAc)$_2$")


def _strip_feature_units(name: str) -> str:
    name = _fix_mn_subscript(name)
    unit_patterns = (
        r"\s*\(wt%\)\s*$",
        r"\s*\(h\)\s*$",
        r"\s*\(g\)\s*$",
        r"\s*\(degree C\)\s*$",
        r"\s*\(degree c\)\s*$",
        r"\s*\(°C\)\s*$",
        r"\s*\(\$\\^\\circ\$C\)\s*$",
    )
    for pattern in unit_patterns:
        name = re.sub(pattern, "", name, flags=re.IGNORECASE)
    return name.strip()


def _format_dependence_feature_name(name: str, *, lowercase: bool = False) -> str:
    name = _strip_feature_units(name)
    key = _norm_feat(name)
    lower_map = {
        "reaction time": "reaction time",
        "temperature": "temperature",
        "benzoic acid": "benzoic acid",
        "benzoic acid amount": "benzoic acid",
    }
    if lowercase and key in lower_map:
        return lower_map[key]
    return name


def _dependence_ylabel_for_feature(feature_label: str) -> str:
    key = _norm_feat(_strip_feature_units(feature_label))
    use_lower = key in {"reaction time", "temperature", "benzoic acid", "benzoic acid amount"}
    feat = _format_dependence_feature_name(feature_label, lowercase=use_lower)
    return f"SHAP value for {feat}"


def _format_dependence_xlabel(label: str) -> str:
    if _norm_feat(label) in {"benzoic acid (g)", "benzoic acid", "benzoic acid amount (g)"}:
        return "Benzoic acid (g)"
    return label


def _split_feature_unit(name: str) -> tuple[str, str]:
    name = _fix_mn_subscript(name.strip())
    match = re.search(r"(\s*\([^)]+\))\s*$", name)
    if match:
        return name[: match.start()].strip(), match.group(1)
    return name, ""


def _format_interaction_label(label: str) -> str:
    label = _fix_mn_subscript(label)
    base, unit = _split_feature_unit(label)
    key = _norm_feat(base)
    lower_map = {
        "reaction time": "reaction time",
        "temperature": "temperature",
        "benzoic acid": "benzoic acid",
        "benzoic acid amount": "benzoic acid",
    }
    if key in lower_map:
        base = lower_map[key]
    return base + unit


def _norm_feat(s: str) -> str:
    s = s.replace("$", "").replace("_", "").replace("{", "").replace("}", "")
    return s.lower().replace("₂", "2").replace("temperaure", "temperature").strip()

def _pretty_feature_names(names: list[str]) -> list[str]:
    pretty = {
        "temperature (degree c)": "Temperature (°C)",
        "benzoic acid amount (g)": "Benzoic acid (g)",
    }
    out = []
    for name in names:
        n = _fix_mn_subscript(name)
        out.append(pretty.get(n.lower().strip(), n))
    return out


def _drop_ps_interaction(inter: np.ndarray, keep_idx: list[int]) -> np.ndarray:
    inter = np.asarray(inter)
    return inter[:, keep_idx, :][:, :, keep_idx]


here = Path.cwd().resolve()
best_dir = here / "reported_run"
if not best_dir.exists():
    raise FileNotFoundError(
        f"'reported_run' not found under {here}. Run 01_ensemble_training_pipeline.py first."
    )
final_dir = here

plot_output_dir_root = final_dir / "shap_outputs"
plot_output_dir_root.mkdir(parents=True, exist_ok=True)


PLOT_SAVE_FORMAT = "png"
PLOT_SAVE_DPI = 600


def _save_plot(fig, stem: str, *, bbox_inches="tight") -> Path:
    _enforce_black_plot_style(fig)
    path = plot_output_dir / f"{stem}.{PLOT_SAVE_FORMAT}"
    fig.savefig(
        path,
        format=PLOT_SAVE_FORMAT,
        dpi=PLOT_SAVE_DPI,
        bbox_inches=bbox_inches,
        facecolor="white",
    )
    print(f"[saved] {path}")
    return path


def _show_and_save_plot(stem: str, *, bbox_inches="tight") -> None:
    _save_plot(plt.gcf(), stem, bbox_inches=bbox_inches)
    plt.show()

all_combs = _list_combs(best_dir)
if not all_combs:
    raise FileNotFoundError(f"No split_comb*_*.xlsx files found under {best_dir}")

for comb_idx, split_xlsx, comb_mae in all_combs:
    plot_output_dir = plot_output_dir_root / f"comb_{comb_idx:02d}"
    plot_output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print(f"comb {comb_idx}  (ensemble_mae={comb_mae:.4f})")
    print("=" * 70)

    checkpoint_dir = best_dir / "checkpoint" / f"comb_{comb_idx:02d}"

    _data_candidates = [
        here / "Polystyrene_autoxidation.xlsx",
        here.parent / "Polystyrene_autoxidation.xlsx",
    ]
    data_path = next((p for p in _data_candidates if p.exists()), None)
    if data_path is None:
        raise FileNotFoundError(
            "Polystyrene_autoxidation.xlsx not found. "
            "Place a copy of the dataset in the same directory as this script."
        )

    print("comb:", comb_idx, "ensemble_mae=", comb_mae)
    print("split:", split_xlsx.name)
    print("checkpoint:", checkpoint_dir)

    df = pd.read_excel(data_path, sheet_name=0, header=0).dropna().reset_index(drop=True)
    target_col = "Yield (%)"

    test_rows = pd.read_excel(split_xlsx, sheet_name="test_rows_full")
    test_ids = test_rows["sample_id"].astype(int).values
    test_mask = df.index.isin(test_ids)
    train_mask = ~test_mask

    scaler = joblib.load(checkpoint_dir / "scaler.pkl")
    mlp_params = joblib.load(checkpoint_dir / "mlp_params.pkl")
    meta = joblib.load(checkpoint_dir / "inference_meta.pkl")
    feature_cols = list(meta["feature_cols"])
    input_dim = int(meta["input_dim"])
    NGB_WEIGHT = meta["NGB_WEIGHT"]
    MLP_WEIGHT = meta["MLP_WEIGHT"]

    X_train = df.loc[train_mask, feature_cols].reset_index(drop=True)
    X_test = df.loc[test_ids, feature_cols]

    X_train_scaled = scaler.transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    X_test_raw = X_test.to_numpy(dtype=np.float32)
    print("X_test_raw[0,:3]:", X_test_raw[0, :3])
    print("X_test_scaled[0,:3]:", X_test_scaled[0, :3])

    ngb_model = joblib.load(checkpoint_dir / "ngb_model.pkl")

    mlp_model = HeteroMLPRegressor(
        input_dim=input_dim,
        hidden1=mlp_params["hidden1"],
        hidden2=mlp_params["hidden2"],
        dropout=mlp_params["dropout"],
    )
    _sd_path = checkpoint_dir / "mlp_model.pt"
    try:
        _state = torch.load(_sd_path, map_location="cpu", weights_only=True)
    except TypeError:
        _state = torch.load(_sd_path, map_location="cpu")
    mlp_model.load_state_dict(_state)
    mlp_model.eval()

    try:
        ngb_exp = shap.TreeExplainer(ngb_model)
        shap_values_ngb = ngb_exp.shap_values(X_test_scaled)
    except Exception:
        ngb_exp = shap.Explainer(ngb_model.predict, X_train_scaled)
        shap_values_ngb = ngb_exp(X_test_scaled).values

    shap_values_ngb = np.asarray(shap_values_ngb)
    sv_ngb, xv_ngb, feat_no_ps, keep_idx = _drop_ps(feature_cols, shap_values_ngb, X_test_raw)
    feat_no_ps = _pretty_feature_names(feat_no_ps)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

    mlp_mu_only = HeteroMLPMuOnly(mlp_model)
    mlp_exp = shap.GradientExplainer(mlp_mu_only, X_train_tensor)
    shap_values_mlp = mlp_exp.shap_values(X_test_tensor)

    if isinstance(shap_values_mlp, list):
        shap_values_mlp = shap_values_mlp[0]
    shap_values_mlp = np.array(shap_values_mlp)
    if shap_values_mlp.ndim == 3 and shap_values_mlp.shape[-1] == 1:
        shap_values_mlp = shap_values_mlp.squeeze(-1)

    print("final shap shape:", shap_values_mlp.shape)
    print("X_test_scaled shape:", X_test_scaled.shape)

    sv_mlp, xv_mlp, _, _ = _drop_ps(feature_cols, shap_values_mlp, X_test_raw)

    shap_ensemble = NGB_WEIGHT * shap_values_ngb + MLP_WEIGHT * shap_values_mlp
    sv_e, xv_e, _, _ = _drop_ps(feature_cols, shap_ensemble, X_test_raw)

    def _style_shap_summary_plot(
        xlabel: str = "SHAP value",
        *,
        ticksize: int = 11,
        axis_labelsize: int = 13,
        xlabel_pad: int = 12,
        fig_width: float = 11.2,
        fig_height: float = 5.2,
    ) -> None:
        fig = plt.gcf()
        fig.set_size_inches(fig_width, fig_height)
        main_ax = max(fig.axes, key=lambda a: a.get_position().width * a.get_position().height)

        for ax in fig.axes:
            ax.tick_params(axis="both", labelsize=ticksize, pad=3, colors="black", labelcolor="black")
            xlab = ax.get_xlabel() or ""
            if "SHAP" in xlab:
                ax.set_xlabel(xlabel, fontsize=axis_labelsize, labelpad=xlabel_pad, fontfamily="Inter")
            ylab = ax.get_ylabel() or ""
            if "Feature" in ylab:
                ax.set_ylabel(ylab, fontsize=axis_labelsize, fontfamily="Inter")
            for text in ax.get_yticklabels() + ax.get_xticklabels():
                raw = _fix_mn_subscript(text.get_text())
                text.set_text(raw)
                text.set_fontsize(ticksize)
                text.set_color("black")
                if "$" not in raw:
                    text.set_fontfamily("Inter")
                text.set_clip_on(False)

        main_ax.set_xlabel(xlabel, fontsize=axis_labelsize, labelpad=xlabel_pad, fontfamily="Inter")
        _enforce_black_plot_style(fig)
        fig.subplots_adjust(left=0.34, right=0.88, bottom=0.20, top=0.96, wspace=0.45)


    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Inter", "DejaVu Sans", "Arial"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Inter",
            "mathtext.it": "Inter:italic",
            "mathtext.bf": "Inter:bold",
            "font.size": 11,
            "axes.labelsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    ):
        shap.summary_plot(sv_e, xv_e, feature_names=feat_no_ps, plot_type="bar", show=False)
        _style_shap_summary_plot("SHAP value")
    _show_and_save_plot("ensemble_no_ps_summary_bar")
    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Inter", "DejaVu Sans", "Arial"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Inter",
            "mathtext.it": "Inter:italic",
            "mathtext.bf": "Inter:bold",
            "font.size": 11,
            "axes.labelsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    ):
        shap.summary_plot(sv_e, xv_e, feature_names=feat_no_ps, show=False)
        _style_shap_summary_plot("SHAP value")
    _show_and_save_plot("ensemble_no_ps_summary_beeswarm")

    mean_abs_shap = np.abs(sv_e).mean(axis=0)
    shap_importance = (
        pd.DataFrame({"feature": feat_no_ps, "mean_abs_shap": mean_abs_shap})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    print("\n[Ensemble SHAP] mean(|SHAP|) values")
    print(shap_importance.to_string(index=False))

    ordered_labels = [
        "NaBr (wt%)",
        "Mn(OAc)$_2$ (wt%)",
        "Reaction time (h)",
        "Temperature (°C)",
        "Benzoic acid (g)",
    ]

    feat_lookup = {_norm_feat(f): f for f in feat_no_ps}
    ordered_pairs = []
    for label in ordered_labels:
        key = _norm_feat(label)
        if key in feat_lookup:
            ordered_pairs.append((feat_lookup[key], label))

    if len(ordered_pairs) != len(ordered_labels):
        missing = [lbl for lbl in ordered_labels if _norm_feat(lbl) not in feat_lookup]
        raise ValueError(f"feature(s) not found: {missing} / available: {feat_no_ps}")

    n_feat = len(ordered_pairs)
    ncols = 3
    nrows = int(np.ceil(n_feat / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 4.1 * nrows))
    axes = np.array(axes).reshape(-1)
    main_axes = axes[:n_feat]
    cbar_axes: list = []
    cbar_labels: list[str] = []


    def _pick_colorbar_axis(fig, main_ax, n_axes_before: int):
        new_axes = [a for a in fig.axes[n_axes_before:] if a is not main_ax]
        if not new_axes:
            return None
        return min(new_axes, key=lambda a: a.get_position().width)


    def _layout_dependence_panels(
        fig,
        main_axes,
        cbar_axes,
        ncols: int,
        nrows: int,
        *,
        left=0.06,
        right=0.96,
        bottom=0.12,
        top=0.93,
        gap_x=0.080,
        gap_y=0.15,
        cbar_w=0.010,
        cbar_pad=0.008,
        cbar_tick_w=0.020,
        name_gap=0.010,
        cbar_label_w=0.022,
    ) -> list[float]:
        total_w = right - left
        total_h = top - bottom
        side_w = cbar_pad + cbar_w + cbar_tick_w + name_gap + cbar_label_w

        plot_w = (total_w - (ncols - 1) * gap_x - ncols * side_w) / ncols
        plot_h = (total_h - (nrows - 1) * gap_y) / nrows
        block_w = plot_w + side_w + gap_x
        label_xs: list[float] = []

        for i, ax in enumerate(main_axes):
            row, col = divmod(i, ncols)
            x = left + col * block_w
            y = bottom + (nrows - 1 - row) * (plot_h + gap_y)
            ax.set_position([x, y, plot_w, plot_h])

            if i < len(cbar_axes) and cbar_axes[i] is not None:
                cbar_axes[i].set_position([x + plot_w + cbar_pad, y, cbar_w, plot_h])

            label_xs.append(x + plot_w + cbar_pad + cbar_w + cbar_tick_w + name_gap)

        return label_xs


    def _place_colorbar_labels(
        fig,
        cbar_axes,
        cbar_labels,
        label_xs: list[float],
    ) -> None:
        for cb_ax, label, label_x in zip(cbar_axes, cbar_labels, label_xs):
            if cb_ax is None or not label:
                continue
            cb_ax.set_ylabel("")
            cb_ax.tick_params(axis="y", labelsize=11, pad=0)
            pos = cb_ax.get_position()
            label_x = max(label_x, pos.x1 + 0.008)
            fig.text(
                label_x,
                pos.y0 + pos.height * 0.5,
                _format_interaction_label(label),
                rotation=90,
                va="center",
                ha="left",
                fontsize=12,
                color="black",
            )

    for i, (fname_internal, display_label) in enumerate(ordered_pairs):
        ax = main_axes[i]
        n_axes_before = len(fig.axes)
        shap.dependence_plot(
            fname_internal,
            sv_e,
            xv_e,
            feature_names=feat_no_ps,
            ax=ax,
            show=False,
        )
        cb_ax = _pick_colorbar_axis(fig, ax, n_axes_before)
        cb_label = _format_interaction_label(cb_ax.get_ylabel()) if cb_ax is not None else ""
        cbar_axes.append(cb_ax)
        cbar_labels.append(cb_label)
        if cb_ax is not None:
            cb_ax.tick_params(labelsize=11)
        panel = f"({chr(ord('a') + i)})"
        ax.set_title(
            panel,
            loc="left",
            x=-0.22,
            ha="left",
            pad=8,
            y=1.03,
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel(_format_dependence_xlabel(display_label), fontsize=14, labelpad=8)
        ax.set_ylabel(_dependence_ylabel_for_feature(display_label), fontsize=12)
        ax.tick_params(axis="both", labelsize=11)
        ax.tick_params(axis="x", pad=4)

    for j in range(n_feat, len(axes)):
        axes[j].axis("off")

    label_xs = _layout_dependence_panels(fig, main_axes, cbar_axes, ncols, nrows)
    _place_colorbar_labels(fig, cbar_axes, cbar_labels, label_xs)
    _enforce_black_plot_style(fig)
    _show_and_save_plot("dependence_no_ps_panels", bbox_inches=None)

    try:
        shap_interaction = shap.TreeExplainer(ngb_model).shap_interaction_values(X_test_scaled)
        shap_interaction = np.asarray(shap_interaction)
        shap_interaction = _drop_ps_interaction(shap_interaction, keep_idx)
        shap.summary_plot(shap_interaction, xv_e, feature_names=feat_no_ps, show=False)
        _enforce_black_plot_style(plt.gcf())
        _show_and_save_plot("ngb_interaction_summary_no_ps")
    except Exception as _e:
        print("shap interaction (NGB):", _e)
