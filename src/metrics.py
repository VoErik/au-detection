from __future__ import annotations

"""
Evaluation protocol for multi-label AU detection:
  find_thresholds  -- per-AU probability threshold maximizing F1 on validation
  score            -- per-AU F1/precision/recall + macro-F1 at given thresholds
  evaluate         -- the fold protocol: tune on val, apply to test
  aggregate_folds  -- per-AU F1 mean/std across folds + macro-F1 mean/std

AUs with zero positives in the targets get NaN F1 and are dropped from the macro
average (a fold's test set may simply not contain a rare AU).
"""

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _as_probs(scores: np.ndarray, from_logits: bool) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    return _sigmoid(scores) if from_logits else scores


def _f1(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return (2 * tp) / denom if denom > 0 else np.nan


def find_thresholds(val_scores: np.ndarray, val_targets: np.ndarray, *,
                    from_logits: bool = True, n_steps: int = 200) -> np.ndarray:
    """Per-AU threshold that maximizes F1 on the validation set (grid over (0,1);
    ties -> lower threshold, which favours recall on rare AUs). AUs with no val
    positives get 0.5."""
    probs = _as_probs(val_scores, from_logits)
    y = np.asarray(val_targets).astype(int)
    n, k = probs.shape
    grid = np.linspace(0.0, 1.0, n_steps + 1)[1:-1]
    thresholds = np.full(k, 0.5, dtype=np.float64)
    for j in range(k):
        if y[:, j].sum() == 0:
            continue
        pred = probs[:, j][:, None] >= grid[None, :]           # (n, G)
        yj = y[:, j][:, None] == 1
        tp = (pred & yj).sum(0).astype(np.float64)
        fp = (pred & ~yj).sum(0).astype(np.float64)
        fn = (~pred & yj).sum(0).astype(np.float64)
        denom = 2 * tp + fp + fn
        f1 = np.divide(2 * tp, denom, out=np.zeros_like(denom), where=denom > 0)
        thresholds[j] = grid[int(np.argmax(f1))]
    return thresholds


@dataclass
class AUMetrics:
    per_au_f1: dict[str, float]           # NaN where the AU has no positives
    macro_f1: float                       # mean over AUs with defined F1
    per_au_precision: dict[str, float]
    per_au_recall: dict[str, float]
    per_au_support: dict[str, int]        # positive count in the targets
    thresholds: dict[str, float]
    n_undefined: int = field(default=0)   # AUs excluded from macro (no positives)

    def to_frame(self):
        import pandas as pd
        return pd.DataFrame({
            "f1": self.per_au_f1,
            "precision": self.per_au_precision,
            "recall": self.per_au_recall,
            "support": self.per_au_support,
            "threshold": self.thresholds,
        })


def score(scores: np.ndarray, targets: np.ndarray, au_names: Sequence[str], *,
          thresholds: Optional[np.ndarray] = None, from_logits: bool = True) -> AUMetrics:
    """Per-AU F1/precision/recall and macro-F1 at the given per-AU thresholds
    (default 0.5). AUs with zero positives get NaN F1 and drop from the macro."""
    probs = _as_probs(scores, from_logits)
    y = np.asarray(targets).astype(int)
    k = y.shape[1]
    if thresholds is None:
        thresholds = np.full(k, 0.5, dtype=np.float64)

    f1, prec, rec, support = {}, {}, {}, {}
    for j, au in enumerate(au_names):
        pred = probs[:, j] >= thresholds[j]
        tp = int((pred & (y[:, j] == 1)).sum())
        fp = int((pred & (y[:, j] == 0)).sum())
        fn = int((~pred & (y[:, j] == 1)).sum())
        support[au] = int((y[:, j] == 1).sum())
        f1[au] = _f1(tp, fp, fn) if support[au] > 0 else np.nan
        prec[au] = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        rec[au] = tp / (tp + fn) if (tp + fn) > 0 else np.nan

    defined = [v for v in f1.values() if not np.isnan(v)]
    macro = float(np.mean(defined)) if defined else np.nan
    return AUMetrics(
        per_au_f1=f1, macro_f1=macro,
        per_au_precision=prec, per_au_recall=rec, per_au_support=support,
        thresholds={au: float(thresholds[j]) for j, au in enumerate(au_names)},
        n_undefined=k - len(defined),
    )


def evaluate(val_scores: np.ndarray, val_targets: np.ndarray,
             test_scores: np.ndarray, test_targets: np.ndarray,
             au_names: Sequence[str], *, from_logits: bool = True) -> AUMetrics:
    """The fold protocol: fit thresholds on val, apply to test, return test metrics."""
    thr = find_thresholds(val_scores, val_targets, from_logits=from_logits)
    return score(test_scores, test_targets, au_names, thresholds=thr, from_logits=from_logits)


def aggregate_folds(fold_metrics: Sequence[AUMetrics]):
    """Combine per-fold AUMetrics -> (per-AU DataFrame with f1 mean/std, macro
    summary dict). NaNs (undefined AUs in a fold) are ignored per AU."""
    import pandas as pd
    aus = list(fold_metrics[0].per_au_f1.keys())
    rows = {}
    for au in aus:
        vals = np.array([m.per_au_f1[au] for m in fold_metrics], dtype=float)
        defined = vals[~np.isnan(vals)]
        rows[au] = {
            "f1_mean": float(defined.mean()) if len(defined) else np.nan,
            "f1_std": float(defined.std()) if len(defined) else np.nan,
            "n_folds_defined": int(len(defined)),
            "support_total": int(sum(m.per_au_support[au] for m in fold_metrics)),
        }
    df = pd.DataFrame(rows).T
    macros = np.array([m.macro_f1 for m in fold_metrics], dtype=float)
    macros = macros[~np.isnan(macros)]
    summary = {
        "macro_f1_mean": float(macros.mean()) if len(macros) else np.nan,
        "macro_f1_std": float(macros.std()) if len(macros) else np.nan,
    }
    return df.sort_values("f1_mean"), summary


# --------------------------------------------------------------------------- #
# LaTeX tables from a run's per-fold metrics                                  #
# --------------------------------------------------------------------------- #

def load_run_metrics(run_dir):
    """Read a run's per-fold metrics_*.csv -> {tag: per-AU DataFrame}, ordered.
    Each DataFrame is AUMetrics.to_frame() (index = AU; f1/precision/recall/
    support/threshold). tag is e.g. 'seed0_fold0'."""
    import pandas as pd
    from pathlib import Path
    frames = {}
    for p in sorted(Path(run_dir).glob("metrics_*.csv")):
        frames[p.stem[len("metrics_"):]] = pd.read_csv(p, index_col=0)
    if not frames:
        raise FileNotFoundError(f"no metrics_*.csv found in {run_dir}")
    return frames


def _tex_escape(s) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def _pm(mean: float, std: float) -> str:
    return f"{mean:.3f} $\\pm$ {std:.3f}"


def fold_summary_latex(frames, *, caption: str = "Per-fold macro-F1.",
                       label: str = "tab:folds") -> str:
    """LaTeX table: macro-F1 for each fold (rows) + a pooled mean $\\pm$ std row.
    `frames` is {tag: per-AU DataFrame} (from load_run_metrics or
    {tag: m.to_frame()}). Per-fold macro = mean over AUs with defined F1."""
    import numpy as np
    tags = list(frames)
    macros = np.array([frames[t]["f1"].mean() for t in tags], dtype=float)  # nan-skipping
    lines = [r"\begin{table}[t]", r"\centering", r"\begin{tabular}{lc}", r"\toprule",
             r"Fold & Macro-F1 \\", r"\midrule"]
    for t, mf in zip(tags, macros):
        lines.append(f"{_tex_escape(t)} & {mf:.3f} \\\\")
    lines += [r"\midrule", f"Mean $\\pm$ Std & {_pm(macros.mean(), macros.std())} \\\\",
              r"\bottomrule", r"\end{tabular}",
              f"\\caption{{{caption}}}", f"\\label{{{label}}}", r"\end{table}"]
    return "\n".join(lines)


def per_au_latex(frames, *, caption: str = "Per-AU F1, precision, recall "
                 "(mean $\\pm$ std over folds) with total support.",
                 label: str = "tab:per_au") -> str:
    """LaTeX table: per-AU F1/precision/recall (mean $\\pm$ std over folds) +
    total positive support, plus a macro row. NaNs (undefined AU in a fold) are
    skipped per cell."""
    import numpy as np
    dfs = list(frames.values())
    aus = list(dfs[0].index)

    def agg(au, col):
        v = np.array([df.loc[au, col] for df in dfs], dtype=float)
        v = v[~np.isnan(v)]
        return _pm(float(v.mean()), float(v.std())) if len(v) else "--"

    lines = [r"\begin{table}[t]", r"\centering", r"\begin{tabular}{lcccr}", r"\toprule",
             r"AU & F1 & Precision & Recall & Support \\", r"\midrule"]
    for au in aus:
        sup = int(np.nansum([df.loc[au, "support"] for df in dfs]))
        lines.append(f"{_tex_escape(au)} & {agg(au,'f1')} & {agg(au,'precision')} "
                     f"& {agg(au,'recall')} & {sup} \\\\")
    macros = np.array([df["f1"].mean() for df in dfs], dtype=float)
    lines += [r"\midrule",
              f"Macro & {_pm(macros.mean(), macros.std())} & & & \\\\",
              r"\bottomrule", r"\end{tabular}",
              f"\\caption{{{caption}}}", f"\\label{{{label}}}", r"\end{table}"]
    return "\n".join(lines)