import argparse

from src.config import AUDetectionConfig
from src.datasets.utils import (
    assert_no_leakage, filter_aus, load_folds, make_folds, subset_units,
)
from src.metrics import aggregate_folds, fold_summary_latex, per_au_latex
from src.utils import Experiment
from src import training, viz


def load_task_units(cfg):
    """Dispatch to the right dataset loader for cfg.task."""
    if cfg.task == "painfacereader":
        from src.datasets.painfacereader import load_units
    elif cfg.task == "disfa":
        from src.datasets.disfa import load_units
    elif cfg.task == "bp4d":
        from src.datasets.bp4d import load_units
    else:
        raise ValueError(f"unknown task: {cfg.task}")
    return load_units(config=cfg.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="assets/configs/painfacereader_mae.yaml", help="yaml config; omit for defaults")
    parser.add_argument("--fold_path", type=str, default="assets/folds/folds_frontal_k5_seed0.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0],
                        help="training seeds; one seed = a single k-fold run, several = a seed sweep")
    parser.add_argument("--name", type=str, default=None, help="experiment name")
    parser.add_argument("--show", action="store_true", help="show the F1 figure")
    args = parser.parse_args()

    cfg = AUDetectionConfig.from_yaml(path=args.config) if args.config else AUDetectionConfig()
    exp = Experiment(config=cfg, name=args.name)   # run dir, copies config.yaml, opens run.log
    log = exp.log

    log.info("Loading units.")
    units, au_names = load_task_units(cfg=cfg)

    if args.fold_path:
        log.info(f"Loading folds from {args.fold_path}")
        folds, kept, _ = load_folds(path=args.fold_path)
    else:
        if cfg.aus:
            log.info(f"Using specified AUs: {cfg.aus}")
            kept = cfg.aus
        else:
            log.info("No AUs specified; selecting by prevalence/carriers.")
            kept = filter_aus(units=units, au_names=au_names, min_prevalence=cfg.min_prevalence,
                              min_subjects=cfg.min_subjects, min_carrier_frames=cfg.min_carrier_frames)
        log.info(f"Computing folds for {len(kept)} AUs.")
        folds = make_folds(units=units, au_names=au_names, kept_aus=kept, k=cfg.n_folds,
                           n_iter=cfg.folds_iter, n_restarts=cfg.folds_n_restart,
                           t0=cfg.folds_t0, t_min=cfg.folds_tmin,
                           min_carrier_frames=cfg.min_carrier_frames, seed=args.seeds[0])
    assert_no_leakage(folds=folds)

    units = subset_units(units=units, au_names=au_names, kept_aus=kept)
    log.info(f"Prepared {len(units)} units, {len(kept)} AUs: {kept}")

    all_metrics = []
    run_frames = {}
    for seed in args.seeds:
        log.info(f"=== seed {seed} ===")
        fold_metrics, _ = training.run_cv(
            cfg=cfg, units=units, folds=folds, kept=kept, crops_dir=cfg.dataset.crops_dir,
            seed=seed, exp=exp, tag=f"seed{seed}_", verbose=True)
        all_metrics += fold_metrics
        for i, m in enumerate(fold_metrics):
            run_frames[f"seed{seed}_fold{i}"] = m.to_frame()
        _, seed_summary = aggregate_folds(fold_metrics=fold_metrics)
        log.info(f"seed {seed}: macro-F1 "
                 f"{seed_summary['macro_f1_mean']:.3f} ± {seed_summary['macro_f1_std']:.3f} (over {cfg.n_folds} folds)")

    log.info(f"Aggregating {len(all_metrics)} evaluations "
             f"({len(args.seeds)} seed(s) x {cfg.n_folds} folds).")
    df, summary = aggregate_folds(fold_metrics=all_metrics)
    exp.save_metrics(df=df, summary=summary)
    exp.save_figure(fig=viz.show_au_f1(agg_df=df), name="au_f1.png")
    exp.path("table_folds.tex").write_text(fold_summary_latex(frames=run_frames))
    exp.path("table_per_au.tex").write_text(per_au_latex(frames=run_frames))
    log.info(f"macro-F1 {summary['macro_f1_mean']:.3f} ± {summary['macro_f1_std']:.3f}")
    log.info(f"artifacts in {exp.dir}")

    if args.show:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()