import argparse
import time

import torch
from torch.utils.data import DataLoader

from src.config import AUDetectionConfig
from src.datasets.dataset import AUDataset
from src.datasets.painfacereader import load_units
from src.datasets.utils import (
    assert_no_leakage, cv_splits, filter_aus, load_folds, make_folds, subset_units,
)
from src.models import build_model, input_transform
from src import training


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="yaml config; omit for defaults")
    ap.add_argument("--fold_path", default="assets/folds/folds_frontal_k5_seed0.json")
    ap.add_argument("--steps", type=int, default=25, help="short run just to prove it trains")
    args = ap.parse_args()

    cfg = AUDetectionConfig.from_yaml(args.config) if args.config else AUDetectionConfig()
    print(f"[smoke] task={cfg.task} model={cfg.model} device={cfg.device} img_size={cfg.img_size}")

    # ---- data ----
    units, au_names = load_units(cfg.painfacereader)
    print(f"[smoke] loaded {len(units)} units, {len(au_names)} AUs")

    try:
        folds, kept, _ = load_folds(args.fold_path)
        print(f"[smoke] loaded folds from {args.fold_path}")
    except (FileNotFoundError, TypeError):
        print(f"[smoke] no fold file at {args.fold_path!r}; computing a quick split")
        kept = cfg.aus or filter_aus(units, au_names, min_prevalence=cfg.min_prevalence,
                                     min_subjects=cfg.min_subjects, min_carrier_frames=cfg.min_carrier_frames)
        folds = make_folds(units, au_names, kept, k=cfg.n_folds, n_iter=cfg.folds_iter,
                           n_restarts=cfg.folds_n_restart, seed=0)
    assert_no_leakage(folds)
    units = subset_units(units, au_names, kept)
    print(f"[smoke] kept {len(kept)} AUs: {kept}")

    # ---- fold 0 datasets ----
    sp = next(cv_splits(folds))
    pick = lambda names: [u for u in units if u.subject in set(names)]
    tr, va, te = pick(sp["train"]), pick(sp["val"]), pick(sp["test"])

    def ds(us, train):
        return AUDataset(us, cfg.painfacereader.crops_dir, kept,
                         unit_type=cfg.unit_type, window_len=cfg.window_len,
                         transform=input_transform(cfg, train=train))
    train_ds, val_ds, test_ds = ds(tr, True), ds(va, False), ds(te, False)
    print(f"[smoke] fold0 units: train={len(tr)} val={len(va)} test={len(te)} | "
          f"items: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    # ---- one-batch sanity ----
    xb, yb = next(iter(DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)))
    print(f"[smoke] batch x={tuple(xb.shape)} {xb.dtype} range=[{xb.min():.2f},{xb.max():.2f}]  y={tuple(yb.shape)}")
    assert xb.shape[1:] == (3, cfg.img_size, cfg.img_size)
    assert yb.shape[1] == len(kept), "label/AU count mismatch"

    # ---- model + short train + eval ----
    model = build_model(cfg, n_classes=len(kept))
    pw = training.pos_weight(train_ds)
    print(f"[smoke] {type(model).__name__} ({model.backbone_name}) "
          f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M params | "
          f"pos_weight[:3]={[round(v,1) for v in pw[:3].tolist()]}")

    t = time.time()
    m, history = training.run_fold(model, train_ds, val_ds, test_ds, kept,
                                   steps=args.steps, batch_size=cfg.batch_size, lr=cfg.lr,
                                   device=cfg.device, eval_every=args.steps,
                                   sampler=training.neutral_active_sampler(train_ds),
                                   num_workers=cfg.n_workers)
    print(f"[smoke] {args.steps} steps + eval in {time.time()-t:.1f}s")
    print(f"[smoke] macro-F1={m.macro_f1:.3f} (rough — only {args.steps} steps)")
    print("[smoke] OK: pipeline runs end-to-end on real data")


if __name__ == "__main__":
    main()