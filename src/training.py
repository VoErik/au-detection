from __future__ import annotations

"""
Training harness. A model is any nn.Module mapping a batched input to (B, K) logits.
Multi-label -> BCEWithLogitsLoss with a per-AU pos_weight.

fit() trains with a cosine LR schedule, evaluates val macro-F1 every eval_every
steps, and keeps the BEST-on-val weights (so the step count is a ceiling, not a
guess, and you never report an over-trained final model). run_fold restores the
best weights, tunes thresholds on val, and scores test. Both return the val
history so the training curve can be plotted.
"""

import copy
import time
from typing import Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler

from src.metrics import AUMetrics, evaluate, find_thresholds, score

# DataLoader workers share tensors over the queue; the default file-descriptor
# strategy can exhaust `ulimit -n` on long runs. file_system avoids that.
torch.multiprocessing.set_sharing_strategy("file_system")


def pos_weight(dataset, *, clip_max: float = 50.0) -> torch.Tensor:
    """Per-AU BCE pos_weight = N_neg / N_pos over the dataset's units, clipped to
    [1, clip_max]. Compute on the TRAIN dataset only."""
    k = dataset.n_classes
    pos = np.zeros(k, dtype=np.float64)
    total = 0
    for u in dataset.units:
        pos += u.labels.sum(axis=0)
        total += u.labels.shape[0]
    neg = total - pos
    w = np.where(pos > 0, neg / np.maximum(pos, 1.0), 1.0)
    return torch.tensor(np.clip(w, 1.0, clip_max), dtype=torch.float32)


def neutral_active_sampler(dataset, *, active_weight: Optional[float] = None) -> WeightedRandomSampler:
    """Balance all-zero (neutral) vs any-AU-active frames by sampling. Balances by
    FRAME CONTENT, not subject. active_weight defaults to (#neutral / #active)."""
    active = np.empty(len(dataset._index), dtype=bool)
    for i, (ui, pos) in enumerate(dataset._index):
        active[i] = dataset.units[ui].labels[pos].any()
    n_active = int(active.sum())
    if active_weight is None:
        active_weight = (len(active) - n_active) / max(n_active, 1)
    w = np.where(active, active_weight, 1.0)
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                 num_samples=len(w), replacement=True)


class BlockShuffleSampler(Sampler):
    """Yield indices in blocks of `block` CONSECUTIVE positions, block order
    shuffled each epoch, so reads within a block are SEQUENTIAL (fast) instead of
    random seeks. A batch of size B mixes ~B/block videos. Keep block small (8-16);
    rely on pos_weight for imbalance. Use INSTEAD of neutral_active_sampler when
    I/O-bound (and the crops don't fit in page cache)."""

    def __init__(self, n: int, block: int = 8, seed: int = 0):
        self.n = n
        self.block = max(1, block)
        self.seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __len__(self) -> int:
        return self.n

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1
        n_blocks = (self.n + self.block - 1) // self.block
        out = np.empty(self.n, dtype=np.int64)
        pos = 0
        for b in rng.permutation(n_blocks):
            s = b * self.block
            e = min(s + self.block, self.n)
            out[pos:pos + (e - s)] = np.arange(s, e)
            pos += e - s
        return iter(out.tolist())


def _cpu_state(model) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


@torch.no_grad()
def infer(model, dataset, *, batch_size: int, device: str, num_workers: int = 0,
          desc: str = "eval", verbose: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Run model over dataset (no shuffle) -> (logits (N,K), targets (N,K))."""
    from tqdm.auto import tqdm
    use_amp = device.startswith("cuda")
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=use_amp)
    scores, targets = [], []
    for batch in tqdm(loader, desc=desc, unit="batch", disable=not verbose, leave=False):
        x = batch[0].to(device, non_blocking=True)
        if use_amp:
            with autocast("cuda", dtype=torch.float16):
                logits = model(x)
        else:
            logits = model(x)
        scores.append(logits.float().cpu().numpy())
        targets.append(batch[1].numpy())
    return np.concatenate(scores), np.concatenate(targets)


def _val_macro_f1(model, val_dataset, au_names, *, batch_size, device, num_workers) -> float:
    vs, vt = infer(model=model, dataset=val_dataset, batch_size=batch_size, device=device,
                   num_workers=num_workers, desc="val", verbose=False)
    thr = find_thresholds(val_scores=vs, val_targets=vt)
    return score(scores=vs, targets=vt, au_names=au_names, thresholds=thr).macro_f1


def fit(model, train_dataset, val_dataset, au_names: Sequence[str], *,
        steps: int, batch_size: int, lr: float, pos_weight: torch.Tensor, device: str,
        eval_every: int = 1500, sampler: Optional[Sampler] = None,
        weight_decay: float = 1e-4, grad_clip: Optional[float] = 1.0,
        num_workers: int = 0, lr_min_factor: float = 0.01, verbose: bool = True):
    """Cosine-LR training with periodic val macro-F1 + best-on-val checkpointing.
    Returns (best_state_dict, history); history entries are
    {step, train_loss, val_macro_f1, lr}."""
    from tqdm.auto import tqdm
    use_amp = device.startswith("cuda")
    torch.backends.cudnn.benchmark = True
    model.train()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * lr_min_factor)
    scaler = GradScaler("cuda", enabled=use_amp)
    loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                        shuffle=(sampler is None), num_workers=num_workers, drop_last=True,
                        pin_memory=use_amp, persistent_workers=(num_workers > 0),
                        prefetch_factor=(4 if num_workers > 0 else None))

    it = iter(loader)
    ema = None
    history: list[dict] = []
    best_f1, best_state = -1.0, _cpu_state(model)
    bar = tqdm(range(1, steps + 1), desc="fit", unit="step", disable=not verbose)
    for step in bar:
        t = time.time()
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        data_ms = (time.time() - t) * 1e3
        x = batch[0].to(device, non_blocking=True)
        y = batch[1].to(device, non_blocking=True)
        opt.zero_grad()
        if use_amp:
            with autocast("cuda", dtype=torch.float16):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
        else:
            loss = loss_fn(model(x), y)
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
        sched.step()
        v = loss.item()
        ema = v if ema is None else 0.98 * ema + 0.02 * v
        cur_lr = opt.param_groups[0]["lr"]
        bar.set_postfix(loss=f"{ema:.3f}", lr=f"{cur_lr:.1e}", data=f"{data_ms:.0f}ms")

        if step % eval_every == 0 or step == steps:
            vf1 = _val_macro_f1(model=model, val_dataset=val_dataset, au_names=au_names,
                                batch_size=batch_size, device=device, num_workers=num_workers)
            history.append({"step": step, "train_loss": float(ema),
                            "val_macro_f1": float(vf1), "lr": float(cur_lr)})
            if vf1 > best_f1:
                best_f1, best_state = vf1, _cpu_state(model)
            model.train()
            bar.set_postfix(loss=f"{ema:.3f}", val_f1=f"{vf1:.3f}", best=f"{best_f1:.3f}")
    return best_state, history


def run_fold(model, train_dataset, val_dataset, test_dataset, au_names: Sequence[str], *,
             steps: int, batch_size: int, lr: float, device: str, eval_every: int = 1500,
             pw: Optional[torch.Tensor] = None, sampler: Optional[Sampler] = None,
             num_workers: int = 0, verbose: bool = True):
    """One fold: fit (cosine LR + best-on-val) -> restore best weights -> tune
    thresholds on val -> score test. Returns (AUMetrics, history). Fresh model per fold."""
    model = model.to(device)
    if pw is None:
        pw = pos_weight(train_dataset)
    best_state, history = fit(model=model, train_dataset=train_dataset, val_dataset=val_dataset,
                              au_names=au_names, steps=steps, batch_size=batch_size, lr=lr,
                              pos_weight=pw, device=device, eval_every=eval_every, sampler=sampler,
                              num_workers=num_workers, verbose=verbose)
    model.load_state_dict(best_state)
    vs, vt = infer(model=model, dataset=val_dataset, batch_size=batch_size, device=device,
                   num_workers=num_workers, desc="eval val", verbose=verbose)
    ts, tt = infer(model=model, dataset=test_dataset, batch_size=batch_size, device=device,
                   num_workers=num_workers, desc="eval test", verbose=verbose)
    return evaluate(val_scores=vs, val_targets=vt, test_scores=ts, test_targets=tt,
                    au_names=au_names, from_logits=True), history


def _build_sampler(cfg, dataset):
    if cfg.sampler == "neutral_active":
        return neutral_active_sampler(dataset=dataset)
    if cfg.sampler == "block":
        return BlockShuffleSampler(n=len(dataset), block=cfg.block_size)
    if cfg.sampler == "none":
        return None
    raise ValueError(f"unknown sampler {cfg.sampler!r}")


def run_cv(cfg, units, folds, kept, crops_dir, *, seed: int = 0, exp=None, tag: str = "",
           sampler_fn=None, num_workers: Optional[int] = None, verbose: bool = True):
    """Run all k folds for ONE training seed with a FIXED fold assignment.

    `seed` drives model init + data order (via set_seed); the split (`folds`) is
    held fixed so this isolates training/seed variance. Builds a fresh model per
    fold, trains with run_fold, and -- if `exp` is given -- saves each fold's
    model / curve / history under `{tag}fold{i}`. Returns (list[AUMetrics],
    list[history]). Pool several seeds' metric lists and hand the flat list to
    metrics.aggregate_folds for a seeds x folds table."""
    import json
    from src.utils import set_seed
    from src.datasets.dataset import AUDataset
    from src.datasets.utils import cv_splits
    from src.models import build_model, input_transform
    from src import viz

    set_seed(seed)
    nw = cfg.n_workers if num_workers is None else num_workers
    fold_metrics, histories = [], []
    for sp in cv_splits(folds):
        i = sp["fold"]
        tr = [u for u in units if u.subject in set(sp["train"])]
        va = [u for u in units if u.subject in set(sp["val"])]
        te = [u for u in units if u.subject in set(sp["test"])]

        def ds(us, train):
            return AUDataset(units=us, crops_dir=crops_dir, au_names=kept, unit_type=cfg.unit_type,
                             window_len=cfg.window_len, transform=input_transform(config=cfg, train=train))
        train_ds, val_ds, test_ds = ds(us=tr, train=True), ds(us=va, train=False), ds(us=te, train=False)

        pw = (pos_weight(dataset=train_ds, clip_max=cfg.pos_weight_clip)
              if cfg.pos_weight_clip and cfg.pos_weight_clip > 1 else torch.ones(len(kept)))
        sampler = sampler_fn(train_ds) if sampler_fn is not None else _build_sampler(cfg=cfg, dataset=train_ds)

        model = build_model(config=cfg, n_classes=len(kept))   # fresh per fold
        m, history = run_fold(model=model, train_dataset=train_ds, val_dataset=val_ds,
                              test_dataset=test_ds, au_names=kept, steps=cfg.n_steps,
                              batch_size=cfg.batch_size, lr=cfg.lr, device=cfg.device,
                              eval_every=cfg.eval_every, pw=pw, sampler=sampler,
                              num_workers=nw, verbose=verbose)
        fold_metrics.append(m)
        histories.append(history)
        if exp is not None:
            pre = f"{tag}fold{i}"
            exp.save_model(model=model, name=f"{pre}.pt")
            exp.path(f"history_{pre}.json").write_text(json.dumps(history, indent=2))
            exp.save_figure(fig=viz.show_training_curve(history=history), name=f"curve_{pre}.png")
            m.to_frame().to_csv(exp.path(f"metrics_{pre}.csv"))   # per-AU table, written per fold
            exp.log.info(f"[{tag}fold {i}] macro-F1 {m.macro_f1:.3f}  "
                         f"({m.n_undefined} AU(s) undefined)")
    return fold_metrics, histories