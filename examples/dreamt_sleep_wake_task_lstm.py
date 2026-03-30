"""DREAMT Sleep/Wake Detection with LSTM — Ablation Study

This example demonstrates:
  1. Signal-subset ablation (ACC-only, BVP/HRV-only, EDA+TEMP-only, all)
  2. Label-granularity ablation (binary wake/sleep vs. 5-class staging)

using a single-layer unidirectional LSTM trained on per-epoch features
extracted by SleepWakeTask / SleepStageTask from the DREAMT dataset.

Evaluation uses 5-fold participant-level cross-validation (no subject
leakage) and reports F1, AUROC, accuracy, and Cohen's Kappa.

Usage:
    python dreamt_sleep_wake_task_lstm.py --root /path/to/dreamt/2.1.0

Reference:
    Wang et al., CHIL 2024, PMLR 248:380-396.
"""

import argparse
import warnings
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

from pyhealth.datasets import DREAMTDataset
from pyhealth.tasks import SleepStageTask, SleepWakeTask

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# LSTM model
# ---------------------------------------------------------------------------

class SleepLSTM(nn.Module):
    """Shallow single-layer unidirectional LSTM for sequence classification."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_classes: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, features)
        out, _ = self.lstm(x)
        logits = self.fc(out)  # (batch, seq_len, num_classes)
        return logits


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

class SequenceDataset(Dataset):
    """Groups epoch samples by patient into full-night sequences."""

    def __init__(self, samples):
        patient_map = defaultdict(list)
        for s in samples:
            patient_map[s["patient_id"]].append(s)

        self.sequences = []
        self.labels_list = []
        for pid in sorted(patient_map):
            epochs = sorted(patient_map[pid], key=lambda e: e["epoch_idx"])
            signals = np.stack([e["signal"] for e in epochs], axis=0)
            labels = np.array([e["label"] for e in epochs])
            self.sequences.append(torch.tensor(signals, dtype=torch.float32))
            self.labels_list.append(torch.tensor(labels, dtype=torch.long))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels_list[idx]


def collate_fn(batch):
    """Pad variable-length sequences in a batch."""
    seqs, labels = zip(*batch)
    max_len = max(s.shape[0] for s in seqs)
    feat_dim = seqs[0].shape[1]
    padded_seqs = torch.zeros(len(seqs), max_len, feat_dim)
    padded_labels = torch.full((len(seqs), max_len), -1, dtype=torch.long)
    masks = torch.zeros(len(seqs), max_len, dtype=torch.bool)
    for i, (s, l) in enumerate(zip(seqs, labels)):
        length = s.shape[0]
        padded_seqs[i, :length] = s
        padded_labels[i, :length] = l
        masks[i, :length] = True
    return padded_seqs, padded_labels, masks


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate(
    train_samples,
    test_samples,
    num_classes=2,
    epochs=30,
    lr=1e-3,
    hidden_dim=64,
    device="cpu",
):
    """Train LSTM on train_samples, evaluate on test_samples."""
    if not train_samples or not test_samples:
        return {}

    feat_dim = train_samples[0]["signal"].shape[0]

    train_ds = SequenceDataset(train_samples)
    test_ds = SequenceDataset(test_samples)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=collate_fn)

    model = SleepLSTM(feat_dim, hidden_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    model.train()
    for _ in range(epochs):
        for seqs, labels, masks in train_loader:
            seqs, labels = seqs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(seqs)
            logits_flat = logits.reshape(-1, num_classes)
            labels_flat = labels.reshape(-1)
            loss = criterion(logits_flat, labels_flat)
            loss.backward()
            optimizer.step()

    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for seqs, labels, masks in test_loader:
            seqs = seqs.to(device)
            logits = model(seqs)
            probs = torch.softmax(logits, dim=-1).cpu()
            preds = logits.argmax(dim=-1).cpu()
            for i in range(seqs.shape[0]):
                valid = masks[i]
                all_preds.extend(preds[i][valid].numpy().tolist())
                all_labels.extend(labels[i][valid].numpy().tolist())
                if num_classes == 2:
                    all_probs.extend(probs[i][valid][:, 1].numpy().tolist())
                else:
                    all_probs.extend(probs[i][valid].numpy().tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    avg = "binary" if num_classes == 2 else "macro"
    results = {
        "f1": f1_score(y_true, y_pred, average=avg, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "kappa": cohen_kappa_score(y_true, y_pred),
    }

    try:
        if num_classes == 2:
            results["auroc"] = roc_auc_score(y_true, np.array(all_probs))
        else:
            results["auroc"] = roc_auc_score(
                y_true, np.array(all_probs), multi_class="ovr", average="macro"
            )
    except ValueError:
        results["auroc"] = float("nan")

    return results


def participant_cv(samples, n_folds=5, num_classes=2, **kwargs):
    """5-fold participant-level cross-validation."""
    patient_ids = sorted(set(s["patient_id"] for s in samples))
    np.random.seed(42)
    np.random.shuffle(patient_ids)

    fold_size = len(patient_ids) // n_folds
    fold_results = []

    for fold in range(n_folds):
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < n_folds - 1 else len(patient_ids)
        test_ids = set(patient_ids[test_start:test_end])
        train_ids = set(patient_ids) - test_ids

        train_samples = [s for s in samples if s["patient_id"] in train_ids]
        test_samples = [s for s in samples if s["patient_id"] in test_ids]

        res = train_and_evaluate(
            train_samples, test_samples, num_classes=num_classes, **kwargs
        )
        if res:
            fold_results.append(res)
            print(
                f"  Fold {fold + 1}: "
                f"F1={res['f1']:.3f}  AUROC={res['auroc']:.3f}  "
                f"Acc={res['accuracy']:.3f}  Kappa={res['kappa']:.3f}"
            )

    if not fold_results:
        return {}

    avg = {}
    for key in fold_results[0]:
        vals = [r[key] for r in fold_results]
        avg[key] = f"{np.mean(vals):.3f} +/- {np.std(vals):.3f}"
    return avg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DREAMT LSTM ablation study")
    parser.add_argument("--root", required=True, help="Path to DREAMT dataset version folder")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs per fold")
    parser.add_argument("--hidden_dim", type=int, default=64, help="LSTM hidden dimension")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    print("Loading DREAMT dataset...")
    dataset = DREAMTDataset(root=args.root)

    # -----------------------------------------------------------------------
    # Ablation 1: Signal subset
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ABLATION 1: Signal Subset (binary wake/sleep)")
    print("=" * 70)

    for subset in ["ACC", "BVP_HRV", "EDA_TEMP", "ALL"]:
        print(f"\n--- Signal subset: {subset} ---")
        task = SleepWakeTask(signal_subset=subset)
        sample_ds = dataset.set_task(task)
        samples = [sample_ds[i] for i in range(len(sample_ds))]
        print(f"  Total samples: {len(samples)}")

        avg = participant_cv(
            samples,
            num_classes=2,
            epochs=args.epochs,
            hidden_dim=args.hidden_dim,
            device=args.device,
        )
        print(f"  Average: {avg}")

    # -----------------------------------------------------------------------
    # Ablation 2: Label granularity
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ABLATION 2: Label Granularity (binary vs 5-class, ALL signals)")
    print("=" * 70)

    print("\n--- Binary (SleepWakeTask) ---")
    task_binary = SleepWakeTask(signal_subset="ALL")
    sample_ds = dataset.set_task(task_binary)
    samples_binary = [sample_ds[i] for i in range(len(sample_ds))]
    avg_binary = participant_cv(
        samples_binary,
        num_classes=2,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        device=args.device,
    )
    print(f"  Average: {avg_binary}")

    print("\n--- 5-class (SleepStageTask) ---")
    task_stage = SleepStageTask(signal_subset="ALL")
    sample_ds = dataset.set_task(task_stage)
    samples_stage = [sample_ds[i] for i in range(len(sample_ds))]
    avg_stage = participant_cv(
        samples_stage,
        num_classes=5,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        device=args.device,
    )
    print(f"  Average: {avg_stage}")

    # Per-class F1 for 5-class
    print("\n--- Per-class F1 (5-class, last fold) ---")
    patient_ids = sorted(set(s["patient_id"] for s in samples_stage))
    test_ids = set(patient_ids[-len(patient_ids) // 5 :])
    train_ids = set(patient_ids) - test_ids
    train_s = [s for s in samples_stage if s["patient_id"] in train_ids]
    test_s = [s for s in samples_stage if s["patient_id"] in test_ids]

    stage_names = ["Wake", "REM", "N1", "N2", "N3"]
    y_true = np.array([s["label"] for s in test_s])
    for i, name in enumerate(stage_names):
        mask = y_true == i
        if mask.sum() > 0:
            count = mask.sum()
            print(f"  {name}: count={count}")

    print("\nDone.")


if __name__ == "__main__":
    main()
