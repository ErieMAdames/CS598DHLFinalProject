"""Tests for SleepWakeTask and SleepStageTask using synthetic data.

All tests use in-memory fake patients with small temporary CSV files —
no real DREAMT data is required.  Tests should complete in milliseconds.
"""

import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pyhealth.tasks.sleep_wake_task import (
    BINARY_LABEL_MAP,
    EPOCH_LEN,
    STAGE_LABEL_MAP,
    SleepStageTask,
    SleepWakeTask,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_csv(
    n_epochs: int,
    stages: list,
    tmpdir: str,
    patient_id: str = "S001",
    flat_bvp: bool = False,
) -> str:
    """Create a synthetic 64 Hz CSV file with *n_epochs* epochs.

    Args:
        n_epochs: number of 30-second epochs.
        stages: list of sleep stage labels, one per epoch (cycled if shorter).
        tmpdir: directory to write the CSV into.
        patient_id: used in the filename.
        flat_bvp: if True, write constant BVP to simulate an artifact.

    Returns:
        Path to the written CSV file.
    """
    rng = np.random.RandomState(42)
    rows = n_epochs * EPOCH_LEN
    data = {
        "TIMESTAMP": np.arange(rows) / 64.0,
        "BVP": np.zeros(rows) if flat_bvp else rng.randn(rows) * 50,
        "IBI": np.clip(rng.rand(rows) * 0.2 + 0.7, 0, 2),
        "EDA": rng.rand(rows) * 5 + 0.1,
        "TEMP": rng.rand(rows) * 4 + 33,
        "ACC_X": rng.randn(rows) * 10,
        "ACC_Y": rng.randn(rows) * 10,
        "ACC_Z": rng.randn(rows) * 10,
        "HR": rng.rand(rows) * 30 + 60,
    }

    # Assign stage labels: each value repeats for EPOCH_LEN rows
    stage_col = []
    for i in range(n_epochs):
        stage = stages[i % len(stages)]
        stage_col.extend([stage] * EPOCH_LEN)
    data["Sleep_Stage"] = stage_col

    df = pd.DataFrame(data)
    path = os.path.join(tmpdir, f"{patient_id}_whole_df.csv")
    df.to_csv(path, index=False)
    return path


def _make_patient(file_path: str, patient_id: str = "S001"):
    """Build a mock patient object that mimics DREAMTDataset's Patient."""
    event = SimpleNamespace(file_64hz=file_path)
    patient = SimpleNamespace(
        patient_id=patient_id,
        get_events=lambda event_type=None: [event],
    )
    return patient


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSleepWakeTask:
    """Binary wake/sleep task tests."""

    def test_sample_count(self, tmp_path):
        """Correct number of non-Missing epochs returned."""
        stages = ["W", "N1", "N2", "N3", "R", "W", "Missing", "N2", "W", "R"]
        csv = _make_csv(10, stages, str(tmp_path))
        patient = _make_patient(csv)
        task = SleepWakeTask()
        samples = task(patient)
        # 10 epochs, 1 is Missing -> 9 valid
        assert len(samples) == 9

    def test_binary_labels(self, tmp_path):
        """W maps to 1; NREM and REM map to 0."""
        stages = ["W", "N1", "N2", "N3", "R"]
        csv = _make_csv(5, stages, str(tmp_path))
        patient = _make_patient(csv)
        task = SleepWakeTask()
        samples = task(patient)

        labels = [s["label"] for s in samples]
        # W=1, N1=0, N2=0, N3=0, R=0
        assert labels == [1, 0, 0, 0, 0]

    def test_missing_dropped(self, tmp_path):
        """Epochs labeled 'Missing' must be excluded."""
        stages = ["Missing", "Missing", "W"]
        csv = _make_csv(3, stages, str(tmp_path))
        patient = _make_patient(csv)
        task = SleepWakeTask()
        samples = task(patient)
        assert len(samples) == 1
        assert samples[0]["label"] == 1

    def test_signal_subset_acc(self, tmp_path):
        """ACC subset produces a feature vector."""
        stages = ["W", "N2"]
        csv = _make_csv(2, stages, str(tmp_path))
        patient = _make_patient(csv)
        task = SleepWakeTask(signal_subset="ACC")
        samples = task(patient)
        assert len(samples) == 2
        # ACC features: 4 per axis x 3 axes + 1 ACC_INDEX = 13
        assert samples[0]["signal"].shape[0] == 13

    def test_all_signals_wider(self, tmp_path):
        """ALL signal subset produces wider vector than ACC alone."""
        stages = ["W", "N2"]
        csv = _make_csv(2, stages, str(tmp_path))
        patient = _make_patient(csv)

        task_acc = SleepWakeTask(signal_subset="ACC")
        task_all = SleepWakeTask(signal_subset="ALL")
        samples_acc = task_acc(patient)
        samples_all = task_all(patient)

        dim_acc = samples_acc[0]["signal"].shape[0]
        dim_all = samples_all[0]["signal"].shape[0]
        assert dim_all > dim_acc

    def test_epoch_ordering(self, tmp_path):
        """epoch_idx is monotonically increasing per patient."""
        stages = ["W", "N1", "N2", "N3", "R"] * 4  # 20 epochs
        csv = _make_csv(20, stages, str(tmp_path))
        patient = _make_patient(csv)
        task = SleepWakeTask()
        samples = task(patient)

        indices = [s["epoch_idx"] for s in samples]
        assert indices == list(range(len(samples)))

    def test_empty_patient(self, tmp_path):
        """Patient with no valid file returns empty list."""
        event = SimpleNamespace(file_64hz=None)
        patient = SimpleNamespace(
            patient_id="S_EMPTY",
            get_events=lambda event_type=None: [event],
        )
        task = SleepWakeTask()
        samples = task(patient)
        assert samples == []

    def test_artifact_epoch(self, tmp_path):
        """Epoch with flat BVP (artifact) is handled without crash."""
        stages = ["W"]
        csv = _make_csv(1, stages, str(tmp_path), flat_bvp=True)
        patient = _make_patient(csv)
        task = SleepWakeTask()
        samples = task(patient)
        # Should either return the epoch (with zeroed features) or skip it
        # -- the key requirement is no exception
        assert isinstance(samples, list)


class TestSleepStageTask:
    """5-class staging task tests."""

    def test_five_class_labels(self, tmp_path):
        """SleepStageTask maps all 5 stages correctly."""
        stages = ["W", "R", "N1", "N2", "N3"]
        csv = _make_csv(5, stages, str(tmp_path))
        patient = _make_patient(csv)
        task = SleepStageTask()
        samples = task(patient)

        labels = [s["label"] for s in samples]
        expected = [
            STAGE_LABEL_MAP["W"],   # 0
            STAGE_LABEL_MAP["R"],   # 1
            STAGE_LABEL_MAP["N1"],  # 2
            STAGE_LABEL_MAP["N2"],  # 3
            STAGE_LABEL_MAP["N3"],  # 4
        ]
        assert labels == expected

    def test_multiclass_schema(self):
        """Output schema should be multiclass."""
        task = SleepStageTask()
        assert task.output_schema == {"label": "multiclass"}
        assert task.task_name == "SleepStageClassification"
