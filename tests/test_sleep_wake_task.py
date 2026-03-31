"""Tests for SleepWakeTask and SleepStageTask.

All tests use in-memory fake patients with small temporary CSV
files — no real DREAMT data is required.  Tests complete in
milliseconds.
"""

import os
from types import SimpleNamespace
from typing import List, Optional

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

# -----------------------------------------------------------
# Helpers
# -----------------------------------------------------------


def _make_csv(
    n_epochs: int,
    stages: List[str],
    tmpdir: str,
    patient_id: str = "S001",
    flat_bvp: bool = False,
) -> str:
    """Create a synthetic 64 Hz CSV with *n_epochs* epochs.

    Args:
        n_epochs: Number of 30-second epochs to generate.
        stages: Sleep stage labels, one per epoch (cycled).
        tmpdir: Directory to write the CSV into.
        patient_id: Used in the filename.
        flat_bvp: If True, write constant-zero BVP.

    Returns:
        Absolute path to the written CSV file.
    """
    rng = np.random.RandomState(42)
    rows = n_epochs * EPOCH_LEN
    data = {
        "TIMESTAMP": np.arange(rows) / 64.0,
        "BVP": (np.zeros(rows) if flat_bvp else rng.randn(rows) * 50),
        "IBI": np.clip(rng.rand(rows) * 0.2 + 0.7, 0, 2),
        "EDA": rng.rand(rows) * 5 + 0.1,
        "TEMP": rng.rand(rows) * 4 + 33,
        "ACC_X": rng.randn(rows) * 10,
        "ACC_Y": rng.randn(rows) * 10,
        "ACC_Z": rng.randn(rows) * 10,
        "HR": rng.rand(rows) * 30 + 60,
    }

    stage_col = []
    for i in range(n_epochs):
        stage = stages[i % len(stages)]
        stage_col.extend([stage] * EPOCH_LEN)
    data["Sleep_Stage"] = stage_col

    df = pd.DataFrame(data)
    path = os.path.join(tmpdir, f"{patient_id}_whole_df.csv")
    df.to_csv(path, index=False)
    return path


def _make_patient(
    file_path: Optional[str],
    patient_id: str = "S001",
) -> SimpleNamespace:
    """Build a mock Patient mimicking DREAMTDataset.

    Args:
        file_path: Path to the CSV, or None for an empty
            patient.
        patient_id: Identifier for the mock patient.

    Returns:
        SimpleNamespace with ``patient_id`` and
        ``get_events`` matching the DREAMT contract.
    """
    event = SimpleNamespace(file_64hz=file_path)
    patient = SimpleNamespace(
        patient_id=patient_id,
        get_events=lambda event_type=None: [event],
    )
    return patient


# -----------------------------------------------------------
# Tests — SleepWakeTask (binary)
# -----------------------------------------------------------


class TestSleepWakeTask:
    """Binary wake/sleep task tests."""

    def test_sample_count(self, tmp_path):
        """Correct number of non-Missing epochs returned."""
        stages = [
            "W",
            "N1",
            "N2",
            "N3",
            "R",
            "W",
            "Missing",
            "N2",
            "W",
            "R",
        ]
        csv = _make_csv(10, stages, str(tmp_path))
        patient = _make_patient(csv)
        task = SleepWakeTask(artifact_threshold=None)
        samples = task(patient)
        assert len(samples) == 9

    def test_binary_labels(self, tmp_path):
        """W maps to 1; NREM and REM map to 0."""
        stages = ["W", "N1", "N2", "N3", "R"]
        csv = _make_csv(5, stages, str(tmp_path))
        patient = _make_patient(csv)
        task = SleepWakeTask(artifact_threshold=None)
        samples = task(patient)
        labels = [s["label"] for s in samples]
        assert labels == [1, 0, 0, 0, 0]

    def test_missing_dropped(self, tmp_path):
        """Epochs labeled 'Missing' must be excluded."""
        stages = ["Missing", "Missing", "W"]
        csv = _make_csv(3, stages, str(tmp_path))
        patient = _make_patient(csv)
        task = SleepWakeTask(artifact_threshold=None)
        samples = task(patient)
        assert len(samples) == 1
        assert samples[0]["label"] == 1

    def test_signal_subset_acc(self, tmp_path):
        """ACC subset produces correct feature dimension."""
        stages = ["W", "N2"]
        csv = _make_csv(2, stages, str(tmp_path))
        patient = _make_patient(csv)
        task = SleepWakeTask(
            signal_subset="ACC",
            artifact_threshold=None,
        )
        samples = task(patient)
        assert len(samples) == 2
        assert samples[0]["signal"].shape[0] == 13

    def test_all_signals_wider(self, tmp_path):
        """ALL produces wider vector than ACC alone."""
        stages = ["W", "N2"]
        csv = _make_csv(2, stages, str(tmp_path))
        patient = _make_patient(csv)

        task_acc = SleepWakeTask(
            signal_subset="ACC",
            artifact_threshold=None,
        )
        task_all = SleepWakeTask(
            signal_subset="ALL",
            artifact_threshold=None,
        )
        samples_acc = task_acc(patient)
        samples_all = task_all(patient)
        assert samples_all[0]["signal"].shape[0] > samples_acc[0]["signal"].shape[0]

    def test_epoch_ordering(self, tmp_path):
        """epoch_idx is monotonically increasing."""
        stages = ["W", "N1", "N2", "N3", "R"] * 4
        csv = _make_csv(20, stages, str(tmp_path))
        patient = _make_patient(csv)
        task = SleepWakeTask(artifact_threshold=None)
        samples = task(patient)
        indices = [s["epoch_idx"] for s in samples]
        assert indices == list(range(len(samples)))

    def test_empty_patient(self, tmp_path):
        """Patient with no valid file returns []."""
        patient = _make_patient(None, patient_id="S_EMPTY")
        task = SleepWakeTask(artifact_threshold=None)
        samples = task(patient)
        assert samples == []

    def test_artifact_epoch(self, tmp_path):
        """Flat BVP (artifact) handled without crash."""
        stages = ["W"]
        csv = _make_csv(
            1,
            stages,
            str(tmp_path),
            flat_bvp=True,
        )
        patient = _make_patient(csv)
        task = SleepWakeTask(artifact_threshold=None)
        samples = task(patient)
        assert isinstance(samples, list)

    def test_artifact_threshold_drops(self, tmp_path):
        """artifact_threshold=0 drops high-motion epochs."""
        stages = ["W", "N2"]
        csv = _make_csv(2, stages, str(tmp_path))
        patient = _make_patient(csv)
        task_none = SleepWakeTask(artifact_threshold=None)
        task_zero = SleepWakeTask(artifact_threshold=0.0)
        assert len(task_none(patient)) >= len(task_zero(patient))

    def test_invalid_signal_subset(self):
        """Invalid signal_subset raises ValueError."""
        with pytest.raises(ValueError, match="signal_subset"):
            SleepWakeTask(signal_subset="INVALID")

    def test_signal_subset_bvp_hrv(self, tmp_path):
        """BVP_HRV subset produces 12-dim feature vector."""
        stages = ["W", "N2"]
        csv = _make_csv(2, stages, str(tmp_path))
        patient = _make_patient(csv)
        task = SleepWakeTask(
            signal_subset="BVP_HRV",
            artifact_threshold=None,
        )
        samples = task(patient)
        assert len(samples) == 2
        assert samples[0]["signal"].shape[0] == 12

    def test_signal_subset_eda_temp(self, tmp_path):
        """EDA_TEMP subset produces 11-dim feature vector."""
        stages = ["W", "N2"]
        csv = _make_csv(2, stages, str(tmp_path))
        patient = _make_patient(csv)
        task = SleepWakeTask(
            signal_subset="EDA_TEMP",
            artifact_threshold=None,
        )
        samples = task(patient)
        assert len(samples) == 2
        # EDA (7) + TEMP (4) = 11
        assert samples[0]["signal"].shape[0] == 11

    def test_patient_id_in_samples(self, tmp_path):
        """Output samples carry the correct patient_id."""
        csv = _make_csv(
            3,
            ["W", "N2", "R"],
            str(tmp_path),
            patient_id="S042",
        )
        patient = _make_patient(csv, patient_id="S042")
        task = SleepWakeTask(artifact_threshold=None)
        samples = task(patient)
        assert all(s["patient_id"] == "S042" for s in samples)

    def test_multi_patient_no_leakage(self, tmp_path):
        """Each patient's samples reference only its own id."""
        csv_a = _make_csv(
            3,
            ["W", "N1", "N2"],
            str(tmp_path),
            patient_id="P_A",
        )
        csv_b = _make_csv(
            2,
            ["N3", "R"],
            str(tmp_path),
            patient_id="P_B",
        )
        patient_a = _make_patient(csv_a, patient_id="P_A")
        patient_b = _make_patient(csv_b, patient_id="P_B")

        task = SleepWakeTask(artifact_threshold=None)
        samples_a = task(patient_a)
        samples_b = task(patient_b)

        ids_a = set(s["patient_id"] for s in samples_a)
        ids_b = set(s["patient_id"] for s in samples_b)
        assert ids_a == {"P_A"}
        assert ids_b == {"P_B"}
        assert ids_a.isdisjoint(ids_b)

        # epoch_idx restarts at 0 for each patient
        assert samples_a[0]["epoch_idx"] == 0
        assert samples_b[0]["epoch_idx"] == 0


# -----------------------------------------------------------
# Tests — SleepStageTask (5-class)
# -----------------------------------------------------------


class TestSleepStageTask:
    """5-class staging task tests."""

    def test_five_class_labels(self, tmp_path):
        """SleepStageTask maps all 5 stages correctly."""
        stages = ["W", "R", "N1", "N2", "N3"]
        csv = _make_csv(5, stages, str(tmp_path))
        patient = _make_patient(csv)
        task = SleepStageTask(artifact_threshold=None)
        samples = task(patient)
        labels = [s["label"] for s in samples]
        expected = [
            STAGE_LABEL_MAP["W"],
            STAGE_LABEL_MAP["R"],
            STAGE_LABEL_MAP["N1"],
            STAGE_LABEL_MAP["N2"],
            STAGE_LABEL_MAP["N3"],
        ]
        assert labels == expected

    def test_multiclass_schema(self):
        """Output schema should be multiclass."""
        task = SleepStageTask()
        assert task.output_schema == {"label": "multiclass"}
        assert task.task_name == "SleepStageClassification"
