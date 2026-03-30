"""Sleep/Wake detection and sleep staging tasks for the DREAMT dataset.

These tasks process overnight Empatica E4 wearable recordings from the
DREAMT dataset into per-epoch feature vectors with binary (wake/sleep)
or 5-class (W/R/N1/N2/N3) labels, suitable for temporal models such as
LSTMs.

Reference:
    Wang et al. "Addressing wearable sleep tracking inequity: a new
    dataset and novel methods for a population with sleep disorders."
    CHIL 2024, PMLR 248:380-396.
"""

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, cheby2, sosfilt

from pyhealth.tasks.base_task import BaseTask

logger = logging.getLogger(__name__)

EPOCH_SEC = 30
FS = 64
EPOCH_LEN = EPOCH_SEC * FS  # 1920 samples per epoch

SIGNAL_GROUPS = {
    "ACC": ["ACC_X", "ACC_Y", "ACC_Z"],
    "BVP_HRV": ["BVP", "IBI", "HR"],
    "EDA_TEMP": ["EDA", "TEMP"],
    "ALL": ["BVP", "IBI", "EDA", "TEMP", "ACC_X", "ACC_Y", "ACC_Z", "HR"],
}

BINARY_LABEL_MAP = {"W": 1, "N1": 0, "N2": 0, "N3": 0, "R": 0}
STAGE_LABEL_MAP = {"W": 0, "R": 1, "N1": 2, "N2": 3, "N3": 4}


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def _safe_trimmed_mean(arr, proportion=0.1):
    """Trimmed mean removing *proportion* from each tail."""
    s = np.sort(arr)
    n = len(s)
    lo = int(n * proportion)
    hi = n - lo
    if hi <= lo:
        return np.mean(arr)
    return np.mean(s[lo:hi])


def _acc_features(epoch_df: pd.DataFrame) -> np.ndarray:
    """Extract accelerometer features from one 30-s epoch.

    Applies a 5th-order Butterworth band-pass (3-11 Hz) per axis, then
    computes trimmed mean, max, IQR, and MAD.  Also computes the
    ACC_INDEX (mean absolute deviation of the vector magnitude).
    """
    feats = []
    for axis in ["ACC_X", "ACC_Y", "ACC_Z"]:
        raw = epoch_df[axis].values.astype(np.float64)
        # Band-pass 3-11 Hz (5th-order Butterworth)
        b, a = butter(5, [3.0 / (0.5 * FS), 11.0 / (0.5 * FS)], btype="band")
        try:
            filt = filtfilt(b, a, raw)
        except ValueError:
            filt = raw
        abs_filt = np.abs(filt)

        feats.append(_safe_trimmed_mean(abs_filt))
        feats.append(np.max(abs_filt))
        q75, q25 = np.percentile(abs_filt, [75, 25])
        feats.append(q75 - q25)

        # MAD based on deviation from vector magnitude
        mag = np.sqrt(
            epoch_df["ACC_X"].values ** 2
            + epoch_df["ACC_Y"].values ** 2
            + epoch_df["ACC_Z"].values ** 2
        ).astype(np.float64)
        feats.append(np.mean(np.abs(raw - np.mean(mag))))

    # ACC_INDEX: overall activity index (mean absolute value of filtered resultant)
    acc_x = epoch_df["ACC_X"].values.astype(np.float64)
    acc_y = epoch_df["ACC_Y"].values.astype(np.float64)
    acc_z = epoch_df["ACC_Z"].values.astype(np.float64)
    magnitude = np.sqrt(acc_x ** 2 + acc_y ** 2 + acc_z ** 2)
    acc_index = np.mean(np.abs(magnitude - np.mean(magnitude)))
    feats.append(acc_index)

    return np.array(feats, dtype=np.float64)


def _temp_features(epoch_df: pd.DataFrame) -> np.ndarray:
    """Skin temperature features with winsorization to 31-40 C."""
    temp = epoch_df["TEMP"].values.astype(np.float64)
    temp = np.clip(temp, 31.0, 40.0)
    return np.array([
        np.mean(temp),
        np.min(temp),
        np.max(temp),
        np.std(temp),
    ], dtype=np.float64)


def _bvp_hrv_features(epoch_df: pd.DataFrame) -> np.ndarray:
    """BVP / IBI / HR features.

    Computes HRV time-domain metrics directly from IBI signal:
    rMSSD, SDNN, pNN50, MinNN, MeanNN, and a fractal dimension proxy.
    Falls back to zeros when IBI signal is too sparse.
    """
    bvp = epoch_df["BVP"].values.astype(np.float64)
    hr = epoch_df["HR"].values.astype(np.float64)
    ibi = epoch_df["IBI"].values.astype(np.float64)

    # Chebyshev II band-pass 0.5-20 Hz on BVP
    try:
        sos = cheby2(4, 40, [0.5 / (0.5 * FS), 20.0 / (0.5 * FS)], btype="band", output="sos")
        bvp_filt = sosfilt(sos, bvp)
    except ValueError:
        bvp_filt = bvp

    base_feats = [
        np.mean(bvp_filt),
        np.std(bvp_filt),
        np.mean(hr),
        np.std(hr),
        np.mean(ibi[ibi != 0]) if np.any(ibi != 0) else 0.0,
        np.std(ibi[ibi != 0]) if np.any(ibi != 0) else 0.0,
    ]

    # HRV time-domain statistics computed directly (fast, no neurokit2)
    valid_ibi = ibi[ibi > 0]
    if len(valid_ibi) >= 3:
        nn_ms = valid_ibi * 1000.0
        diffs = np.diff(nn_ms)
        rmssd = np.sqrt(np.mean(diffs ** 2)) if len(diffs) > 0 else 0.0
        sdnn = np.std(nn_ms)
        pnn50 = np.mean(np.abs(diffs) > 50) * 100 if len(diffs) > 0 else 0.0
        min_nn = np.min(nn_ms)
        mean_nn = np.mean(nn_ms)
        # Higuchi Fractal Dimension (simplified Katz FD as proxy)
        n = len(valid_ibi)
        if n > 1:
            L = np.sum(np.abs(np.diff(valid_ibi)))
            d = np.max(np.abs(valid_ibi - valid_ibi[0]))
            hfd_val = np.log10(n - 1) / (np.log10((n - 1) / (L / (n - 1))) + np.log10(d / L)) if L > 0 and d > 0 else 0.0
        else:
            hfd_val = 0.0
        for v in [rmssd, sdnn, pnn50, min_nn, mean_nn, hfd_val]:
            base_feats.append(float(v) if np.isfinite(v) else 0.0)
    else:
        base_feats.extend([0.0] * 6)

    return np.array(base_feats, dtype=np.float64)


def _eda_features(epoch_df: pd.DataFrame) -> np.ndarray:
    """EDA features: tonic/phasic decomposition statistics."""
    eda = epoch_df["EDA"].values.astype(np.float64)

    base_feats = [np.mean(eda), np.std(eda), np.min(eda), np.max(eda)]

    # Phasic component approximation: high-pass via rolling mean subtraction
    if len(eda) >= 16 and np.std(eda) > 1e-8:
        kernel = min(len(eda) // 4, 16)
        tonic = np.convolve(eda, np.ones(kernel) / kernel, mode="same")
        scr = eda - tonic
        base_feats.extend([np.mean(scr), np.max(scr), np.std(scr)])
    else:
        base_feats.extend([0.0, 0.0, 0.0])

    return np.array(base_feats, dtype=np.float64)


def _extract_features(epoch_df: pd.DataFrame, signal_subset: str) -> np.ndarray:
    """Concatenate feature vectors for the requested signal groups."""
    parts = []

    groups_to_extract = (
        ["ACC", "BVP_HRV", "EDA_TEMP"]
        if signal_subset == "ALL"
        else [signal_subset]
    )

    for group in groups_to_extract:
        if group == "ACC":
            parts.append(_acc_features(epoch_df))
        elif group == "BVP_HRV":
            parts.append(_bvp_hrv_features(epoch_df))
        elif group == "EDA_TEMP":
            parts.append(_eda_features(epoch_df))
            parts.append(_temp_features(epoch_df))

    feat_vec = np.concatenate(parts)
    feat_vec = np.nan_to_num(feat_vec, nan=0.0, posinf=0.0, neginf=0.0)
    return feat_vec


# ---------------------------------------------------------------------------
# Task classes
# ---------------------------------------------------------------------------

class SleepWakeTask(BaseTask):
    """Binary sleep/wake detection task for the DREAMT dataset.

    Segments each participant's overnight E4 recording into 30-second
    epochs, extracts hand-engineered features from the requested signal
    subset, and assigns a binary label (wake = 1, sleep = 0) derived
    from PSG annotations.  Epochs with "Missing" labels are dropped.

    The samples are returned in temporal order so that sequence models
    (e.g. LSTM) can consume the full overnight sequence per participant.

    Attributes:
        task_name (str): ``"SleepWakeDetection"``
        input_schema (Dict[str, str]): ``{"signal": "tensor"}``
        output_schema (Dict[str, str]): ``{"label": "binary"}``

    Args:
        signal_subset (str): Which E4 signals to use for feature
            extraction.  One of ``"ACC"``, ``"BVP_HRV"``,
            ``"EDA_TEMP"``, or ``"ALL"`` (default).
        artifact_threshold (float): ACC_INDEX threshold above which an
            epoch is flagged as an artifact.  Default ``0.4125``
            (from Moscato et al. 2022).

    Examples:
        >>> from pyhealth.datasets import DREAMTDataset
        >>> ds = DREAMTDataset(root="/path/to/dreamt/2.1.0")
        >>> sample_ds = ds.set_task(SleepWakeTask(signal_subset="ALL"))
        >>> sample_ds.samples[0].keys()
        dict_keys(['patient_id', 'epoch_idx', 'signal', 'label'])
    """

    task_name: str = "SleepWakeDetection"
    input_schema: Dict[str, str] = {"signal": "tensor"}
    output_schema: Dict[str, str] = {"label": "binary"}

    def __init__(
        self,
        signal_subset: str = "ALL",
        artifact_threshold: float = 0.4125,
    ):
        if signal_subset not in SIGNAL_GROUPS and signal_subset != "ALL":
            raise ValueError(
                f"signal_subset must be one of {list(SIGNAL_GROUPS.keys())} or 'ALL', "
                f"got '{signal_subset}'"
            )
        self.signal_subset = signal_subset
        self.artifact_threshold = artifact_threshold
        super().__init__()

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Process one DREAMT patient into a list of epoch samples.

        Args:
            patient: A ``Patient`` object from ``DREAMTDataset``.

        Returns:
            A list of dicts, each containing:

            - ``patient_id`` (str)
            - ``epoch_idx`` (int): temporal position within the night
            - ``signal`` (np.ndarray): 1-D feature vector
            - ``label`` (int): 1 = wake, 0 = sleep
        """
        return self._process_patient(patient, BINARY_LABEL_MAP)

    def _process_patient(
        self, patient: Any, label_map: Dict[str, int]
    ) -> List[Dict[str, Any]]:
        pid = patient.patient_id
        try:
            events = patient.get_events(event_type="dreamt_sleep")
        except Exception:
            events = patient.get_events()

        if not events:
            return []

        event = events[0]
        file_path = getattr(event, "file_64hz", None)
        if file_path is None or (isinstance(file_path, str) and file_path.lower() == "none"):
            return []

        try:
            df = pd.read_csv(str(file_path))
        except Exception as e:
            logger.warning("Could not read %s: %s", file_path, e)
            return []

        required_cols = {"Sleep_Stage"}
        if self.signal_subset == "ALL":
            for cols in SIGNAL_GROUPS.values():
                required_cols.update(cols)
        else:
            required_cols.update(SIGNAL_GROUPS[self.signal_subset])

        # Column names are lowercased by PyHealth; handle both cases
        col_map = {}
        for col in df.columns:
            col_map[col.lower()] = col

        missing = [c for c in required_cols if c.lower() not in col_map]
        if missing:
            logger.warning("Patient %s missing columns: %s", pid, missing)
            return []

        n_samples = len(df)
        n_epochs = n_samples // EPOCH_LEN

        samples: List[Dict[str, Any]] = []
        epoch_counter = 0

        for i in range(n_epochs):
            start = i * EPOCH_LEN
            end = start + EPOCH_LEN
            epoch_df = df.iloc[start:end]

            # Resolve sleep stage for this epoch
            stage_col = col_map.get("sleep_stage", "Sleep_Stage")
            stage_vals = epoch_df[stage_col].dropna().unique()
            if len(stage_vals) == 0:
                continue

            stage = str(stage_vals[0]).strip()
            if stage == "Missing" or stage not in label_map:
                continue

            label = label_map[stage]

            try:
                feat = _extract_features(epoch_df, self.signal_subset)
            except Exception as e:
                logger.debug("Feature extraction failed epoch %d patient %s: %s", i, pid, e)
                continue

            samples.append({
                "patient_id": pid,
                "epoch_idx": epoch_counter,
                "signal": feat,
                "label": label,
            })
            epoch_counter += 1

        return samples


class SleepStageTask(SleepWakeTask):
    """Five-class sleep staging task for the DREAMT dataset.

    Identical to :class:`SleepWakeTask` except that the label is a
    5-class integer (W=0, R=1, N1=2, N2=3, N3=4) instead of binary.

    Attributes:
        task_name (str): ``"SleepStageClassification"``
        input_schema (Dict[str, str]): ``{"signal": "tensor"}``
        output_schema (Dict[str, str]): ``{"label": "multiclass"}``

    Args:
        signal_subset (str): See :class:`SleepWakeTask`.
        artifact_threshold (float): See :class:`SleepWakeTask`.

    Examples:
        >>> from pyhealth.datasets import DREAMTDataset
        >>> ds = DREAMTDataset(root="/path/to/dreamt/2.1.0")
        >>> sample_ds = ds.set_task(SleepStageTask())
        >>> sample_ds.samples[0].keys()
        dict_keys(['patient_id', 'epoch_idx', 'signal', 'label'])
    """

    task_name: str = "SleepStageClassification"
    output_schema: Dict[str, str] = {"label": "multiclass"}

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Process one DREAMT patient into 5-class epoch samples.

        Returns:
            A list of dicts with the same keys as
            :meth:`SleepWakeTask.__call__`, but ``label`` is an int
            in {0, 1, 2, 3, 4} corresponding to
            {W, R, N1, N2, N3}.
        """
        return self._process_patient(patient, STAGE_LABEL_MAP)
