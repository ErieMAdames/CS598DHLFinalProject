"""Sleep/Wake detection and sleep staging tasks for DREAMT.

These tasks process overnight Empatica E4 wearable recordings
from the DREAMT dataset into per-epoch feature vectors with
binary (wake/sleep) or 5-class (W/R/N1/N2/N3) labels, suitable
for temporal models such as LSTMs.

Reference:
    Wang et al. "Addressing wearable sleep tracking inequity:
    a new dataset and novel methods for a population with sleep
    disorders." CHIL 2024, PMLR 248:380-396.
"""

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.signal import butter, cheby2, filtfilt, sosfilt

from pyhealth.tasks.base_task import BaseTask

logger = logging.getLogger(__name__)

EPOCH_SEC: int = 30
FS: int = 64
EPOCH_LEN: int = EPOCH_SEC * FS  # 1920 samples per epoch

SIGNAL_GROUPS: Dict[str, List[str]] = {
    "ACC": ["ACC_X", "ACC_Y", "ACC_Z"],
    "BVP_HRV": ["BVP", "IBI", "HR"],
    "EDA_TEMP": ["EDA", "TEMP"],
    "ALL": [
        "BVP", "IBI", "EDA", "TEMP",
        "ACC_X", "ACC_Y", "ACC_Z", "HR",
    ],
}

BINARY_LABEL_MAP: Dict[str, int] = {
    "W": 1, "N1": 0, "N2": 0, "N3": 0, "R": 0,
}
STAGE_LABEL_MAP: Dict[str, int] = {
    "W": 0, "R": 1, "N1": 2, "N2": 3, "N3": 4,
}


# ---------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------


def _safe_trimmed_mean(
    arr: np.ndarray,
    proportion: float = 0.1,
) -> float:
    """Trimmed mean removing *proportion* from each tail.

    Args:
        arr: 1-D array of numeric values.
        proportion: Fraction of values to trim from each end.

    Returns:
        The trimmed mean as a float.
    """
    s = np.sort(arr)
    n = len(s)
    lo = int(n * proportion)
    hi = n - lo
    if hi <= lo:
        return float(np.mean(arr))
    return float(np.mean(s[lo:hi]))


def _acc_features(epoch_df: pd.DataFrame) -> np.ndarray:
    """Extract accelerometer features from one 30-s epoch.

    Applies a 5th-order Butterworth band-pass filter (3-11 Hz)
    per axis, then computes trimmed mean, max, IQR, and MAD.
    Also computes the ACC_INDEX (mean absolute deviation of the
    vector magnitude).

    Args:
        epoch_df: DataFrame slice with columns ACC_X, ACC_Y,
            ACC_Z at 64 Hz (``EPOCH_LEN`` rows).

    Returns:
        1-D feature array of length 13 (4 per axis + 1 index).
    """
    feats: List[float] = []
    nyq = 0.5 * FS
    for axis in ["ACC_X", "ACC_Y", "ACC_Z"]:
        raw = epoch_df[axis].values.astype(np.float64)
        b, a = butter(
            5, [3.0 / nyq, 11.0 / nyq], btype="band",
        )
        try:
            filt = filtfilt(b, a, raw)
        except ValueError:
            filt = raw
        abs_filt = np.abs(filt)

        feats.append(_safe_trimmed_mean(abs_filt))
        feats.append(float(np.max(abs_filt)))
        q75, q25 = np.percentile(abs_filt, [75, 25])
        feats.append(float(q75 - q25))

        mag = np.sqrt(
            epoch_df["ACC_X"].values ** 2
            + epoch_df["ACC_Y"].values ** 2
            + epoch_df["ACC_Z"].values ** 2
        ).astype(np.float64)
        feats.append(float(np.mean(np.abs(raw - np.mean(mag)))))

    acc_x = epoch_df["ACC_X"].values.astype(np.float64)
    acc_y = epoch_df["ACC_Y"].values.astype(np.float64)
    acc_z = epoch_df["ACC_Z"].values.astype(np.float64)
    magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    acc_index = float(
        np.mean(np.abs(magnitude - np.mean(magnitude)))
    )
    feats.append(acc_index)

    return np.array(feats, dtype=np.float64)


def _temp_features(epoch_df: pd.DataFrame) -> np.ndarray:
    """Extract skin temperature features with winsorization.

    Clips temperature values to [31, 40] C, then computes mean,
    min, max, and standard deviation.

    Args:
        epoch_df: DataFrame slice with a TEMP column at 64 Hz.

    Returns:
        1-D feature array of length 4.
    """
    temp = epoch_df["TEMP"].values.astype(np.float64)
    temp = np.clip(temp, 31.0, 40.0)
    return np.array(
        [np.mean(temp), np.min(temp), np.max(temp), np.std(temp)],
        dtype=np.float64,
    )


def _bvp_hrv_features(epoch_df: pd.DataFrame) -> np.ndarray:
    """Extract BVP, HR, IBI and HRV time-domain features.

    Applies a Chebyshev Type II band-pass filter (0.5-20 Hz) to
    BVP, then computes mean/std for BVP, HR, and IBI.  Derives
    HRV time-domain metrics (rMSSD, SDNN, pNN50, MinNN, MeanNN)
    and a simplified Katz fractal dimension proxy from valid IBI
    intervals.  Falls back to zeros when IBI is too sparse.

    Args:
        epoch_df: DataFrame slice with BVP, HR, IBI columns.

    Returns:
        1-D feature array of length 12.
    """
    bvp = epoch_df["BVP"].values.astype(np.float64)
    hr = epoch_df["HR"].values.astype(np.float64)
    ibi = epoch_df["IBI"].values.astype(np.float64)

    nyq = 0.5 * FS
    try:
        sos = cheby2(
            4, 40,
            [0.5 / nyq, 20.0 / nyq],
            btype="band",
            output="sos",
        )
        bvp_filt = sosfilt(sos, bvp)
    except ValueError:
        bvp_filt = bvp

    nz_ibi = ibi[ibi != 0]
    base_feats: List[float] = [
        float(np.mean(bvp_filt)),
        float(np.std(bvp_filt)),
        float(np.mean(hr)),
        float(np.std(hr)),
        float(np.mean(nz_ibi)) if len(nz_ibi) else 0.0,
        float(np.std(nz_ibi)) if len(nz_ibi) else 0.0,
    ]

    valid_ibi = ibi[ibi > 0]
    if len(valid_ibi) >= 3:
        nn_ms = valid_ibi * 1000.0
        diffs = np.diff(nn_ms)
        rmssd = (
            np.sqrt(np.mean(diffs**2))
            if len(diffs) > 0 else 0.0
        )
        sdnn = float(np.std(nn_ms))
        pnn50 = (
            float(np.mean(np.abs(diffs) > 50) * 100)
            if len(diffs) > 0 else 0.0
        )
        min_nn = float(np.min(nn_ms))
        mean_nn = float(np.mean(nn_ms))

        n = len(valid_ibi)
        hfd_val = 0.0
        if n > 1:
            total_len = np.sum(np.abs(np.diff(valid_ibi)))
            diameter = np.max(
                np.abs(valid_ibi - valid_ibi[0])
            )
            if total_len > 0 and diameter > 0:
                avg_step = total_len / (n - 1)
                hfd_val = np.log10(n - 1) / (
                    np.log10((n - 1) / avg_step)
                    + np.log10(diameter / total_len)
                )

        for v in [
            rmssd, sdnn, pnn50, min_nn, mean_nn, hfd_val,
        ]:
            val = float(v) if np.isfinite(v) else 0.0
            base_feats.append(val)
    else:
        base_feats.extend([0.0] * 6)

    return np.array(base_feats, dtype=np.float64)


def _eda_features(epoch_df: pd.DataFrame) -> np.ndarray:
    """Extract EDA features with tonic/phasic decomposition.

    Computes mean, std, min, max of raw EDA, then approximates
    the phasic (SCR) component via rolling-mean subtraction and
    extracts mean, max, and std of the phasic signal.

    Args:
        epoch_df: DataFrame slice with an EDA column at 64 Hz.

    Returns:
        1-D feature array of length 7.
    """
    eda = epoch_df["EDA"].values.astype(np.float64)

    base_feats: List[float] = [
        float(np.mean(eda)),
        float(np.std(eda)),
        float(np.min(eda)),
        float(np.max(eda)),
    ]

    if len(eda) >= 16 and np.std(eda) > 1e-8:
        kernel = min(len(eda) // 4, 16)
        tonic = np.convolve(
            eda, np.ones(kernel) / kernel, mode="same",
        )
        scr = eda - tonic
        base_feats.extend([
            float(np.mean(scr)),
            float(np.max(scr)),
            float(np.std(scr)),
        ])
    else:
        base_feats.extend([0.0, 0.0, 0.0])

    return np.array(base_feats, dtype=np.float64)


def _extract_features(
    epoch_df: pd.DataFrame,
    signal_subset: str,
) -> np.ndarray:
    """Concatenate feature vectors for requested signal groups.

    Args:
        epoch_df: DataFrame slice for one 30-s epoch.
        signal_subset: One of ``"ACC"``, ``"BVP_HRV"``,
            ``"EDA_TEMP"``, or ``"ALL"``.

    Returns:
        1-D feature array with NaNs replaced by zero.
    """
    parts: List[np.ndarray] = []

    groups = (
        ["ACC", "BVP_HRV", "EDA_TEMP"]
        if signal_subset == "ALL"
        else [signal_subset]
    )

    for group in groups:
        if group == "ACC":
            parts.append(_acc_features(epoch_df))
        elif group == "BVP_HRV":
            parts.append(_bvp_hrv_features(epoch_df))
        elif group == "EDA_TEMP":
            parts.append(_eda_features(epoch_df))
            parts.append(_temp_features(epoch_df))

    feat_vec = np.concatenate(parts)
    feat_vec = np.nan_to_num(
        feat_vec, nan=0.0, posinf=0.0, neginf=0.0,
    )
    return feat_vec


def _compute_acc_index(epoch_df: pd.DataFrame) -> float:
    """Compute ACC_INDEX for artifact detection.

    ACC_INDEX is defined as the mean absolute deviation of the
    tri-axial accelerometer vector magnitude from its mean
    (Moscato et al. 2022).

    Args:
        epoch_df: DataFrame slice with ACC_X, ACC_Y, ACC_Z.

    Returns:
        ACC_INDEX value as a float.
    """
    acc_x = epoch_df["ACC_X"].values.astype(np.float64)
    acc_y = epoch_df["ACC_Y"].values.astype(np.float64)
    acc_z = epoch_df["ACC_Z"].values.astype(np.float64)
    magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    return float(
        np.mean(np.abs(magnitude - np.mean(magnitude)))
    )


# ---------------------------------------------------------------
# Task classes
# ---------------------------------------------------------------


class SleepWakeTask(BaseTask):
    """Binary sleep/wake detection task for the DREAMT dataset.

    Segments each participant's overnight E4 recording into
    30-second epochs, extracts hand-engineered features from the
    requested signal subset, and assigns a binary label
    (wake = 1, sleep = 0) derived from PSG annotations.

    Epochs with ``"Missing"`` labels are dropped.  Epochs whose
    ACC_INDEX exceeds ``artifact_threshold`` are also dropped to
    reduce motion-artifact contamination (Moscato et al. 2022).

    Samples are returned in temporal order so that sequence
    models (e.g. LSTM) can consume the full overnight sequence
    per participant.

    Attributes:
        task_name: ``"SleepWakeDetection"``
        input_schema: ``{"signal": "tensor"}``
        output_schema: ``{"label": "binary"}``

    Args:
        signal_subset: Which E4 signals to use for feature
            extraction.  One of ``"ACC"``, ``"BVP_HRV"``,
            ``"EDA_TEMP"``, or ``"ALL"`` (default).
        artifact_threshold: ACC_INDEX value above which an
            epoch is discarded as a motion artifact.  Default
            ``0.4125`` (from Moscato et al. 2022).  Set to
            ``None`` to disable artifact rejection.

    Examples:
        >>> from pyhealth.datasets import DREAMTDataset
        >>> ds = DREAMTDataset(
        ...     root="/path/to/dreamt/2.1.0",
        ... )
        >>> task = SleepWakeTask(signal_subset="ALL")
        >>> sample_ds = ds.set_task(task)
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
    ) -> None:
        valid = set(SIGNAL_GROUPS.keys())
        if signal_subset not in valid:
            raise ValueError(
                f"signal_subset must be one of {sorted(valid)}"
                f", got '{signal_subset}'"
            )
        self.signal_subset = signal_subset
        self.artifact_threshold = artifact_threshold
        super().__init__()

    def __call__(
        self, patient: Any,
    ) -> List[Dict[str, Any]]:
        """Process one DREAMT patient into epoch samples.

        Args:
            patient: A ``Patient`` object from
                ``DREAMTDataset``.

        Returns:
            List of dicts each containing:

            - ``patient_id`` (str)
            - ``epoch_idx`` (int): temporal position
            - ``signal`` (np.ndarray): 1-D feature vector
            - ``label`` (int): 1 = wake, 0 = sleep
        """
        return self._process_patient(patient, BINARY_LABEL_MAP)

    def _process_patient(
        self,
        patient: Any,
        label_map: Dict[str, int],
    ) -> List[Dict[str, Any]]:
        """Shared epoching / feature extraction pipeline.

        Args:
            patient: A ``Patient`` object.
            label_map: Mapping from stage string to int label.

        Returns:
            Ordered list of sample dicts for this patient.
        """
        pid: str = patient.patient_id
        try:
            events = patient.get_events(
                event_type="dreamt_sleep",
            )
        except Exception:
            events = patient.get_events()

        if not events:
            return []

        event = events[0]
        file_path = getattr(event, "file_64hz", None)
        if file_path is None or (
            isinstance(file_path, str)
            and file_path.lower() == "none"
        ):
            return []

        try:
            df = pd.read_csv(str(file_path))
        except Exception as exc:
            logger.warning(
                "Could not read %s: %s", file_path, exc,
            )
            return []

        required_cols: set = {"Sleep_Stage"}
        if self.signal_subset == "ALL":
            for cols in SIGNAL_GROUPS.values():
                required_cols.update(cols)
        else:
            required_cols.update(
                SIGNAL_GROUPS[self.signal_subset]
            )

        col_map: Dict[str, str] = {}
        for col in df.columns:
            col_map[col.lower()] = col

        missing = [
            c for c in required_cols
            if c.lower() not in col_map
        ]
        if missing:
            logger.warning(
                "Patient %s missing columns: %s",
                pid, missing,
            )
            return []

        n_samples = len(df)
        n_epochs = n_samples // EPOCH_LEN

        has_acc = all(
            c.lower() in col_map
            for c in ["ACC_X", "ACC_Y", "ACC_Z"]
        )

        samples: List[Dict[str, Any]] = []
        epoch_counter = 0

        for i in range(n_epochs):
            start = i * EPOCH_LEN
            end = start + EPOCH_LEN
            epoch_df = df.iloc[start:end]

            stage_col = col_map.get(
                "sleep_stage", "Sleep_Stage",
            )
            stage_vals = epoch_df[stage_col].dropna().unique()
            if len(stage_vals) == 0:
                continue

            stage = str(stage_vals[0]).strip()
            if stage == "Missing" or stage not in label_map:
                continue

            if (
                self.artifact_threshold is not None
                and has_acc
            ):
                acc_idx = _compute_acc_index(epoch_df)
                if acc_idx > self.artifact_threshold:
                    logger.debug(
                        "Epoch %d patient %s dropped "
                        "(ACC_INDEX=%.4f > %.4f)",
                        i, pid, acc_idx,
                        self.artifact_threshold,
                    )
                    continue

            label = label_map[stage]

            try:
                feat = _extract_features(
                    epoch_df, self.signal_subset,
                )
            except Exception as exc:
                logger.debug(
                    "Feature extraction failed epoch %d "
                    "patient %s: %s",
                    i, pid, exc,
                )
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

    Identical to :class:`SleepWakeTask` except that the label
    is a 5-class integer (W=0, R=1, N1=2, N2=3, N3=4) instead
    of binary.

    Attributes:
        task_name: ``"SleepStageClassification"``
        input_schema: ``{"signal": "tensor"}``
        output_schema: ``{"label": "multiclass"}``

    Args:
        signal_subset: See :class:`SleepWakeTask`.
        artifact_threshold: See :class:`SleepWakeTask`.

    Examples:
        >>> from pyhealth.datasets import DREAMTDataset
        >>> ds = DREAMTDataset(
        ...     root="/path/to/dreamt/2.1.0",
        ... )
        >>> task = SleepStageTask()
        >>> sample_ds = ds.set_task(task)
        >>> sample_ds.samples[0].keys()
        dict_keys(['patient_id', 'epoch_idx', 'signal', 'label'])
    """

    task_name: str = "SleepStageClassification"
    output_schema: Dict[str, str] = {"label": "multiclass"}

    def __call__(
        self, patient: Any,
    ) -> List[Dict[str, Any]]:
        """Process one DREAMT patient into 5-class samples.

        Args:
            patient: A ``Patient`` object from
                ``DREAMTDataset``.

        Returns:
            List of dicts with the same keys as
            :meth:`SleepWakeTask.__call__`, but ``label``
            is an int in {0, 1, 2, 3, 4} corresponding
            to {W, R, N1, N2, N3}.
        """
        return self._process_patient(
            patient, STAGE_LABEL_MAP,
        )
