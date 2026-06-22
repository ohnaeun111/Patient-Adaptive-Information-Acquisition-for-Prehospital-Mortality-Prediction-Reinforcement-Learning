# mask_utils.py
import numpy as np
import pandas as pd

from preprocess import preprocess_from_raw, get_feature_matrix
from missing_rules import is_missing_group_from_call_raw


VITAL_IMPUTE_CODES = {
    "Sbp_value": 6,   # SBP
    "Dbp_value": 6,   # DBP
    "Pr_value": 5,    # PR
    "Rr_value": 5,    # RR
    "Bt_value": 7,    # BT
    "Spo2_value": 5,  # SpO2
}


def _is_blank_or_nan(v) -> bool:
    if v is None:
        return True
    if pd.isna(v):
        return True
    if isinstance(v, str) and v.strip() == "":
        return True
    return False


def _apply_vital_imputation_codes(df_call_raw: pd.DataFrame) -> pd.DataFrame:

    df = df_call_raw.copy()

    for col, code in VITAL_IMPUTE_CODES.items():
        if col not in df.columns:
            continue

        mask = df[col].apply(_is_blank_or_nan)
        if mask.any():
            df.loc[mask, col] = code

    return df


def build_X_call_onehot_from_call_raw(df_call_aligned_raw: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    df_filled = _apply_vital_imputation_codes(df_call_aligned_raw)

    proc = preprocess_from_raw(df_filled)
    X_df, _ = get_feature_matrix(proc)
    X_df = X_df.reindex(columns=feature_names, fill_value=0)
    return X_df.to_numpy(dtype=np.float32)


def build_init_mask_onehot_from_call_raw(
    df_call_aligned_raw: pd.DataFrame,
    group_names: list[str],
    group_to_indices: dict[str, list[int]],
) -> np.ndarray:

    n = len(df_call_aligned_raw)

    max_idx = 0
    for idxs in group_to_indices.values():
        if idxs:
            max_idx = max(max_idx, max(idxs))
    d = max_idx + 1

    mask = np.zeros((n, d), dtype=np.uint8)

    for i in range(n):
        row = df_call_aligned_raw.iloc[i]
        for gname in group_names:
            idxs = group_to_indices.get(gname, [])
            if not idxs:
                continue

            missing = bool(is_missing_group_from_call_raw(row, gname))
            if not missing:
                mask[i, idxs] = 1
            else:
                mask[i, idxs] = 0

    return mask
