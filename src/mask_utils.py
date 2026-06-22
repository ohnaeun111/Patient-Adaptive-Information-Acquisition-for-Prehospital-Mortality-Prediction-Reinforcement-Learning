# mask_utils.py
import numpy as np
import pandas as pd

from preprocess import preprocess_from_raw, get_feature_matrix
from missing_rules import is_missing_group_from_call_raw


# =========================================================
# ✅ Vital raw missing(NaN/빈칸) 시 oracle imputation code로 채우기
# (사용자 제공 고정 코드)
# =========================================================
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
    """
    ✅ 정책:
    - Vital sign은 raw NaN/빈칸이면 oracle에서 쓰던 방식과 맞추기 위해
      지정된 '대체 코드'로 CALL raw를 채운다.
    - 단, 질문 허용은 missing_rules에서 'raw missing indicator'로 따로 처리하므로
      RL은 여전히 질문 가능(= missing_group_mask True).
    """
    df = df_call_raw.copy()

    for col, code in VITAL_IMPUTE_CODES.items():
        if col not in df.columns:
            continue

        mask = df[col].apply(_is_blank_or_nan)
        if mask.any():
            df.loc[mask, col] = code

    return df


def build_X_call_onehot_from_call_raw(df_call_aligned_raw: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    """
    CALL raw(코드 컬럼들) -> preprocess 동일 파이프라인 -> one-hot 행렬 생성

    ✅ 변경점:
    - Vital raw missing(NaN/빈칸)은 '대체 코드'로 먼저 채운 뒤 one-hot 생성
      (oracle 입력과 정합성 맞추기 위함)
    """
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
    """
    ✅ init_mask는 (X!=0)로 만들면 안 됨.
    여기서는 "CALL raw 기준으로 해당 그룹이 missing인지"를 판단하여:
      - missing이면 그 그룹의 모든 컬럼 mask=0
      - missing이 아니면 그 그룹의 모든 컬럼 mask=1
    으로 그룹 단위 세팅.

    ✅ 중요한 정합성:
    - Vital raw missing은 위에서 one-hot은 대체 코드로 채우더라도,
      missing_rules에서 raw missing indicator로 missing=True가 되므로
      init_mask는 0(unknown) 상태가 유지됨 → RL이 질문 가능
    """
    n = len(df_call_aligned_raw)

    # d는 group_to_indices의 최대 인덱스로 결정
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
