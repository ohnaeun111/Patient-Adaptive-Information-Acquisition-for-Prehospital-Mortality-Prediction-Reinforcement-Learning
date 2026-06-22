# call_align.py
import numpy as np
import pandas as pd


def _is_blank(v) -> bool:
    return isinstance(v, str) and v.strip() == ""


def _is_nan_or_blank(v) -> bool:
    return bool(pd.isna(v) or _is_blank(v))


def align_call_to_all_by_ocs_date(
    df_all: pd.DataFrame,
    df_call: pd.DataFrame,
    ocs_col="OCS등록번호",
    visit_col="내원일시",
):
    df_all2 = df_all.copy()
    df_call2 = df_call.copy()

    df_all2[visit_col] = pd.to_datetime(df_all2[visit_col], errors="coerce")
    df_call2[visit_col] = pd.to_datetime(df_call2[visit_col], errors="coerce")

    key_all = df_all2[ocs_col].astype(str) + "|" + df_all2[visit_col].astype(str)
    key_call = df_call2[ocs_col].astype(str) + "|" + df_call2[visit_col].astype(str)

    call_map = {k: i for i, k in enumerate(key_call)}
    match_idx = np.array([call_map.get(k, -1) for k in key_all], dtype=int)

    matched = match_idx >= 0
    match_rate = float(matched.mean()) if len(matched) else 0.0

    df_call_aligned = pd.DataFrame(index=df_all2.index, columns=df_call2.columns)
    df_call_aligned.loc[matched] = df_call2.iloc[match_idx[matched]].to_numpy()
    df_call_aligned.loc[~matched] = np.nan

    df_all_aligned = df_all2.copy()

    # ✅ 핵심: Age/Gender는 "결측일 때만" 강제 질문(True)
    # (절대 one-hot/male 컬럼으로 판단하면 안 되고, raw 코드 컬럼에서 판단)
    forced_age = np.zeros(len(df_all_aligned), dtype=bool)
    forced_gender = np.zeros(len(df_all_aligned), dtype=bool)

    if "Age_code" in df_call_aligned.columns:
        forced_age = df_call_aligned["Age_code"].apply(_is_nan_or_blank).to_numpy(dtype=bool)
    if "Gender_code" in df_call_aligned.columns:
        forced_gender = df_call_aligned["Gender_code"].apply(_is_nan_or_blank).to_numpy(dtype=bool)

    return df_all_aligned, df_call_aligned, forced_age, forced_gender, match_rate
