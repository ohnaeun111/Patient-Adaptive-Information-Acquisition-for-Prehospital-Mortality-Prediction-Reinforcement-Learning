# missing_rules.py
from __future__ import annotations

import numpy as np
import pandas as pd


# =========================================================
# ✅ group_map.GROUPS_21 와 1:1로 정확히 맞춘 키들
# =========================================================
GROUP_COLUMNS: dict[str, list[str]] = {
    # demographics
    "Age": ["Age_code"],
    "Gender": ["Gender_code"],

    # injury details
    "Intentionality": ["Intentionality_code"],
    "Injury mechanism": ["Injury mechanism_code"],
    "Injury type": ["Type of injury_code"],
    "Injury time": ["Time_HHMM"],
    "Protective": ["Protective_code"],
    "Job-related": ["Job-related_code"],
    "AVPU": ["Initial AVPU scale_code"],

    # vitals (binned codes)
    "SBP": ["Sbp_value"],
    "DBP": ["Dbp_value"],
    "PR": ["Pr_value"],
    "RR": ["Rr_value"],
    "BT": ["Bt_value"],
    "SpO2": ["Spo2_value"],

    # arrival / transport
    "Accident location": ["Accident location_code"],
    "Hospital visit route": ["Hospital visit route_code"],
    "Transport mode": ["Mode of arrival_code"],
    "Insurance type": ["Insurance type_code"],
    "Prearrival cardiac arrest": ["Pre-hospital caradiac arrest_code"],
    "Prearrival Report": ["Pre-hospital notification_code"],
}

# =========================================================
# ✅ Vital sign은 "raw NaN/빈칸"만 missing으로 본다.
#    (unchecked code==0 등은 더 이상 missing 판단에 사용하지 않음)
# =========================================================
VITAL_GROUPS = {"SBP", "DBP", "PR", "RR", "BT", "SpO2"}

# =========================================================
# ✅ Vital 제외 항목은 "코드북 missing data code"를 missing으로 본다.
#    + raw NaN/빈칸도 missing으로 본다.
#  (21개 변수_v1.docx 테이블 기반: "missing data"로 명시된 값들만)
# =========================================================
GROUP_MISSING_CODES: dict[str, set[int]] = {
    "Intentionality": {5},
    "Injury mechanism": {15},
    "Injury type": {5},
    "Injury time": {2},
    "Job-related": {3},
    "AVPU": {5},
    "Accident location": {21},
    "Hospital visit route": {5},
    "Transport mode": {9},
    "Insurance type": {9},
    "Prearrival cardiac arrest": {3},
    "Prearrival Report": {4},
    "Protective": {2}
    # Protective는 docx 상 "missing data" 코드가 별도로 명시되지 않아
    # raw NaN/빈칸만 missing으로 처리함.
}


def _is_blank_or_nan(v) -> bool:
    if v is None:
        return True
    # pandas NaN
    if isinstance(v, float) and np.isnan(v):
        return True
    if pd.isna(v):
        return True
    # empty string
    if isinstance(v, str) and v.strip() == "":
        return True
    return False


def _to_int_if_possible(v):
    """
    엑셀에서 숫자가 float로 읽히는 경우(9.0) 등도 안전하게 int로 변환.
    변환 불가하면 None 반환.
    """
    if _is_blank_or_nan(v):
        return None
    try:
        # "9", "9.0", 9.0 등 처리
        fv = float(v)
        if np.isnan(fv):
            return None
        return int(fv)
    except Exception:
        return None


def is_missing_group_from_all_raw(row: pd.Series, group_name: str) -> bool:
    """
    ✅ 최종 합의 정책:
    - Vital sign 그룹(SBP/DBP/PR/RR/BT/SpO2):
        raw NaN/빈칸일 때만 missing=True
        (unchecked code==0 기반 missing 판정은 삭제)
    - Vital 제외 그룹:
        raw NaN/빈칸이면 missing=True
        또는 코드북에서 "missing data"로 지정된 code이면 missing=True
    """
    cols = GROUP_COLUMNS.get(group_name, [])
    if not cols:
        # 정의되지 않은 그룹은 missing으로 보지 않음(환경 sanity_check로 잡히는 편이 안전)
        return False

    # 1) Vital sign: raw NaN/빈칸만 missing
    if group_name in VITAL_GROUPS:
        for c in cols:
            if c not in row.index:
                continue
            if _is_blank_or_nan(row[c]):
                return True
        return False

    # 2) Non-vital: raw NaN/빈칸 OR (missing code)
    miss_codes = GROUP_MISSING_CODES.get(group_name, set())
    for c in cols:
        if c not in row.index:
            continue

        v = row[c]
        if _is_blank_or_nan(v):
            return True

        iv = _to_int_if_possible(v)
        if iv is not None and iv in miss_codes:
            return True

    return False

def is_missing_group_from_call_raw(row: pd.Series, group_name: str) -> bool:
    """
    ✅ 최종 합의 정책:
    - Vital sign 그룹(SBP/DBP/PR/RR/BT/SpO2):
        raw NaN/빈칸일 때만 missing=True
        (unchecked code==0 기반 missing 판정은 삭제)
    - Vital 제외 그룹:
        raw NaN/빈칸이면 missing=True
        또는 코드북에서 "missing data"로 지정된 code이면 missing=True
    """
    cols = GROUP_COLUMNS.get(group_name, [])
    if not cols:
        # 정의되지 않은 그룹은 missing으로 보지 않음(환경 sanity_check로 잡히는 편이 안전)
        return False

    # 1) Vital sign: raw NaN/빈칸만 missing
    if group_name in VITAL_GROUPS:
        for c in cols:
            if c not in row.index:
                continue
            if _is_blank_or_nan(row[c]):
                return True
        return False

    # 2) Non-vital: raw NaN/빈칸 OR (missing code)
    miss_codes = GROUP_MISSING_CODES.get(group_name, set())
    for c in cols:
        if c not in row.index:
            continue

        v = row[c]
        if _is_blank_or_nan(v):
            return True

        iv = _to_int_if_possible(v)
        if iv is not None and iv in miss_codes:
            return True

    return False
