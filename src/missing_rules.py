# missing_rules.py
from __future__ import annotations

import numpy as np
import pandas as pd

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

VITAL_GROUPS = {"SBP", "DBP", "PR", "RR", "BT", "SpO2"}

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
    if _is_blank_or_nan(v):
        return None
    try:
        fv = float(v)
        if np.isnan(fv):
            return None
        return int(fv)
    except Exception:
        return None


def is_missing_group_from_all_raw(row: pd.Series, group_name: str) -> bool:
    cols = GROUP_COLUMNS.get(group_name, [])
    if not cols:
        return False

    if group_name in VITAL_GROUPS:
        for c in cols:
            if c not in row.index:
                continue
            if _is_blank_or_nan(row[c]):
                return True
        return False

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
    cols = GROUP_COLUMNS.get(group_name, [])
    if not cols:
        return False

    if group_name in VITAL_GROUPS:
        for c in cols:
            if c not in row.index:
                continue
            if _is_blank_or_nan(row[c]):
                return True
        return False

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
