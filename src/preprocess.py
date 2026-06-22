import numpy as np
import pandas as pd

_VITAL_RAW_COLS = [
    "Sbp_value", "Dbp_value", "Pr_value", "Rr_value", "Bt_value", "Spo2_value"
]

_MISSING_DATA_FILL = {
    # Injury time
    "Time_HHMM": 2,  # injury_time_map: missing_data=2

    # Intentionality
    "Intentionality_code": 5,  # intentionality_map: missing_data=5

    # Injury mechanism
    "Injury mechanism_code": 15,  # injury_mechanism_map: missing_data=15

    # Type of injury
    "Type of injury_code": 5,  # injury_type_map: missing_data=5

    # Job-related
    "Job-related_code": 3,  # job_map: missing_data=3

    # Accident location
    "Accident location_code": 21,  # accident_location_map: missing_data=21

    # Hospital visit route
    "Hospital visit route_code": 5,  # hospital_route_map: missing_data=5

    # Mode of arrival (Transport mode)
    "Mode of arrival_code": 9,  # transport_mode_map: missing_data=9

    # Insurance type
    "Insurance type_code": 9,  # insurance_type_map: missing_data=9

    # Pre-hospital cardiac arrest
    "Pre-hospital caradiac arrest_code": 3,  # prearrival_cardiac_map: missing_data=3

    # Pre-hospital notification (Prearrival Report)
    "Pre-hospital notification_code": 4,  # prearrival_report_map: missing_data=4

    # AVPU
    "Initial AVPU scale_code": 5,  # avpu_map: response_missing_data=5
}

_UNKNOWN_FILL = {
    "Protective_code": 2,  # protective_map: unknown=2
}

_VITAL_UNCHECKED_CODE = 0


def _coerce_numeric(s: pd.Series) -> pd.Series:
    """숫자형으로 강제 변환(문자/빈칸 -> NaN), 이후 정책에 따라 fill."""
    return pd.to_numeric(s, errors="coerce")


def _apply_missing_policy(df: pd.DataFrame) -> pd.DataFrame:
    """
    ✅ 핵심: raw CALL/ALL 데이터에서 NaN/빈칸을 코드북 정책대로 치환
    - Age/Gender는 여기서 건드리지 않음(기존 로직 유지)
    """
    df = df.copy()

    for c in _VITAL_RAW_COLS:
        if c in df.columns:
            df[c] = _coerce_numeric(df[c]).fillna(_VITAL_UNCHECKED_CODE).astype(int)

    for c, fill_code in _MISSING_DATA_FILL.items():
        if c in df.columns:
            df[c] = _coerce_numeric(df[c]).fillna(fill_code).astype(int)

    for c, fill_code in _UNKNOWN_FILL.items():
        if c in df.columns:
            df[c] = _coerce_numeric(df[c]).fillna(fill_code).astype(int)

    return df


def preprocess_from_raw(df: pd.DataFrame) -> pd.DataFrame:

    df = _apply_missing_policy(df)

    record = df[["key"]].copy()
    record["date"] = df["date"]
    record["Mortality"] = df["Mortality"]

    cols = {}

    injury_time_map = {"day": 0, "night": 1, "missing_data": 2}
    for name, code in injury_time_map.items():
        cols[f"Injury time - {name}"] = (df["Time_HHMM"] == code).astype("uint8")

    age = df["Age_code"].astype("float")
    age_norm = ((age - 1.0) / 25.0).astype("float32")
    age_norm = age_norm.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")
    cols["age"] = age_norm

    male = (df["Gender_code"] == 1).astype("float")
    male = male.fillna(0.0).astype("uint8")
    cols["male"] = male

    # Intentionality
    intentionality_map = {
        "accident": 0, "suicide": 1, "assult": 2, "others": 3, "unknown": 4, "missing_data": 5
    }
    for name, code in intentionality_map.items():
        cols[f"Intentionality - {name}"] = (df["Intentionality_code"] == code).astype("uint8")

    # Injury mechanism
    injury_mechanism_map = {
        "traffic_car": 0, "traffic_bike": 1, "traffic_motorcycle": 2, "traffic_others": 3, "traffic_unknown": 4,
        "fall": 5, "slip_down": 6, "collide": 7, "cut_firearm": 8, "machine": 9, "fire": 10,
        "drown": 11, "choking": 12, "others": 13, "unknown": 14, "missing_data": 15,
    }
    for name, code in injury_mechanism_map.items():
        cols[f"Injury mechanism - {name}"] = (df["Injury mechanism_code"] == code).astype("uint8")

    # Injury type
    injury_type_map = {"blunt": 0, "penetrating": 1, "fire": 2, "others": 3, "unknown": 4, "missing_data": 5}
    for name, code in injury_type_map.items():
        cols[f"injury_{name}"] = (df["Type of injury_code"] == code).astype("uint8")

    # Job-related
    job_map = {"yes": 0, "no": 1, "unknown": 2, "missing_data": 3}
    for name, code in job_map.items():
        cols[f"Job-related - {name}"] = (df["Job-related_code"] == code).astype("uint8")

    # Accident location
    accident_location_map = {
        0: "home", 1: "residential_area", 2: "accommodation", 3: "office_company", 4: "factory_workplace",
        5: "construction_site", 6: "school_educational", 7: "general_road", 8: "highway", 9: "medical_institution",
        10: "mountain_farm_livestock", 11: "community_facility", 12: "river_sea", 13: "transportation_facility",
        14: "bicycle_path_sidewalk", 15: "military_area", 16: "cultural_public_facility", 17: "sports_facility",
        18: "commercial_facility", 19: "others", 20: "unknown", 21: "missing_data",
    }
    for code, name in accident_location_map.items():
        cols[f"Accident Location - {name}"] = (df["Accident location_code"] == code).astype("uint8")

    # Hospital visit route
    hospital_route_map = {
        0: "direct_visit", 1: "transfer_from_external", 2: "referral_from_external",
        3: "others", 4: "unknown", 5: "missing_data",
    }
    for code, name in hospital_route_map.items():
        cols[f"Hospital visit route - {name}"] = (df["Hospital visit route_code"] == code).astype("uint8")

    # Transport mode (Mode of arrival)
    transport_mode_map = [
        (0, "ambulance_119"), (1, "ambulance_hospital"), (2, "ambulance_other"),
        (3, "police vehicle other public"), (4, "air transport"), (5, "other vehicles"),
        (6, "on foot"), (7, "others"), (8, "unknown"), (9, "missing_data"),
    ]
    for code, name in transport_mode_map:
        cols[f"Transport mode - {name}"] = (df["Mode of arrival_code"] == code).astype("uint8")

    # Insurance type
    insurance_type_map = [
        (0, "health insurance"), (1, "auto_insurance"), (2, "industrial_accident"), (3, "Private_Insurance"),
        (4, "Medical_Benefit_Type_1"), (5, "Medical_Benefit_Type_2"), (6, "General"), (7, "others"),
        (8, "unknown"), (9, "missing_data"),
    ]
    for code, name in insurance_type_map:
        cols[f"Insurance type - {name}"] = (df["Insurance type_code"] == code).astype("uint8")

    # Prearrival cardiac arrest
    prearrival_cardiac_map = [(0, "Y"), (1, "N"), (2, "unknown"), (3, "missing_data")]
    for code, name in prearrival_cardiac_map:
        cols[f"Prearrival cardiac arrest - {name}"] = (df["Pre-hospital caradiac arrest_code"] == code).astype("uint8")

    # Prearrival report
    prearrival_report_map = [(0, "Y"), (1, "N"), (2, "not applicable"), (3, "unknown"), (4, "missing_data")]
    for code, name in prearrival_report_map:
        cols[f"Prearrival Report - {name}"] = (df["Pre-hospital notification_code"] == code).astype("uint8")

    # AVPU
    avpu_map = [
        (0, "alert"), (1, "drowsy"), (2, "semicoma"), (3, "coma"),
        (4, "not_measured_response"), (5, "response_missing_data")
    ]
    for code, name in avpu_map:
        cols[f"AVPU - {name}"] = (df["Initial AVPU scale_code"] == code).astype("uint8")

    # Vitals (binned)
    vital_maps = {
        "Sbp_value":  {0: "unchecked", 1: "uncheckable", 2: "0_mmHg", 3: "1_49", 4: "50_75", 5: "76_89", 6: "over_89"},
        "Dbp_value":  {0: "unchecked", 1: "uncheckable", 2: "0_mmHg", 3: "1_29", 4: "30_45", 5: "46_59", 6: "over_59"},
        "Pr_value":   {0: "unchecked", 1: "uncheckable", 2: "0", 3: "1_29", 4: "30_59", 5: "60_100", 6: "101_119", 7: "over_119"},
        "Rr_value":   {0: "unchecked", 1: "uncheckable", 2: "0", 3: "1_5", 4: "6_9", 5: "10_29", 6: "over_29"},
        "Bt_value":   {0: "unchecked", 1: "uncheckable", 2: "0", 3: "0_24", 4: "24_28", 5: "28_32", 6: "32_35", 7: "35_37_8", 8: "over_37_8"},
        "Spo2_value": {0: "unchecked", 1: "uncheckable", 2: "0", 3: "1_80", 4: "81_90", 5: "91_95", 6: "over_95"},
    }

    uncheckable_cols = []
    for col, mapping in vital_maps.items():
        for code, name in mapping.items():
            new_col = f"{col}_{name}"
            cols[new_col] = (df[col] == code).astype("uint8")
            if name == "uncheckable":
                uncheckable_cols.append(new_col)

    # total_uncheckable
    tmp_df = pd.DataFrame(cols)
    cols["total_uncheckable"] = tmp_df[uncheckable_cols].any(axis=1).astype("uint8")

    protective_map = {"use": 0, "not_use": 1, "unknown": 2}
    for name, code in protective_map.items():
        cols[f"Protective - {name}"] = (df["Protective_code"] == code).astype("uint8")

    record = pd.concat([record, pd.DataFrame(cols)], axis=1)
    return record


def get_feature_matrix(processed: pd.DataFrame):
    y = processed["Mortality"].astype(int).to_numpy()
    X = processed.drop(columns=["key", "date", "Mortality"])
    return X, y
