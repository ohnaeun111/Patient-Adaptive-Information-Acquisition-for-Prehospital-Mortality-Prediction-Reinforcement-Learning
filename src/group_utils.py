import numpy as np
import pandas as pd


def build_groups_from_onehot_columns(onehot_cols):
    """
    onehot column naming convention에서 21 그룹을 자동 구성.
    예: "Insurance type - missing_data" -> 그룹 "Insurance type"
        "Sbp_value_over_89" -> 그룹 "SBP" (원하면 규칙 보정 가능)
        "male" -> 그룹 "Gender"
        "age" -> 그룹 "Age"
    """

    group_names = []
    group_to_indices = {}

    def add(g, idx):
        if g not in group_to_indices:
            group_to_indices[g] = []
            group_names.append(g)
        group_to_indices[g].append(idx)

    for j, c in enumerate(onehot_cols):
        if c == "age":
            add("Age", j)
        elif c == "male":
            add("Gender", j)
        elif "Insurance type" in c:
            add("Insurance type", j)
        elif c.startswith("Sbp_value"):
            add("SBP", j)
        elif c.startswith("Dbp_value"):
            add("DBP", j)
        elif c.startswith("Pr_value"):
            add("PR", j)
        elif c.startswith("Rr_value"):
            add("RR", j)
        elif c.startswith("Bt_value"):
            add("BT", j)
        elif c.startswith("Spo2_value") or c == "Spo2":
            add("SpO2", j)
        elif c.startswith("Hospital visit route"):
            add("Hospital visit route", j)
        elif c.startswith("Accident Location"):
            add("Accident location", j)
        elif c.startswith("Transport mode"):
            add("Transport mode", j)
        elif c.startswith("Injury time"):
            add("Injury time", j)
        elif c.startswith("Intentionality"):
            add("Intentionality", j)
        elif c.startswith("Injury mechanism"):
            add("Injury mechanism", j)
        elif c.startswith("injury_"):
            add("Injury type", j)
        elif c.startswith("Job-related"):
            add("Job-related", j)
        elif c in ["alert", "drowsy", "semicoma", "coma", "not_measured_response", "response_missing_data"]:
            add("AVPU", j)
        elif c.startswith("Prearrival cardiac arrest"):
            add("Prearrival cardiac arrest", j)
        elif c.startswith("Prearrival Report"):
            add("Prearrival Report", j)
        elif c.startswith("Protective"):
            add("Protective", j)
        else:
            # fallback: "X - Y" 패턴이면 X를 그룹으로
            if " - " in c:
                add(c.split(" - ")[0], j)
            else:
                add(c, j)

    return group_names, group_to_indices


def compute_group_missing_from_raw_call(df_call_raw: pd.DataFrame, group_names):
    """
    call raw의 code 값을 보고, 그룹별로 missing/unmeasured인지 판단.
    여기서의 판단이 '질문 가능한 대상'이 됩니다.
    """

    n = len(df_call_raw)
    G = len(group_names)
    out = np.zeros((n, G), dtype=bool)

    # 컬럼명(당신 call raw 기준)
    # Age_code, Gender_code는 NaN이면 missing으로 처리
    # Insurance type_code는 9가 missing_data
    # Vital들은 0 unchecked, 1 uncheckable 등을 missing/unmeasured로 볼지 정책 결정
    # (아래는 “측정/확인되지 않음은 질문 대상”으로 보는 보수적 기준)

    def col(name):
        return df_call_raw[name] if name in df_call_raw.columns else pd.Series([np.nan]*n)

    age = col("Age_code")
    gender = col("Gender_code")

    for g, name in enumerate(group_names):
        if name == "Age":
            out[:, g] = age.isna()
        elif name == "Gender":
            out[:, g] = gender.isna()

        elif name == "Insurance type":
            out[:, g] = (col("Insurance type_code") == 9) | col("Insurance type_code").isna()

        elif name == "Injury time":
            out[:, g] = col("Time_HHMM").isna()  # (당신 코드에서 missing_data=2지만, raw가 NaN이면 missing)
        elif name == "AVPU":
            out[:, g] = col("Initial AVPU scale_code").isna() | (col("Initial AVPU scale_code") == 5)  # response_missing_data
        elif name == "SBP":
            out[:, g] = col("Sbp_value").isna() | col("Sbp_value").isin([0, 1])  # unchecked/uncheckable
        elif name == "DBP":
            out[:, g] = col("Dbp_value").isna() | col("Dbp_value").isin([0, 1])
        elif name == "PR":
            out[:, g] = col("Pr_value").isna() | col("Pr_value").isin([0, 1])
        elif name == "RR":
            out[:, g] = col("Rr_value").isna() | col("Rr_value").isin([0, 1])
        elif name == "BT":
            out[:, g] = col("Bt_value").isna() | col("Bt_value").isin([0, 1])
        elif name == "SpO2":
            out[:, g] = col("Spo2_value").isna() | col("Spo2_value").isin([0, 1])

        # 나머지 범주형은 missing_data code가 있으면 그 값 기준으로 처리
        elif name == "Intentionality":
            out[:, g] = col("Intentionality_code").isna() | (col("Intentionality_code") == 5)
        elif name == "Injury mechanism":
            out[:, g] = col("Injury mechanism_code").isna() | (col("Injury mechanism_code") == 15)
        elif name == "Injury type":
            out[:, g] = col("Type of injury_code").isna() | (col("Type of injury_code") == 5)
        elif name == "Job-related":
            out[:, g] = col("Job-related_code").isna() | (col("Job-related_code") == 3)
        elif name == "Accident location":
            out[:, g] = col("Accident location_code").isna() | (col("Accident location_code") == 21)
        elif name == "Hospital visit route":
            out[:, g] = col("Hospital visit route_code").isna() | (col("Hospital visit route_code") == 5)
        elif name == "Transport mode":
            out[:, g] = col("Mode of arrival_code").isna() | (col("Mode of arrival_code") == 9)
        elif name == "Prearrival cardiac arrest":
            out[:, g] = col("Pre-hospital caradiac arrest_code").isna() | (col("Pre-hospital caradiac arrest_code") == 3)
        elif name == "Prearrival Report":
            out[:, g] = col("Pre-hospital notification_code").isna() | (col("Pre-hospital notification_code") == 4)
        elif name == "Protective":
            out[:, g] = col("Protective_code").isna()  # protective는 missing_data code가 없으니 NaN만
        else:
            out[:, g] = False

    return out
