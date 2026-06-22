from __future__ import annotations
from typing import Dict, List, Tuple

GROUPS_21 = [
    "Age",
    "Gender",
    "Intentionality",
    "Injury mechanism",
    "Injury type",
    "Injury time",
    "Protective",
    "Job-related",
    "AVPU",
    "SBP",
    "DBP",
    "PR",
    "RR",
    "BT",
    "SpO2",
    "Accident location",
    "Hospital visit route",
    "Transport mode",
    "Insurance type",
    "Prearrival cardiac arrest",
    "Prearrival Report",
]

def build_group_to_indices(feature_names: List[str]) -> Tuple[List[str], Dict[str, List[int]]]:
    name_to_idx = {c: i for i, c in enumerate(feature_names)}
    g2i: Dict[str, List[int]] = {g: [] for g in GROUPS_21}

    def add_prefix(prefix: str, group: str):
        for c, i in name_to_idx.items():
            if c.startswith(prefix):
                g2i[group].append(i)

    # Age / Gender
    if "age" in name_to_idx:
        g2i["Age"].append(name_to_idx["age"])
    if "male" in name_to_idx:
        g2i["Gender"].append(name_to_idx["male"])

    # Categorical one-hot prefixes (preprocess.py와 정확히 일치해야 함)
    add_prefix("Injury time - ", "Injury time")
    add_prefix("Intentionality - ", "Intentionality")
    add_prefix("Injury mechanism - ", "Injury mechanism")
    add_prefix("Protective - ", "Protective")              # ✅ FIX
    add_prefix("Job-related - ", "Job-related")
    add_prefix("Accident Location - ", "Accident location")
    add_prefix("Hospital visit route - ", "Hospital visit route")
    add_prefix("Transport mode - ", "Transport mode")
    add_prefix("Insurance type - ", "Insurance type")
    add_prefix("Prearrival cardiac arrest - ", "Prearrival cardiac arrest")
    add_prefix("Prearrival Report - ", "Prearrival Report")

    # Injury type: preprocess에서는 injury_{name}
    for c, i in name_to_idx.items():
        if c.startswith("injury_"):
            g2i["Injury type"].append(i)

    # AVPU: preprocess에서는 "AVPU - alert" 형태
    add_prefix("AVPU - ", "AVPU")

    # Vitals bins
    for c, i in name_to_idx.items():
        if c.startswith("Sbp_value_"):
            g2i["SBP"].append(i)
        elif c.startswith("Dbp_value_"):
            g2i["DBP"].append(i)
        elif c.startswith("Pr_value_"):
            g2i["PR"].append(i)
        elif c.startswith("Rr_value_"):
            g2i["RR"].append(i)
        elif c.startswith("Bt_value_"):
            g2i["BT"].append(i)
        elif c.startswith("Spo2_value_"):
            g2i["SpO2"].append(i)

    return GROUPS_21, g2i
