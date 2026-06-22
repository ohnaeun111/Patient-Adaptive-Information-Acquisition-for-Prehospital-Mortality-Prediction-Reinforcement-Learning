# export_death_pcall_pall_full.py
import numpy as np
import pandas as pd

from config import Paths
from oracle import FrozenOracle
from preprocess import preprocess_from_raw, get_feature_matrix
from call_align import align_call_to_all_by_ocs_date
from group_map import build_group_to_indices
from missing_rules import is_missing_group_from_call_raw
from mask_utils import build_X_call_onehot_from_call_raw


def safe_join(xs):
    if xs is None:
        return ""
    if isinstance(xs, (list, tuple, np.ndarray)):
        return "|".join([str(x) for x in xs])
    return str(xs)


def main():
    P = Paths()
    P.OUT.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # 1) Load raw ALL / CALL
    # -----------------------------
    df_all_raw = pd.read_excel(P.ALL_XLSX)
    df_call_raw = pd.read_excel(P.CALL_XLSX)

    # -----------------------------
    # 2) Align (1:1 by OCS + date)
    # -----------------------------
    df_all_aligned, df_call_aligned, forced_age, forced_gender, match_rate = align_call_to_all_by_ocs_date(
        df_all_raw, df_call_raw, ocs_col="OCS등록번호", visit_col="내원일시"
    )
    print(f"[ALIGN] match_rate={match_rate:.3f}")

    # -----------------------------
    # 3) Build ALL feature matrix (X_all) + y
    # -----------------------------
    df_all_proc = preprocess_from_raw(df_all_aligned)
    X_df, y = get_feature_matrix(df_all_proc)

    X_all = X_df.to_numpy(dtype=np.float32)
    y = y.astype(int)
    feature_names = list(X_df.columns)

    # metadata columns
    ocs = df_all_aligned["OCS등록번호"].to_numpy()
    visit = pd.to_datetime(df_all_aligned["내원일시"], errors="coerce").to_numpy()

    # -----------------------------
    # 4) Build CALL one-hot for p_call input
    #    (missing 채우는 방식은 build_X_call... 내부 로직 그대로 사용)
    # -----------------------------
    X_call = build_X_call_onehot_from_call_raw(df_call_aligned, feature_names).astype(np.float32)

    # -----------------------------
    # 5) Group mapping + CALL raw missing groups (21grp)
    # -----------------------------
    group_names, group_to_indices = build_group_to_indices(feature_names)
    G = len(group_names)

    missing_groups = [""] * len(df_call_aligned)
    call_missing_var_count = np.zeros(len(df_call_aligned), dtype=int)

    for i in range(len(df_call_aligned)):
        row = df_call_aligned.iloc[i]
        miss_list = []
        for gi, gname in enumerate(group_names):
            if bool(is_missing_group_from_call_raw(row, gname)):
                miss_list.append(gname)
        missing_groups[i] = safe_join(miss_list)
        call_missing_var_count[i] = len(miss_list)

    # -----------------------------
    # 6) Oracle probabilities: p_call, p_all
    # -----------------------------
    oracle = FrozenOracle(str(P.MODEL_PKL))
    p_call = oracle.predict_proba(X_call).astype(np.float32)
    p_all = oracle.predict_proba(X_all).astype(np.float32)

    pred_call = (p_call >= 0.5).astype(int)
    pred_all = (p_all >= 0.5).astype(int)

    correct_call = (pred_call == y).astype(int)
    correct_all = (pred_all == y).astype(int)

    # -----------------------------
    # 7) Filter: only deaths (Mortality=1)
    # -----------------------------
    death_mask = (y == 1)
    n_death = int(death_mask.sum())
    print(f"[INFO] total={len(y)}, deaths={n_death}")

    out = pd.DataFrame({
        "OCS등록번호": ocs[death_mask],
        "내원일시": visit[death_mask],
        "y_true": y[death_mask],

        # CALL missing info (21grp)
        "call_missing_var_count(21grp)": call_missing_var_count[death_mask],
        "missing_groups(21grp)": np.array(missing_groups, dtype=object)[death_mask],

        # probabilities / predictions
        "p_call": p_call[death_mask],
        "p_all": p_all[death_mask],
        "pred_call": pred_call[death_mask],
        "pred_all": pred_all[death_mask],
        "correct_call": correct_call[death_mask],
        "correct_all": correct_all[death_mask],

        # optional deltas
        "delta_all_call": (p_all - p_call)[death_mask],
    })

    # -----------------------------
    # 8) Save
    # -----------------------------
    from datetime import datetime
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = P.OUT / f"death_only_pcall_pall_full_{stamp}.xlsx"
    out.to_excel(save_path, index=False)

    print(f"[OK] Saved: {save_path}")

    # -----------------------------
    # 9) Quick summary
    # -----------------------------
    print("[SUMMARY - DEATH ONLY]")
    print("n_death:", n_death)
    print("acc_call:", float(out["correct_call"].mean()) if n_death > 0 else np.nan)
    print("acc_all :", float(out["correct_all"].mean()) if n_death > 0 else np.nan)
    print("mean_call_missing(21grp):", float(out["call_missing_var_count(21grp)"].mean()) if n_death > 0 else np.nan)


if __name__ == "__main__":
    main()
