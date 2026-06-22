import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sb3_contrib import MaskablePPO
try:
    from sb3_contrib.common.maskable.wrappers import ActionMasker
    from sb3_contrib.common.maskable.utils import get_action_masks
except Exception:
    from sb3_contrib.common.wrappers import ActionMasker
    from sb3_contrib.common.maskable.utils import get_action_masks

from config import Paths, RLConfig
from oracle import FrozenOracle
from preprocess import preprocess_from_raw, get_feature_matrix
from call_align import align_call_to_all_by_ocs_date
from group_map import build_group_to_indices
from missing_rules import (
    is_missing_group_from_call_raw,
    is_missing_group_from_all_raw,
)
from mask_utils import (
    build_X_call_onehot_from_call_raw,
    build_init_mask_onehot_from_call_raw,
)
from env_feature_acq import FeatureAcquisitionEnv


def mask_fn(env: FeatureAcquisitionEnv):
    return env._action_mask()


def safe_join(xs):
    if xs is None:
        return ""
    if isinstance(xs, (list, tuple, np.ndarray)):
        return "|".join(map(str, xs))
    return str(xs)


def run_eval_and_export(
    split_name,
    idx,
    X_full,
    X_call,
    y,
    ocs,
    visit,
    init_mask,
    missing_call_mask,
    missing_all_mask,
    force_mask,
    group_names,
    group_to_indices,
    oracle,
    rl_model,
    model_name,
    P,
    C,
):
    Xf, Xc = X_full[idx], X_call[idx]
    y_split = y[idx]

    base_env = FeatureAcquisitionEnv(
        X_full=Xf,
        y=y_split,
        init_mask_onehot=init_mask[idx],
        X_init_known_onehot=Xc,
        oracle_predict_fn=oracle.predict_proba,
        group_names=group_names,
        group_to_indices=group_to_indices,
        missing_group_mask=missing_call_mask[idx],
        forced_group_mask=force_mask[idx],
        min_questions=C.MIN_QUESTIONS,
        max_questions=C.MAX_QUESTIONS,
        question_cost=C.QUESTION_COST,
        shap_warmstart_steps=C.SHAP_WARMSTART_STEPS,
        allow_redundant_questions=C.ALLOW_REDUNDANT_QUESTIONS,
        seed=123,
        sanity_check_groups=True,
    )
    env = ActionMasker(base_env, mask_fn)

    n = len(idx)
    p_call = oracle.predict_proba(Xc)
    p_all = oracle.predict_proba(Xf)
    p_rl = np.zeros(n)
    q_cnt = np.zeros(n, int)

    asked_groups = [""] * n
    warmstart_groups = [""] * n
    call_missing_groups = [""] * n
    all_missing_groups = [""] * n
    all_missing_count = np.zeros(n, int)

    G = len(group_names)

    for i, gi in enumerate(idx):
        obs, _ = env.reset(options={"idx": i})

        call_missing_groups[i] = safe_join(
            [group_names[g] for g in range(G) if missing_call_mask[gi, g]]
        )

        all_miss = [group_names[g] for g in range(G) if missing_all_mask[gi, g]]
        all_missing_groups[i] = safe_join(all_miss)
        all_missing_count[i] = len(all_miss)

        done = False
        while not done:
            action_masks = get_action_masks(env)
            action, _ = rl_model.predict(obs, deterministic=True, action_masks=action_masks)
            obs, _, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated

        p_rl[i] = env.unwrapped._predict_p()
        q_cnt[i] = env.unwrapped.questions_asked
        asked_groups[i] = safe_join(env.unwrapped.asked_group_names)
        warmstart_groups[i] = safe_join(env.unwrapped.warmstart_group_names)

    out = pd.DataFrame({
        "OCS등록번호": ocs[idx],
        "내원일시": visit[idx],
        "y_true": y_split,

        "call_missing_var_count(21grp)": missing_call_mask[idx].sum(axis=1),
        "all_missing_var_count(21grp)": all_missing_count,

        "questions_asked(21grp)": q_cnt,

        "p_call": p_call,
        "p_rl": p_rl,
        "p_all": p_all,

        "missing_groups_call(21grp)": call_missing_groups,
        "missing_groups_all(21grp)": all_missing_groups,

        "asked_groups(21grp)": asked_groups,
        "warmstart_groups(21grp)": warmstart_groups,
    })

    from datetime import datetime
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = P.OUT / f"compare_call_rl_all_{split_name}_{model_name}_{stamp}.xlsx"
    out.to_excel(save_path, index=False)

    print(f"[OK] Saved {split_name}: {save_path}")


def main():
    P = Paths()
    C = RLConfig()
    P.OUT.mkdir(parents=True, exist_ok=True)

    df_all_raw = pd.read_excel(P.ALL_XLSX)
    df_call_raw = pd.read_excel(P.CALL_XLSX)

    df_all, df_call, forced_age, forced_gender, match_rate = align_call_to_all_by_ocs_date(
        df_all_raw, df_call_raw, "OCS등록번호", "내원일시"
    )
    print(f"[ALIGN] match_rate={match_rate:.3f}")

    df_all = df_all.reset_index(drop=True)
    df_call = df_call.reset_index(drop=True)

    df_all_proc = preprocess_from_raw(df_all)
    X_df, y = get_feature_matrix(df_all_proc)

    X_full = X_df.to_numpy(np.float32)
    y = y.astype(int)
    feature_names = list(X_df.columns)

    ocs = df_all["OCS등록번호"].to_numpy()
    visit = pd.to_datetime(df_all["내원일시"], errors="coerce").to_numpy()

    group_names, group_to_indices = build_group_to_indices(feature_names)
    G = len(group_names)
    g_age = group_names.index("Age")
    g_gender = group_names.index("Gender")

    X_call = build_X_call_onehot_from_call_raw(df_call, feature_names)
    init_mask = build_init_mask_onehot_from_call_raw(df_call, group_names, group_to_indices)

    missing_call_mask = np.zeros((len(df_call), G), dtype=bool)
    for i, row in df_call.iterrows():
        for g, name in enumerate(group_names):
            missing_call_mask[i, g] = is_missing_group_from_call_raw(row, name)

    missing_all_mask = np.zeros((len(df_all), G), dtype=bool)
    for i, row in df_all.iterrows():
        for g, name in enumerate(group_names):
            missing_all_mask[i, g] = is_missing_group_from_all_raw(row, name)

    force_mask = np.zeros_like(missing_call_mask)
    force_mask[:, g_age] = forced_age
    force_mask[:, g_gender] = forced_gender

    idx_tr, idx_tmp = train_test_split(
        np.arange(len(y)), test_size=0.30, stratify=y, random_state=C.RANDOM_SEED
    )
    idx_va, idx_te = train_test_split(
        idx_tmp, test_size=0.50, stratify=y[idx_tmp], random_state=C.RANDOM_SEED
    )

    oracle = FrozenOracle(str(P.MODEL_PKL))
    model_path = P.OUT / "best_maskableppo_feature_acq_0121_default_a0.5_b1.0.zip"
    model_name = model_path.stem   # 👉 zip 제거된 순수 모델 이름

    rl_model = MaskablePPO.load(str(model_path))

    run_eval_and_export(
        "valid",
        idx_va,
        X_full, X_call, y,
        ocs, visit,
        init_mask,
        missing_call_mask,
        missing_all_mask,
        force_mask,
        group_names,
        group_to_indices,
        oracle,
        rl_model,
        model_name,
        P, C,
    )

    run_eval_and_export(
        "test",
        idx_te,
        X_full, X_call, y,
        ocs, visit,
        init_mask,
        missing_call_mask,
        missing_all_mask,
        force_mask,
        group_names,
        group_to_indices,
        oracle,
        rl_model,
        model_name,
        P, C,
    )


if __name__ == "__main__":
    main()
