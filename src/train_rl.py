# train_rl.py
import numpy as np
import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from sb3_contrib import MaskablePPO
try:
    from sb3_contrib.common.maskable.wrappers import ActionMasker
except Exception:
    from sb3_contrib.common.wrappers import ActionMasker

from config import Paths, RLConfig
from oracle import FrozenOracle
from preprocess import preprocess_from_raw, get_feature_matrix
from call_align import align_call_to_all_by_ocs_date
from group_map import build_group_to_indices
from missing_rules import is_missing_group_from_call_raw
from mask_utils import (
    build_X_call_onehot_from_call_raw,
    build_init_mask_onehot_from_call_raw,
)
from env_feature_acq import FeatureAcquisitionEnv
from callbacks import ValidBestModelCallback

def set_global_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def mask_fn(env: FeatureAcquisitionEnv):
    return env._action_mask()


def main():
    # set_global_seed(SEED)

    P = Paths()
    C = RLConfig()
    set_global_seed(C.RANDOM_SEED)
    P.OUT.mkdir(parents=True, exist_ok=True)

    df_all_raw = pd.read_excel(P.ALL_XLSX)
    df_call_raw = pd.read_excel(P.CALL_XLSX)

    # align + forced flags (Age/Gender는 결측일 때만 True)
    df_all_aligned, df_call_aligned, forced_age, forced_gender, match_rate = align_call_to_all_by_ocs_date(
        df_all_raw, df_call_raw, ocs_col="OCS등록번호", visit_col="내원일시"
    )
    print(f"[ALIGN] match_rate={match_rate:.3f}")

    # ALL -> X_full
    df_all_proc = preprocess_from_raw(df_all_aligned)
    X_df, y = get_feature_matrix(df_all_proc)
    X_full = X_df.to_numpy(dtype=np.float32)
    y = y.astype(int)
    feature_names = list(X_df.columns)

    # group mapping
    group_names, group_to_indices = build_group_to_indices(feature_names)
    G = len(group_names)

    # CALL one-hot (p_call 입력용)
    X_call_onehot = build_X_call_onehot_from_call_raw(df_call_aligned, feature_names)

    # ✅ init_mask는 raw missing_rules(표 기준)로 생성
    init_mask_onehot = build_init_mask_onehot_from_call_raw(
        df_call_aligned_raw=df_call_aligned,
        group_names=group_names,
        group_to_indices=group_to_indices,
    )

    # missing_group_mask & forced_group_mask
    missing_group_mask = np.zeros((len(df_call_aligned), G), dtype=bool)
    forced_group_mask = np.zeros((len(df_call_aligned), G), dtype=bool)

    g_age = group_names.index("Age")
    g_gender = group_names.index("Gender")

    forced_group_mask[:, g_age] = forced_age
    forced_group_mask[:, g_gender] = forced_gender

    for i in range(len(df_call_aligned)):
        row = df_call_aligned.iloc[i]
        for gi, gname in enumerate(group_names):
            missing_group_mask[i, gi] = bool(is_missing_group_from_call_raw(row, gname))

    # split
    idx_all = np.arange(len(y))
    idx_tr, idx_tmp, y_tr, y_tmp = train_test_split(
        idx_all, y, test_size=0.30, stratify=y, random_state=C.RANDOM_SEED
    )
    idx_va, idx_te, y_va, y_te = train_test_split(
        idx_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=C.RANDOM_SEED
    )

    def slice_all(idx):
        return (
            X_full[idx],
            y[idx],
            init_mask_onehot[idx],
            X_call_onehot[idx],
            missing_group_mask[idx],
            forced_group_mask[idx],
        )

    X_tr, y_tr2, m_tr, Xc_tr, miss_tr, force_tr = slice_all(idx_tr)
    X_va, y_va2, m_va, Xc_va, miss_va, force_va = slice_all(idx_va)

    # oracle
    oracle = FrozenOracle(str(P.MODEL_PKL))

    def oracle_fn(x_np: np.ndarray):
        return oracle.predict_proba(x_np)

    # -----------------------------
    # ✅ Episode oversampling: TRAIN ONLY
    # -----------------------------
    use_ep_os_train = bool(getattr(C, "USE_EPISODE_OVERSAMPLING", False))
    ep_pos_prob = float(getattr(C, "EPISODE_POS_PROB", 0.5))

    if use_ep_os_train:
        print(f"[EP-OVERSAMPLE] TRAIN ONLY enabled | EPISODE_POS_PROB={ep_pos_prob:.3f}")
    else:
        print("[EP-OVERSAMPLE] disabled")

    # envs
    train_base = FeatureAcquisitionEnv(
        X_full=X_tr,
        y=y_tr2,
        init_mask_onehot=m_tr,
        X_init_known_onehot=Xc_tr,
        oracle_predict_fn=oracle_fn,
        group_names=group_names,
        group_to_indices=group_to_indices,
        missing_group_mask=miss_tr,
        forced_group_mask=force_tr,
        shap_rank_group_indices=list(range(G)),
        min_questions=C.MIN_QUESTIONS,
        max_questions=C.MAX_QUESTIONS,
        question_cost=C.QUESTION_COST,
        shap_warmstart_steps=C.SHAP_WARMSTART_STEPS,
        seed=C.RANDOM_SEED,
        alpha_loss=C.ALPHA_LOSS,
        beta_prob=C.BETA_PROB,
        stop_penalty=C.STOP_PENALTY,
        stop_reward_mode=getattr(C, "STOP_REWARD_MODE", "loss"),
        stop_reward_scale=float(getattr(C, "STOP_REWARD_SCALE", 1.0)),
        allow_redundant_questions=bool(getattr(C, "ALLOW_REDUNDANT_QUESTIONS", False)),

        # ✅ TRAIN에만 적용
        use_episode_oversampling=use_ep_os_train,
        episode_pos_prob=ep_pos_prob,

        sanity_check_groups=True,
    )
    train_env = ActionMasker(train_base, mask_fn)
    train_base.seed(C.RANDOM_SEED)

    valid_base = FeatureAcquisitionEnv(
        X_full=X_va,
        y=y_va2,
        init_mask_onehot=m_va,
        X_init_known_onehot=Xc_va,
        oracle_predict_fn=oracle_fn,
        group_names=group_names,
        group_to_indices=group_to_indices,
        missing_group_mask=miss_va,
        forced_group_mask=force_va,
        shap_rank_group_indices=list(range(G)),
        min_questions=C.MIN_QUESTIONS,
        max_questions=C.MAX_QUESTIONS,
        question_cost=C.QUESTION_COST,
        shap_warmstart_steps=C.SHAP_WARMSTART_STEPS,
        seed=123,
        alpha_loss=C.ALPHA_LOSS,
        beta_prob=C.BETA_PROB,
        stop_penalty=C.STOP_PENALTY,
        stop_reward_mode=getattr(C, "STOP_REWARD_MODE", "loss"),
        stop_reward_scale=float(getattr(C, "STOP_REWARD_SCALE", 1.0)),
        allow_redundant_questions=bool(getattr(C, "ALLOW_REDUNDANT_QUESTIONS", False)),

        # ✅ VALID는 항상 OFF (원본 분포 유지)
        use_episode_oversampling=False,
        episode_pos_prob=0.0,

        sanity_check_groups=True,
    )
    valid_env = ActionMasker(valid_base, mask_fn)
    valid_base.seed(C.RANDOM_SEED)

    model = MaskablePPO(
        "MlpPolicy",
        train_env,
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=1,
        tensorboard_log=str(P.OUT / "tb"),
        learning_rate=C.LEARNING_RATE,
        n_steps=C.N_STEPS,
        batch_size=C.BATCH_SIZE,
        gamma=C.GAMMA,
        seed=C.RANDOM_SEED,
    )

    best_path = P.OUT / "best_maskableppo_feature_acq_0121_default_a0.5_b1.0.zip" # model 이름 저장 하는 거
    cb = ValidBestModelCallback(
        valid_env=valid_env,
        oracle_predict_fn=oracle_fn,
        save_path=best_path,
        eval_freq=C.EVAL_FREQ,
        patience=C.EARLYSTOP_PATIENCE,
        min_delta=C.EARLYSTOP_MIN_DELTA,
        verbose=1,
    )

    model.learn(total_timesteps=C.TOTAL_TIMESTEPS, callback=cb)

    last_path = P.OUT / "last_maskableppo_feature_acq_0121_default_a0.5_b1.0.zip"
    model.save(str(last_path))
    print(f"[OK] Saved last model -> {last_path}")


if __name__ == "__main__":
    main()
