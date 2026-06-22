# src/analyze_policy_importance.py
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pathlib import Path

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
from missing_rules import is_missing_group_from_call_raw
from mask_utils import build_X_call_onehot_from_call_raw, build_init_mask_onehot_from_call_raw
from env_feature_acq import FeatureAcquisitionEnv


def mask_fn(env: FeatureAcquisitionEnv):
    return env._action_mask()


def safe_join(xs):
    if xs is None:
        return ""
    if isinstance(xs, (list, tuple, np.ndarray)):
        return "|".join([str(x) for x in xs])
    return str(xs)


def _save_barplot(df_plot: pd.DataFrame, x_col: str, y_col: str, title: str, y_label: str, save_path: Path):
    plt.figure()
    plt.bar(df_plot[x_col].values, df_plot[y_col].values)
    plt.xticks(rotation=60, ha="right")
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def _save_heatmap(mat: np.ndarray, x_labels: list[str], y_labels: list[str], title: str, save_path: Path):
    """
    mat: shape (n_rows, n_cols) -> imshow
    """
    plt.figure()
    plt.imshow(mat, aspect="auto")
    plt.colorbar()
    plt.xticks(np.arange(len(x_labels)), x_labels, rotation=60, ha="right")
    plt.yticks(np.arange(len(y_labels)), y_labels)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def rollout_policy_on_split(
    split_name: str,
    rl_model: MaskablePPO,
    base_env: FeatureAcquisitionEnv,
    y_split: np.ndarray,
    group_names: list[str],
    out_dir: Path,
    max_heatmap_steps: int | None = None,
):
    """
    Train/Valid split에서 policy rollout 후:

    A) 정책 중요도 테이블:
      - asked_episode_rate (전체/클래스별)
      - never_asked_rate
      - mean_first_step
      - asked_episode_count
      - total_ask_count_split (split 전체 기준)

    B) 시각화:
      1) asked_episode_rate (all / y=0 / y=1) bar
      2) mean_first_step (all) bar
      3) Question order heatmap:
         - all / y=0 / y=1
         - (step x group) : 각 step에서 해당 group이 선택된 비율

    C) episode log 저장(선택)
    """
    env = ActionMasker(base_env, mask_fn)
    G = len(group_names)
    n = len(y_split)

    # episode-level indicators
    asked_any = np.zeros((n, G), dtype=int)       # episode에서 해당 group을 한번이라도 질문했는지
    first_step = np.full((n, G), np.nan)          # 최초 질문 step (0-based)
    total_asks = np.zeros(G, dtype=int)           # split 전체에서 group별 질문 총 횟수
    questions_asked = np.zeros(n, dtype=int)      # episode 질문 수
    asked_groups_str = [""] * n

    # question order recording (for heatmap)
    # 각 episode에서 질문 순서대로 group index 리스트를 저장
    action_seq = [[] for _ in range(n)]
    max_len = 0

    for i in range(n):
        obs, _ = env.reset(options={"idx": i})  # ✅ idx 지정 -> 해석에서 oversampling 영향 없음
        done = False
        step = 0
        asked_list = []

        while not done:
            am = get_action_masks(env)
            action, _ = rl_model.predict(obs, deterministic=True, action_masks=am)
            action = int(action)

            obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)

            if action < G:
                g = action
                asked_list.append(group_names[g])
                action_seq[i].append(g)

                if asked_any[i, g] == 0:
                    asked_any[i, g] = 1
                    first_step[i, g] = step

                total_asks[g] += 1
                step += 1  # 질문 step만 카운트(Stop 포함 step은 분석에 의미 없음)
            else:
                # STOP이면 step 증가 없이 종료
                pass

        questions_asked[i] = int(env.unwrapped.questions_asked)
        asked_groups_str[i] = safe_join(asked_list)
        max_len = max(max_len, len(action_seq[i]))

    # heatmap step 제한(너무 크면 보기 어려움)
    if max_heatmap_steps is None:
        max_heatmap_steps = int(max_len)
    max_heatmap_steps = int(min(max_heatmap_steps, max_len))

    # ---------- summary table ----------
    def summarize(sub_idx: np.ndarray, tag: str) -> pd.DataFrame:
        if len(sub_idx) == 0:
            return pd.DataFrame()

        ep_rate = asked_any[sub_idx].mean(axis=0)         # 0..1
        never_rate = 1.0 - ep_rate

        mfs = []
        for g in range(G):
            vals = first_step[sub_idx, g]
            vals = vals[~np.isnan(vals)]
            mfs.append(float(vals.mean()) if len(vals) else np.nan)

        df = pd.DataFrame({
            "split": split_name,
            "subset": tag,
            "group": group_names,
            "asked_episode_rate": ep_rate,
            "never_asked_rate": never_rate,
            "mean_first_step": mfs,
            "asked_episode_count": asked_any[sub_idx].sum(axis=0),
            "n_episodes": len(sub_idx),
        })
        return df

    idx_all = np.arange(n)
    idx_0 = np.where(y_split == 0)[0]
    idx_1 = np.where(y_split == 1)[0]

    df_all = summarize(idx_all, "all")
    df_0 = summarize(idx_0, "y=0")
    df_1 = summarize(idx_1, "y=1")
    df = pd.concat([df_all, df_0, df_1], ignore_index=True)

    total_ask_map = {group_names[g]: int(total_asks[g]) for g in range(G)}
    df["total_ask_count_split"] = df["group"].map(total_ask_map)

    # ---------- save logs ----------
    ep_log = pd.DataFrame({
        "split": split_name,
        "y": y_split,
        "questions_asked": questions_asked,
        "asked_groups": asked_groups_str,
    })
    ep_log_path = out_dir / f"policy_episode_log_{split_name}.xlsx"
    ep_log.to_excel(ep_log_path, index=False)

    df_path = out_dir / f"policy_importance_{split_name}.xlsx"
    df.to_excel(df_path, index=False)

    # ---------- plots ----------
    # (1) asked_episode_rate bar : all / y=0 / y=1
    for subset in ["all", "y=0", "y=1"]:
        df_plot = df[df["subset"] == subset].copy()
        df_plot = df_plot.sort_values("asked_episode_rate", ascending=False)
        _save_barplot(
            df_plot=df_plot,
            x_col="group",
            y_col="asked_episode_rate",
            title=f"Policy Feature Importance (Asked episode rate) - {split_name} [{subset}]",
            y_label="Asked episode rate",
            save_path=out_dir / f"policy_importance_asked_rate_{split_name}_{subset}.png",
        )

    # (2) mean_first_step bar : all 기준 (NaN은 맨 뒤로 보이게 max+1 처리)
    df_plot2 = df[df["subset"] == "all"].copy()
    df_plot2 = df_plot2.sort_values("mean_first_step", ascending=True)
    if df_plot2["mean_first_step"].notna().any():
        fill_val = float(df_plot2["mean_first_step"].max()) + 1.0
    else:
        fill_val = 0.0
    df_plot2["mean_first_step_fill"] = df_plot2["mean_first_step"].fillna(fill_val)
    _save_barplot(
        df_plot=df_plot2,
        x_col="group",
        y_col="mean_first_step_fill",
        title=f"Policy Urgency (Mean first asked step) - {split_name} [all] (NaN->max+1)",
        y_label="Mean first step (0-based)",
        save_path=out_dir / f"policy_importance_first_step_{split_name}_all.png",
    )

    # (3) Question order heatmap: step x group
    #     step t에서 group g가 선택된 비율 = count(t,g)/n_episode(subset)
    def build_step_heat(sub_idx: np.ndarray) -> np.ndarray:
        if len(sub_idx) == 0 or max_heatmap_steps == 0:
            return np.zeros((0, G), dtype=float)
        counts = np.zeros((max_heatmap_steps, G), dtype=float)
        for i in sub_idx:
            seq = action_seq[i]
            L = min(len(seq), max_heatmap_steps)
            for t in range(L):
                counts[t, seq[t]] += 1.0
        # episode 수로 나눠 비율로
        counts /= float(len(sub_idx))
        return counts

    for subset, sub_idx in [("all", idx_all), ("y=0", idx_0), ("y=1", idx_1)]:
        heat = build_step_heat(sub_idx)
        if heat.shape[0] == 0:
            continue

        x_labels = group_names
        y_labels = [f"step{t+1}" for t in range(heat.shape[0])]  # 보기 좋게 1-based 라벨
        _save_heatmap(
            mat=heat,
            x_labels=x_labels,
            y_labels=y_labels,
            title=f"Question Order Heatmap (P(ask g at step t)) - {split_name} [{subset}]",
            save_path=out_dir / f"policy_question_order_heatmap_{split_name}_{subset}.png",
        )

    print(f"[OK] {split_name}: saved\n - {df_path}\n - {ep_log_path}")
    return df


def main():
    P = Paths()
    C = RLConfig()
    P.OUT.mkdir(parents=True, exist_ok=True)

    out_dir = P.OUT / "policy_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # 1) Load raw & align
    # ----------------------------
    df_all_raw = pd.read_excel(P.ALL_XLSX)
    df_call_raw = pd.read_excel(P.CALL_XLSX)

    df_all_aligned, df_call_aligned, forced_age, forced_gender, match_rate = align_call_to_all_by_ocs_date(
        df_all_raw, df_call_raw, ocs_col="OCS등록번호", visit_col="내원일시"
    )
    print(f"[ALIGN] match_rate={match_rate:.3f}")

    # ----------------------------
    # 2) ALL -> X_full, y
    # ----------------------------
    df_all_proc = preprocess_from_raw(df_all_aligned)
    X_df, y = get_feature_matrix(df_all_proc)
    X_full = X_df.to_numpy(dtype=np.float32)
    y = y.astype(int)
    feature_names = list(X_df.columns)

    # ----------------------------
    # 3) group mapping
    # ----------------------------
    group_names, group_to_indices = build_group_to_indices(feature_names)
    G = len(group_names)

    # ----------------------------
    # 4) CALL -> onehot + init_mask from raw missing rules
    # ----------------------------
    X_call_onehot = build_X_call_onehot_from_call_raw(df_call_aligned, feature_names)
    init_mask_onehot = build_init_mask_onehot_from_call_raw(df_call_aligned, group_names, group_to_indices)

    # ----------------------------
    # 5) missing / forced masks
    # ----------------------------
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

    # ----------------------------
    # 6) split (train/valid에서 policy 해석)
    # ----------------------------
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

    # ----------------------------
    # 7) oracle
    # ----------------------------
    oracle = FrozenOracle(str(P.MODEL_PKL))

    def oracle_fn(x_np: np.ndarray):
        return oracle.predict_proba(x_np)

    # ----------------------------
    # 8) load RL policy (explicit path)
    # ----------------------------
    model_path = Path(getattr(C, "POLICY_MODEL_PATH", "")).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(
            f"Policy model not found: {model_path}\n"
            f"Set RLConfig.POLICY_MODEL_PATH in config.py to a valid .zip path."
        )
    print(f"[MODEL] using explicitly specified model: {model_path}")
    rl_model = MaskablePPO.load(str(model_path))

    # ----------------------------
    # 9) build env for train/valid
    #     - 해석 단계에서는 oversampling OFF 고정
    # ----------------------------
    allow_red = bool(getattr(C, "ALLOW_REDUNDANT_QUESTIONS", False))

    env_tr = FeatureAcquisitionEnv(
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
        sanity_check_groups=True,
        allow_redundant_questions=allow_red,
        use_episode_oversampling=False,
    )

    env_va = FeatureAcquisitionEnv(
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
        sanity_check_groups=True,
        allow_redundant_questions=allow_red,
        use_episode_oversampling=False,
    )

    # ----------------------------
    # 10) rollout & summarize
    # ----------------------------
    # heatmap step 수가 너무 길면 보기 힘들어서 예: 15로 컷 (원하면 바꾸세요)
    max_steps_for_heatmap = 15

    df_imp_tr = rollout_policy_on_split(
        split_name="train",
        rl_model=rl_model,
        base_env=env_tr,
        y_split=y_tr2,
        group_names=group_names,
        out_dir=out_dir,
        max_heatmap_steps=max_steps_for_heatmap,
    )

    df_imp_va = rollout_policy_on_split(
        split_name="valid",
        rl_model=rl_model,
        base_env=env_va,
        y_split=y_va2,
        group_names=group_names,
        out_dir=out_dir,
        max_heatmap_steps=max_steps_for_heatmap,
    )

    merged = pd.concat([df_imp_tr, df_imp_va], ignore_index=True)
    merged_path = out_dir / "policy_importance_train_valid_merged.xlsx"
    merged.to_excel(merged_path, index=False)
    print(f"[OK] merged saved: {merged_path}")
    print(f"[DONE] outputs -> {out_dir}")


if __name__ == "__main__":
    main()
