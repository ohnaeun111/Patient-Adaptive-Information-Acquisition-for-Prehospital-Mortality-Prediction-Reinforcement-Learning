# env_feature_acq.py
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding

class FeatureAcquisitionEnv(gym.Env):
    """
    21개 그룹 단위 질문 환경
    Action:
      0..G-1 : group reveal (ALL 값으로 채움)
      G      : STOP

    Observation:
      concat([x_known(one-hot d), mask(one-hot d)])

    - missing_group_mask: CALL raw 기준 missing(질문 필요) 여부
    - forced_group_mask : Age/Gender 등 "결측이면 무조건 질문" 그룹
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        X_full: np.ndarray,
        y: np.ndarray,
        init_mask_onehot: np.ndarray,
        X_init_known_onehot: np.ndarray,
        oracle_predict_fn,
        group_names,
        group_to_indices,
        missing_group_mask: np.ndarray,  # (n,G)
        forced_group_mask: np.ndarray,   # (n,G)
        shap_rank_group_indices=None,
        min_questions: int = 0,
        max_questions: int = 15,
        question_cost: float = 0.01,
        shap_warmstart_steps: int = 0,
        seed: int = 42,
        alpha_loss: float = 1.0,
        beta_prob: float = 0.3,
        stop_penalty: float = 0.0,
        invalid_penalty: float = 0.05,
        terminal_bonus_scale: float = 0.2,
        same_tol: float = 1e-12,
        stop_reward_mode: str = "loss",
        stop_reward_scale: float = 1.0,
        sanity_check_groups: bool = True,

        # ✅ 기존 기능: redundant 질문 허용 토글
        allow_redundant_questions: bool = False,

        # ✅ NEW: episode sampling oversampling
        use_episode_oversampling: bool = False,
        episode_pos_prob: float = 0.5,
    ):
        super().__init__()

        self.X_full = np.asarray(X_full, dtype=np.float32)
        self.y = np.asarray(y, dtype=int)
        self.init_mask = np.asarray(init_mask_onehot, dtype=np.uint8)
        self.X_init_known = np.asarray(X_init_known_onehot, dtype=np.float32)
        self.oracle_predict_fn = oracle_predict_fn

        if self.X_full.shape != self.X_init_known.shape:
            raise ValueError("X_full and X_init_known_onehot must have same shape.")

        self.n, self.d = self.X_full.shape

        self.group_names = list(group_names)
        self.group_to_indices = {k: list(v) for k, v in group_to_indices.items()}
        self.G = len(self.group_names)

        if sanity_check_groups:
            empty = [g for g in self.group_names if len(self.group_to_indices.get(g, [])) == 0]
            if empty:
                raise ValueError(
                    f"[SANITY] group_to_indices has empty groups: {empty}. Fix group_map to match preprocess column names."
                )

        self.missing_group_mask = np.asarray(missing_group_mask, dtype=bool)
        self.forced_group_mask = np.asarray(forced_group_mask, dtype=bool)

        self.shap_rank_group_indices = list(shap_rank_group_indices) if shap_rank_group_indices is not None else []
        self.min_questions = int(min_questions)
        self.max_questions = int(max_questions)
        self.question_cost = float(question_cost)
        self.shap_warmstart_steps = int(shap_warmstart_steps)

        self.alpha_loss = float(alpha_loss)
        self.beta_prob = float(beta_prob)
        self.stop_penalty = float(stop_penalty)
        self.invalid_penalty = float(invalid_penalty)
        self.terminal_bonus_scale = float(terminal_bonus_scale)
        self.same_tol = float(same_tol)

        self.stop_reward_mode = str(stop_reward_mode).lower().strip()
        if self.stop_reward_mode not in ("loss", "aligned_prob", "hybrid"):
            raise ValueError(
                f"stop_reward_mode must be one of ['loss','aligned_prob','hybrid'], got={stop_reward_mode}"
            )
        self.stop_reward_scale = float(stop_reward_scale)

        # ✅ redundant 질문 허용
        self.allow_redundant_questions = bool(allow_redundant_questions)

        # ✅ episode oversampling 설정
        self.use_episode_oversampling = bool(use_episode_oversampling)
        self.episode_pos_prob = float(np.clip(episode_pos_prob, 0.0, 1.0))

        # self.rng = np.random.default_rng(seed)

        # class weights for weighted logloss
        pos = np.sum(self.y == 1)
        neg = np.sum(self.y == 0)
        self.w_pos = self.n / (2.0 * max(pos, 1))
        self.w_neg = self.n / (2.0 * max(neg, 1))

        # indices for episode oversampling
        self.pos_indices = np.where(self.y == 1)[0]
        self.neg_indices = np.where(self.y == 0)[0]

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(2 * self.d,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.G + 1)
        self.np_random = None
        # self._reset_episode()
        self.seed(seed)
        self._reset_episode()
    # =========================
    # Seed handling
    # =========================
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def _obs(self):
        return np.concatenate([self.x_known, self.mask.astype(np.float32)])

    def _predict_p(self):
        p = self.oracle_predict_fn(self.x_known.reshape(1, -1))
        return float(np.asarray(p).reshape(-1)[0])

    def _weighted_logloss(self, y_true, p):
        eps = 1e-7
        p = float(np.clip(p, eps, 1 - eps))
        if y_true == 1:
            return -self.w_pos * np.log(p)
        return -self.w_neg * np.log(1 - p)

    def _current_loss(self):
        return self._weighted_logloss(self.y[self.idx], self._predict_p())

    def _aligned_prob_gain(self, p_from: float, p_to: float) -> float:
        if self.y[self.idx] == 1:
            return float(p_to - p_from)
        return float(p_from - p_to)

    def _group_indices(self, g_idx: int):
        name = self.group_names[g_idx]
        return self.group_to_indices.get(name, [])

    def _group_is_revealed(self, g_idx: int) -> bool:
        idxs = self._group_indices(g_idx)
        if not idxs:
            return True
        return bool(np.all(self.mask[idxs] == 1))

    def _group_would_change(self, g_idx: int) -> bool:
        idxs = self._group_indices(g_idx)
        if not idxs:
            return False
        full_vec = self.X_full[self.idx, idxs]
        known_vec = self.x_known[idxs]
        return bool(np.any(np.abs(full_vec - known_vec) > self.same_tol))

    def _apply_group(self, g_idx: int, count_as_question: bool, record_to: str):
        name = self.group_names[g_idx]
        idxs = self._group_indices(g_idx)
        if not idxs:
            return

        self.mask[idxs] = 1
        self.x_known[idxs] = self.X_full[self.idx, idxs]

        if record_to == "warmstart":
            self.warmstart_group_indices.append(int(g_idx))
            self.warmstart_group_names.append(str(name))
        elif record_to == "asked":
            self.asked_group_indices.append(int(g_idx))
            self.asked_group_names.append(str(name))

        if count_as_question:
            self.questions_asked += 1

    def _apply_forced_question(self, g_idx: int):
        """
        forced_group_mask=True 인 경우에만 호출됨.
        - 반드시 questions_asked 카운트
        - 반드시 asked_groups 기록
        - 반드시 ALL 값으로 채움
        """
        self._apply_group(g_idx, count_as_question=True, record_to="asked")

    def _action_mask(self):
        m = np.zeros(self.G + 1, dtype=bool)

        for g in range(self.G):
            # 1) CALL 기준 missing이어야 질문 후보
            if not self.missing_group_mask[self.idx, g]:
                continue

            # 2) 이미 reveal이면 제외
            if self._group_is_revealed(g):
                continue

            # 3) redundant 질문 허용 여부
            #    - False: ALL==CALL(변화 없음)이면 질문 불가
            #    - True : 변화 없어도 질문 허용(다만 missing인 경우에 한해)
            if (not self.allow_redundant_questions) and (not self._group_would_change(g)):
                continue

            m[g] = True

        m[self.G] = (self.questions_asked >= self.min_questions)
        return m

    def _sample_episode_index(self) -> int:
        """
        ✅ NEW: episode sampling oversampling
        - idx가 지정되지 않은 reset에서만 사용됨.
        - pos/neg 둘 다 존재할 때만 동작.
        """
        if (not self.use_episode_oversampling):
            # return int(self.rng.integers(0, self.n))
            return int(self.np_random.integers(0, self.n))

        if len(self.pos_indices) == 0 or len(self.neg_indices) == 0:
            # return int(self.rng.integers(0, self.n))
            return int(self.np_random.integers(0, self.n))

        # if float(self.rng.random()) < self.episode_pos_prob:
            # return int(self.rng.choice(self.pos_indices))
        # return int(self.rng.choice(self.neg_indices))
        if float(self.np_random.random()) < self.episode_pos_prob:
            return int(self.np_random.choice(self.pos_indices))
        return int(self.np_random.choice(self.neg_indices))

    def _reset_episode(self, idx=None):
        # ✅ idx가 주어지면 그대로(평가/엑셀 생성은 항상 이 경로)
        if idx is None:
            self.idx = int(self._sample_episode_index())
        else:
            self.idx = int(idx)

        self.t = 0
        self.questions_asked = 0
        self.asked_group_indices = []
        self.asked_group_names = []
        self.warmstart_group_indices = []
        self.warmstart_group_names = []

        self.x_known = self.X_init_known[self.idx].copy().astype(np.float32)
        self.mask = self.init_mask[self.idx].copy()

        self.done = False

        self.p_call = self._predict_p()
        self.loss_call = self._weighted_logloss(self.y[self.idx], self.p_call)
        self.initial_loss = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        idx = options.get("idx") if isinstance(options, dict) else None
        self._reset_episode(idx)

        # forced 질문 (결측일 때만 True로 들어옴)
        forced_idxs = np.where(self.forced_group_mask[self.idx])[0].tolist()
        for g in forced_idxs:
            g = int(g)
            if 0 <= g < self.G:
                self._apply_forced_question(g)

        # SHAP warmstart (질문 카운트 X)
        k = 0
        for g in self.shap_rank_group_indices:
            if k >= self.shap_warmstart_steps:
                break
            g = int(g)
            if 0 <= g < self.G:
                if (
                    self.missing_group_mask[self.idx, g]
                    and (not self._group_is_revealed(g))
                    and (self.allow_redundant_questions or self._group_would_change(g))
                ):
                    self._apply_group(g, count_as_question=False, record_to="warmstart")
                    k += 1

        self.initial_loss = self._current_loss()
        return self._obs(), {}


    def step(self, action: int):
        if self.done:
            return self._obs(), 0.0, True, False, {}

        action = int(action)
        terminated, truncated = False, False

        p_before = self._predict_p()
        loss_before = self._current_loss()

        # STOP
        if action == self.G:
            p_now = p_before
            loss_now = loss_before

            if self.stop_reward_mode == "loss":
                core = (self.loss_call - loss_now)
            elif self.stop_reward_mode == "aligned_prob":
                core = self._aligned_prob_gain(self.p_call, p_now)
            else:
                core = (self.loss_call - loss_now) + self._aligned_prob_gain(self.p_call, p_now)

            reward = self.stop_reward_scale * core - self.stop_penalty
            self.done = True
            return self._obs(), float(reward), True, False, {}

        am = self._action_mask()
        if action < 0 or action >= self.G or (not am[action]):
            reward = -self.invalid_penalty
            self.t += 1
            if self.t >= self.max_questions:
                truncated = True
                self.done = True
                reward += self.terminal_bonus_scale * (self.loss_call - loss_before)
            return self._obs(), float(reward), terminated, truncated, {}

        # 질문 실행
        self._apply_group(action, count_as_question=True, record_to="asked")
        self.t += 1

        p_after = self._predict_p()
        loss_after = self._current_loss()

        delta_loss = loss_before - loss_after
        delta_prob = self._aligned_prob_gain(p_before, p_after)

        reward = (self.alpha_loss * delta_loss + self.beta_prob * delta_prob - self.question_cost)

        if self.questions_asked >= self.max_questions or self.t >= self.max_questions:
            truncated = True
            self.done = True
            reward += self.terminal_bonus_scale * (self.loss_call - loss_after)

        return self._obs(), float(reward), terminated, truncated, {}
