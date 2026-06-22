from __future__ import annotations
import numpy as np
from pathlib import Path
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib.common.maskable.utils import get_action_masks

class ValidBestModelCallback(BaseCallback):
    def __init__(
        self,
        valid_env,
        oracle_predict_fn,
        save_path: Path,
        eval_freq: int,
        patience: int,
        min_delta: float,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.valid_env = valid_env
        self.oracle_predict_fn = oracle_predict_fn
        self.save_path = Path(save_path)
        self.eval_freq = int(eval_freq)
        self.patience = int(patience)
        self.min_delta = float(min_delta)

        self.best_loss = np.inf
        self.bad_count = 0

    def _evaluate_weighted_logloss(self) -> float:
        # valid_env는 FeatureAcquisitionEnv(단일) + ActionMasker로 감싸져 있다고 가정
        env = self.valid_env
        base_env = env.unwrapped

        losses = []
        n = base_env.n
        for i in range(n):
            obs, _ = env.reset(options={"idx": i})

            done = False
            while not done:
                action_masks = get_action_masks(env)
                action, _ = self.model.predict(obs, deterministic=True, action_masks=action_masks)
                obs, reward, terminated, truncated, _ = env.step(int(action))
                done = bool(terminated or truncated)

            # 종료 시점 loss
            loss = base_env._current_loss()
            losses.append(loss)

        return float(np.mean(losses))

    def _on_step(self) -> bool:
        if self.eval_freq <= 0:
            return True
        if (self.num_timesteps % self.eval_freq) != 0:
            return True

        cur = self._evaluate_weighted_logloss()

        if self.verbose:
            print(f"[VALID @ {self.num_timesteps}] mean_weighted_logloss={cur:.6f} best={self.best_loss:.6f}")

        improved = (self.best_loss - cur) > self.min_delta
        if improved:
            self.best_loss = cur
            self.bad_count = 0
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(str(self.save_path))
            if self.verbose:
                print(f"[BEST] saved -> {self.save_path}")
        else:
            self.bad_count += 1
            if self.verbose:
                print(f"[EARLYSTOP] no improve ({self.bad_count}/{self.patience})")
            if self.bad_count >= self.patience:
                if self.verbose:
                    print("[EARLYSTOP] stopping training.")
                return False

        return True
