# config.py
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Paths:
    ROOT: Path = Path(__file__).resolve().parents[1]
    DATA: Path = ROOT / "data"

    ALL_XLSX: Path = DATA / "cb_ch_all.xlsx"
    CALL_XLSX: Path = DATA / "cb_ch_call_ver1.1.xlsx"
    MODEL_PKL: Path = DATA / "Prehospital_final_model.pkl"

    OUT: Path = ROOT / "outputs"


@dataclass(frozen=True)
class RLConfig:
    # env (✅ 21개 그룹 기준)
    MAX_QUESTIONS: int = 21
    MIN_QUESTIONS: int = 0
    SHAP_WARMSTART_STEPS: int = 0

    QUESTION_COST: float = 0.04 #default0.04
    STOP_PENALTY: float = 0.0

    # reward weights
    ALPHA_LOSS: float = 0.5 #1.0
    BETA_PROB: float = 1.0 #0.3

    # train
    TOTAL_TIMESTEPS: int = 1_000_000
    RANDOM_SEED: int = 42

    # valid / early stopping
    EVAL_FREQ: int = 20_480
    EARLYSTOP_PATIENCE: int = 10
    EARLYSTOP_MIN_DELTA: float = 1e-4

    # STOP reward experiment
    STOP_REWARD_MODE: str = "hybrid"   # "loss" | "aligned_prob" | "hybrid"
    STOP_REWARD_SCALE: float = 0.5 #default0.5

    # ✅ redundant 질문 허용 토글 (기존)
    ALLOW_REDUNDANT_QUESTIONS: bool = True  # True면 CALL==ALL이어도 질문 허용

    # ✅ (NEW) Episode sampling oversampling (진짜 RL용)
    # reset(idx=None)일 때, 사망자 케이스를 더 자주 뽑을지
    USE_EPISODE_OVERSAMPLING: bool = False

    # 사망자(y==1) 에피소드가 뽑힐 확률 (0~1)
    # 예: 0.5 => reset 할 때 절반은 사망자에서 샘플링(가능한 경우)
    EPISODE_POS_PROB: float = 0.5

    # Selected policy checkpoint used by evaluation/analysis scripts.
    POLICY_MODEL_PATH: str = "outputs/best_maskableppo_feature_acq_0121_default_a0.5_b1.0.zip"

    # PPO
    LEARNING_RATE: float = 3e-4
    N_STEPS: int = 2048
    BATCH_SIZE: int = 256
    GAMMA: float = 0.99

    # SEED
    # SEED = 42
