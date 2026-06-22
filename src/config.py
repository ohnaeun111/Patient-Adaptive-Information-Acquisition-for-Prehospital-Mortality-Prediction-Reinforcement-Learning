# config.py
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Paths:
    ROOT: Path = Path(__file__).resolve().parents[1]
    DATA: Path = ROOT / "data"
    OUT: Path = ROOT / "outputs"

    ALL_XLSX: Path = DATA / ".xlsx"
    CALL_XLSX: Path = DATA / ".xlsx"
    MODEL_PKL: Path = DATA / ".pkl"

    BEST_POLICY: Path = OUT / "best_maskableppo_feature_acq.zip"
    LAST_POLICY: Path = OUT / "last_maskableppo_feature_acq.zip"


@dataclass(frozen=True)
class RLConfig:
    # env (✅ 21개 그룹 기준)
    MAX_QUESTIONS: int = 21
    MIN_QUESTIONS: int = 0
    SHAP_WARMSTART_STEPS: int = 0

    QUESTION_COST: float = 0.1 
    STOP_PENALTY: float = 0.0

    # reward weights
    ALPHA_LOSS: float = 1.0
    
    TOTAL_TIMESTEPS: int = 1_000_000
    RANDOM_SEED: int = 42

    EVAL_FREQ: int = 20_480
    EARLYSTOP_PATIENCE: int = 10
    EARLYSTOP_MIN_DELTA: float = 1e-4

    STOP_REWARD_MODE: str = "hybrid"  
    STOP_REWARD_SCALE: float = 0.5 


    ALLOW_REDUNDANT_QUESTIONS: bool = True 
    USE_EPISODE_OVERSAMPLING: bool = False

    EPISODE_POS_PROB: float = 0.5

    # PPO
    LEARNING_RATE: float = 3e-4
    N_STEPS: int = 2048
    BATCH_SIZE: int = 256
    GAMMA: float = 0.99

    # SEED
    # SEED = 42
