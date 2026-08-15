"""Candidateパイプラインで使用するモデルと実行条件。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final


@dataclass(frozen=True)
class CandidateModel:
    role: str
    model_id: str
    runtime: str
    quantization: str
    max_tokens: int
    temperature: float = 0.0
    enable_thinking: bool = False

    def as_dict(self) -> dict[str, str | int | float | bool]:
        return asdict(self)


# 公式MXFP4 checkpointはvLLM向けなので、Apple Siliconでは
# MLXへ変換されたcheckpointをCandidateとして使用する。
GENERATOR: Final = CandidateModel(
    role="generator",
    model_id="tocchitocchi/GPT-OSS-Swallow-120B-RL-v0.1-6bit-mlx",
    runtime="mlx-lm",
    quantization="6bit-mlx-community-conversion",
    max_tokens=4096,
)

REVIEWER: Final = CandidateModel(
    role="reviewer",
    model_id="mlx-community/Qwen3.6-27B-5bit",
    runtime="mlx-lm",
    quantization="5bit-mlx-community-conversion-non-mtp",
    max_tokens=8192,
)

PRIMARY_JUDGE: Final = CandidateModel(
    role="primary_judge",
    model_id=REVIEWER.model_id,
    runtime=REVIEWER.runtime,
    quantization=REVIEWER.quantization,
    max_tokens=3600,
)

INDEPENDENT_JUDGE: Final = CandidateModel(
    role="independent_judge",
    model_id="mlx-community/gemma-3-text-27b-it-4bit",
    runtime="mlx-lm",
    quantization="4bit",
    max_tokens=3600,
)

PIPELINE_VERSION: Final = "candidate-multi-model-v1.0.0"


def model_manifest() -> dict[str, object]:
    return {
        "pipeline_version": PIPELINE_VERSION,
        "models": [
            model.as_dict()
            for model in (
                GENERATOR,
                REVIEWER,
                PRIMARY_JUDGE,
                INDEPENDENT_JUDGE,
            )
        ],
        "independent_evaluation": (
            PRIMARY_JUDGE.model_id != INDEPENDENT_JUDGE.model_id
        ),
    }
