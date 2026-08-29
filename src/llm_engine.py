"""vLLM-based batched inference for Qwen2.5-Coder-7B-Instruct.

Two engines:
- LLMEngine: a single vLLM instance that is reused across many calls.
- LoRAEngine: optional LoRA adapter loading via `enable_lora=True`.

This module is the *only* place that talks to vLLM. Everything else (baselines,
VISTA agent, repair loop) goes through `LLMEngine.generate(messages_batch, ...)`.
"""
from __future__ import annotations

import os
import time
from typing import Any, Iterable, Sequence

from loguru import logger

from . import config as C


class LLMEngine:
    def __init__(
        self,
        model_path: str = C.BASE_MODEL,
        gpu_memory_utilization: float = C.VLLM_GPU_MEMORY_UTILIZATION,
        max_model_len: int = C.VLLM_MAX_MODEL_LEN,
        dtype: str = C.VLLM_DTYPE,
        enable_lora: bool = False,
        max_loras: int = 1,
        max_lora_rank: int = 32,
        seed: int = 42,
        download_dir: str | None = None,
    ):
        # Import vLLM lazily so that simply importing this module is cheap.
        from vllm import LLM, SamplingParams  # noqa: F401
        # Suppress some noisy logs.
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        logger.info(f"[LLMEngine] loading model: {model_path}")

        self.model_path = model_path
        self.llm = LLM(
            model=model_path,
            tokenizer=model_path,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enable_lora=enable_lora,
            max_loras=max_loras if enable_lora else 1,
            max_lora_rank=max_lora_rank,
            seed=seed,
            trust_remote_code=True,
            download_dir=download_dir,
            disable_log_stats=True,
        )
        self._SamplingParams = SamplingParams
        self._tokenizer = self.llm.get_tokenizer()
        logger.info("[LLMEngine] model loaded")

    def _make_sampling(self, **kw: Any):
        # Defaults aligned with config.py
        sp = dict(
            temperature=C.GEN_TEMPERATURE_GREEDY,
            top_p=C.GEN_TOP_P,
            max_tokens=C.GEN_MAX_TOKENS,
            n=1,
            stop=None,
        )
        sp.update({k: v for k, v in kw.items() if v is not None})
        return self._SamplingParams(**sp)

    def chat_template(self, messages: Sequence[dict]) -> str:
        return self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    # ------------------------------------------------------------------
    # Main entry point used by every caller in this codebase.
    # ------------------------------------------------------------------
    def generate(
        self,
        messages_batch: Sequence[Sequence[dict]],
        *,
        temperature: float = C.GEN_TEMPERATURE_GREEDY,
        top_p: float = C.GEN_TOP_P,
        max_tokens: int = C.GEN_MAX_TOKENS,
        n: int = 1,
        stop: Sequence[str] | None = None,
        seed: int | None = None,
        lora_request: Any = None,
        track_latency: bool = True,
    ) -> list[list[dict]]:
        """Run vLLM on a batch of message lists.

        Returns a list (same length as `messages_batch`) of lists of dicts:
            [[{"text": str, "latency_s": float}, ...n samples per query], ...]
        """
        prompts = [self.chat_template(msgs) for msgs in messages_batch]
        sp = self._make_sampling(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            n=n,
            stop=list(stop) if stop else None,
            seed=seed,
        )
        kwargs: dict[str, Any] = {}
        if lora_request is not None:
            kwargs["lora_request"] = lora_request

        t0 = time.perf_counter()
        outputs = self.llm.generate(prompts, sp, use_tqdm=False, **kwargs)
        total = time.perf_counter() - t0
        # vLLM batches everything on the GPU in parallel; per-item wall-clock isn't
        # meaningful. Distribute the total wall time evenly as an approximation,
        # mirroring what NetArena would record at the HTTP layer.
        per_query_lat = total / max(1, len(prompts))

        results: list[list[dict]] = []
        for out in outputs:
            samples = []
            for o in out.outputs:
                samples.append({"text": o.text, "latency_s": per_query_lat})
            results.append(samples)
        return results


# ---------- LoRA helper ----------


def make_lora_request(adapter_path: str, *, lora_id: int = 1, lora_name: str = "sft"):
    """Build an LoRARequest for use with LLMEngine.generate(lora_request=...)."""
    from vllm.lora.request import LoRARequest
    return LoRARequest(lora_name=lora_name, lora_int_id=lora_id, lora_path=adapter_path)
