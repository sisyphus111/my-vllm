import argparse
import os
from typing import Optional

from vllm import LLM, SamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run offline inference with vLLM (V1) using tensor parallelism. "
            "Batch size is 1, for a single prompt."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/mnt/disk1/lsy/models/Byted-DAPO-Qwen-32B",
        help=(
            "Model repo id or local path (e.g. 'meta-llama/Llama-3.1-8B-Instruct' "
            "or '/path/to/model')."
        ),
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="写一部武侠小说，背景设定在唐朝，主人公是一个叫做李白的剑客，他因为一次意外的相遇，卷入了一场江湖的纷争，最终成为了江湖中的一代宗师。",
        help="Single prompt string to generate from.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        "--tp",
        dest="tp",
        type=int,
        default=4,
        help="Tensor parallel world size (default: 4).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=20480,
        help="Maximum number of new tokens to generate (default: 64).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature; 0.0 means greedy (default: 0.0).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p nucleus sampling probability (default: 1.0).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=-1,
        help="Top-k sampling (default: -1).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        choices=[None, "auto", "float16", "bfloat16", "float32"],
        help="Optional model dtype override (default: auto from model config).",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow execution of custom code from model repo if needed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Ensure V1 code path so our instrumentation in v1/gpu_model_runner executes.
    os.environ.setdefault("VLLM_USE_V1", "1")

    # Initialize engine (single-node TP). Ensure CUDA_VISIBLE_DEVICES exposes >= tp GPUs.
    llm_kwargs: dict[str, object] = {
        "model": args.model,
        "tensor_parallel_size": args.tp,
    }
    if args.dtype is not None:
        llm_kwargs["dtype"] = args.dtype
    if args.trust_remote_code:
        llm_kwargs["trust_remote_code"] = True

    llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    print(f"\n[Info] TP={args.tp} | max_tokens={args.max_tokens} | temperature={args.temperature} | top_p={args.top_p}")
    print(f"[Info] Model: {args.model}")
    print(f"[Prompt] {args.prompt}\n")

    # Batch size = 1: pass a single string
    outputs = llm.generate(args.prompt, sampling_params)

    # There will be exactly one RequestOutput in the list
    for output in outputs:
        generated_text: Optional[str] = None
        if output.outputs:
            generated_text = output.outputs[0].text
        print("-" * 60)
        print("Generated text:")
        print(generated_text or "")
        print("-" * 60)

    print("\n[Info] If you enabled our decode timing instrumentation, check: /tmp/native_vllm_gpu_model_runner_execute_model_timing.log")


if __name__ == "__main__":
    main()


