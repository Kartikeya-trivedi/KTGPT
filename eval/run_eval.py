"""
Evaluation Wrapper
==================

Runs HumanEval, MBPP, and general benchmarks (ARC, HellaSwag) against
KT_GPT checkpoints. Logs results to W&B and saves to JSON.

Uses lm-evaluation-harness for standardized benchmark execution.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class EvalConfig:
    """Configuration for evaluation runs."""
    checkpoint_path: str = ""
    output_dir: str = "./eval_results"
    benchmarks: list[str] = field(default_factory=lambda: [
        "humaneval",
        "mbpp",
    ])
    # Generation parameters
    max_new_tokens: int = 512
    temperature: float = 0.0       # greedy for pass@1
    top_p: float = 1.0
    num_samples: int = 1           # pass@1 by default
    batch_size: int = 4
    wandb_project: str = "kt-gpt"


class ModelEvaluator:
    """Evaluates KT_GPT model on code and general benchmarks.

    Supports:
      - HumanEval (pass@1, pass@10)
      - MBPP (pass@1)
      - General: ARC, HellaSwag (via lm-evaluation-harness)

    Results are saved to JSON and optionally logged to W&B.
    """

    def __init__(
        self,
        model,
        tokenizer,
        config: EvalConfig,
        device: torch.device,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        """Generate a completion for a single prompt."""
        max_tokens = max_new_tokens or self.config.max_new_tokens
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        output_ids = input_ids.clone()
        self.model.eval()

        for _ in range(max_tokens):
            with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                logits, _ = self.model(output_ids)

            next_logit = logits[:, -1, :]

            if self.config.temperature == 0.0:
                next_token = next_logit.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(next_logit / self.config.temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            output_ids = torch.cat([output_ids, next_token], dim=1)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        generated = output_ids[0][input_ids.shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def eval_humaneval(self) -> dict:
        """Evaluate on HumanEval benchmark.

        Loads problems, generates solutions, executes against test cases.
        Returns pass@1 and pass@k metrics.
        """
        try:
            from human_eval.data import read_problems
            from human_eval.evaluation import evaluate_functional_correctness
        except ImportError:
            print("[EVAL] human-eval not installed. Skipping HumanEval.")
            return {"humaneval_pass@1": None}

        print("[EVAL] Running HumanEval...")
        problems = read_problems()
        samples = []

        for task_id, problem in problems.items():
            prompt = problem["prompt"]
            completion = self.generate(prompt)
            samples.append({
                "task_id": task_id,
                "completion": completion,
            })

        # Write samples to temp file for evaluation
        output_path = Path(self.config.output_dir) / "humaneval_samples.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        results = evaluate_functional_correctness(
            str(output_path),
            k=[1, 10],
        )

        print(f"[EVAL] HumanEval: pass@1={results.get('pass@1', 0):.3f}, "
              f"pass@10={results.get('pass@10', 0):.3f}")
        return results

    def eval_mbpp(self) -> dict:
        """Evaluate on MBPP benchmark."""
        try:
            from datasets import load_dataset
        except ImportError:
            print("[EVAL] datasets not installed. Skipping MBPP.")
            return {"mbpp_pass@1": None}

        print("[EVAL] Running MBPP...")
        ds = load_dataset("mbpp", "sanitized", split="test")

        correct = 0
        total = 0

        for sample in ds:
            prompt = (
                f"Write a Python function to solve the following problem.\n\n"
                f"Problem: {sample['text']}\n\n"
                f"Test cases:\n{chr(10).join(sample['test_list'][:2])}\n\n"
                f"Solution:\n"
            )

            completion = self.generate(prompt)
            # Try executing the completion with the test cases
            from data.filter import execute_with_tests
            test_code = "\n".join(sample["test_list"])
            passed, _ = execute_with_tests(completion, test_code, timeout=10)

            if passed:
                correct += 1
            total += 1

            if total % 50 == 0:
                print(f"  [{total}/{len(ds)}] pass@1 so far: {correct/total:.3f}")

        result = {"mbpp_pass@1": correct / max(total, 1), "mbpp_total": total}
        print(f"[EVAL] MBPP: pass@1={result['mbpp_pass@1']:.3f} ({correct}/{total})")
        return result

    def eval_lm_harness(self, tasks: list[str]) -> dict:
        """Run benchmarks using lm-evaluation-harness.

        Supports: arc_easy, arc_challenge, hellaswag, winogrande, etc.
        """
        try:
            import lm_eval
        except ImportError:
            print("[EVAL] lm-eval not installed. Skipping harness benchmarks.")
            return {}

        print(f"[EVAL] Running lm-eval-harness tasks: {tasks}")

        # This requires wrapping our model for the harness API
        # For now, return a placeholder
        print("[EVAL] lm-eval-harness integration requires model wrapper. "
              "Use `lm_eval --model custom` with a wrapper class.")
        return {task: None for task in tasks}

    def run_all(self) -> dict:
        """Run all configured benchmarks and save results."""
        all_results: dict = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "checkpoint": self.config.checkpoint_path,
        }

        for benchmark in self.config.benchmarks:
            if benchmark == "humaneval":
                all_results.update(self.eval_humaneval())
            elif benchmark == "mbpp":
                all_results.update(self.eval_mbpp())
            elif benchmark in ("arc", "hellaswag", "winogrande"):
                all_results.update(self.eval_lm_harness([benchmark]))
            else:
                print(f"[EVAL] Unknown benchmark: {benchmark}")

        # Save results
        output_path = Path(self.config.output_dir) / "results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n[EVAL] Results saved to {output_path}")

        # Log to W&B if available
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({f"eval/{k}": v for k, v in all_results.items()
                          if isinstance(v, (int, float)) and v is not None})
        except ImportError:
            pass

        return all_results
