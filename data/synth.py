"""
Synthetic Data Generation (R1-style)
=====================================

Generates high-quality code reasoning data for SFT by:
  1. Sampling problems from CodeContests / competitive programming datasets
  2. Generating N candidate solutions per problem using the base model
  3. Executing each solution against test cases
  4. Keeping only solutions that pass ALL tests
  5. Formatting as [PROBLEM] -> [REASONING] -> [CODE] -> [TESTS]

This produces training data for supervised fine-tuning (SFT) that
teaches the model to reason step-by-step about code problems.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from data.filter import execute_with_tests


@dataclass
class SynthConfig:
    """Configuration for synthetic data generation."""
    num_candidates: int = 8       # solutions to generate per problem
    temperature: float = 0.8      # sampling temperature for diversity
    max_new_tokens: int = 2048    # max generation length
    execution_timeout: int = 10   # seconds per test execution
    output_path: str = "./data/synth_output.jsonl"
    seed: int = 42


@dataclass
class SynthSample:
    """A single synthetic training example."""
    problem: str
    reasoning: str
    code: str
    tests: str
    source: str = ""              # e.g. "codecontests", "leetcode"
    hash: str = ""                # dedup key

    def to_training_text(self) -> str:
        """Format as the training string for SFT."""
        return (
            f"[PROBLEM]\n{self.problem}\n\n"
            f"[REASONING]\n{self.reasoning}\n\n"
            f"[CODE]\n{self.code}\n\n"
            f"[TESTS]\n{self.tests}"
        )

    def to_dict(self) -> dict:
        return {
            "problem": self.problem,
            "reasoning": self.reasoning,
            "code": self.code,
            "tests": self.tests,
            "source": self.source,
            "hash": self.hash,
            "text": self.to_training_text(),
        }


class SyntheticDataGenerator:
    """Generates synthetic code reasoning data from a trained KT_GPT model.

    Pipeline:
      1. Load problems with test cases
      2. For each problem, prompt the model N times with temperature sampling
      3. Execute each candidate against test cases
      4. Keep passing solutions, deduplicate, save to JSONL

    This should only run AFTER the base model is trained (post Phase 2).
    """

    def __init__(
        self,
        model,
        tokenizer,
        config: SynthConfig,
        device: torch.device,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.seen_hashes: set[str] = set()

    def _compute_hash(self, code: str) -> str:
        """Hash the normalized code for deduplication."""
        normalized = code.strip().replace(" ", "").replace("\n", "")
        return hashlib.md5(normalized.encode()).hexdigest()

    def _generate_candidates(self, prompt: str) -> list[str]:
        """Generate N candidate solutions for a given problem prompt."""
        candidates: list[str] = []
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        eos_id = self.tokenizer.eos_token_id

        for _ in range(self.config.num_candidates):
            generated = input_ids.clone()

            with torch.no_grad():
                for _ in range(self.config.max_new_tokens):
                    with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                        logits, _ = self.model(generated)

                    next_logits = logits[:, -1, :] / self.config.temperature
                    probs = F.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    generated = torch.cat([generated, next_token], dim=1)

                    if next_token.item() == eos_id:
                        break

            text = self.tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True)
            candidates.append(text)

        return candidates

    def _extract_code(self, response: str) -> tuple[str, str]:
        """Extract code and reasoning from a generated response.

        Tries to split on [CODE] marker, otherwise treats the whole
        response as code with empty reasoning.
        """
        if "[CODE]" in response:
            parts = response.split("[CODE]", 1)
            reasoning = parts[0].replace("[REASONING]", "").strip()
            code = parts[1].strip()
        else:
            # Try to find a code block
            if "```python" in response:
                start = response.index("```python") + 9
                end = response.index("```", start) if "```" in response[start:] else len(response)
                code = response[start:end].strip()
                reasoning = response[:start - 9].strip()
            else:
                reasoning = ""
                code = response.strip()

        return reasoning, code

    def process_problem(
        self,
        problem: str,
        tests: str,
        source: str = "",
    ) -> list[SynthSample]:
        """Generate and validate solutions for a single problem.

        Returns only solutions that:
          1. Pass all test cases
          2. Are not duplicates of previously seen solutions
        """
        prompt = f"Solve the following programming problem. Think step by step.\n\n[PROBLEM]\n{problem}\n\n[REASONING]\n"

        candidates = self._generate_candidates(prompt)
        valid_samples: list[SynthSample] = []

        for candidate in candidates:
            reasoning, code = self._extract_code(candidate)

            if not code:
                continue

            # Execute against tests
            passed, output = execute_with_tests(
                code, tests, timeout=self.config.execution_timeout
            )
            if not passed:
                continue

            # Dedup
            code_hash = self._compute_hash(code)
            if code_hash in self.seen_hashes:
                continue
            self.seen_hashes.add(code_hash)

            valid_samples.append(SynthSample(
                problem=problem,
                reasoning=reasoning,
                code=code,
                tests=tests,
                source=source,
                hash=code_hash,
            ))

        return valid_samples

    def generate_dataset(
        self,
        problems: list[dict],
        problem_key: str = "description",
        tests_key: str = "tests",
        source_key: str = "source",
    ) -> list[SynthSample]:
        """Generate synthetic data from a list of problems.

        Args:
            problems: List of dicts with problem descriptions and test cases
            problem_key: Key for the problem description
            tests_key: Key for the test code
            source_key: Key for the data source name

        Returns:
            List of validated, deduplicated SynthSamples
        """
        all_samples: list[SynthSample] = []
        total_problems = len(problems)

        for i, prob in enumerate(problems):
            problem_text = prob.get(problem_key, "")
            test_text = prob.get(tests_key, "")
            source = prob.get(source_key, "")

            if not problem_text or not test_text:
                continue

            samples = self.process_problem(problem_text, test_text, source)
            all_samples.extend(samples)

            if (i + 1) % 10 == 0:
                rate = len(all_samples) / max(i + 1, 1)
                print(f"[SYNTH] {i+1}/{total_problems} problems, "
                      f"{len(all_samples)} valid solutions ({rate:.1f}/problem)")

        # Save to JSONL
        output_path = Path(self.config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for sample in all_samples:
                f.write(json.dumps(sample.to_dict()) + "\n")

        print(f"\n[SYNTH] Complete. {len(all_samples)} samples saved to {output_path}")
        print(f"  Problems: {total_problems}")
        print(f"  Unique solutions: {len(all_samples)}")
        print(f"  Avg solutions/problem: {len(all_samples)/max(total_problems,1):.1f}")

        return all_samples
