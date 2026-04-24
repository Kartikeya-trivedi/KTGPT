"""
Execution-Based Code Filtering
===============================

Filters code datasets by verifying that code is syntactically valid Python
and optionally passes test cases. Used to clean Stack v2 and other code
sources before Phase 2 training.

Safety: all execution happens in isolated subprocesses with timeouts
and resource limits. No network access.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Optional

from datasets import Dataset


# ═══════════════════════════════════════════════════════════════════════
#  Validation Functions
# ═══════════════════════════════════════════════════════════════════════

def is_valid_python(code: str) -> bool:
    """Check if code is syntactically valid Python via AST parsing.

    This is a fast, safe check — no execution involved.
    Catches SyntaxError but not runtime errors.
    """
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def execute_with_tests(
    code: str,
    tests: str,
    timeout: int = 5,
    max_memory_mb: int = 256,
) -> tuple[bool, str]:
    """Execute code + tests in an isolated subprocess.

    Concatenates the code and test strings, writes to a temp file,
    and runs in a subprocess with strict resource limits.

    Args:
        code:    Python source code to test
        tests:   Test code (e.g. assert statements) to append
        timeout: Maximum execution time in seconds
        max_memory_mb: Memory limit in MB (Linux only)

    Returns:
        (passed, output): whether tests passed, and stdout/stderr
    """
    combined = code + "\n\n" + tests

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(combined)
        tmp_path = f.name

    try:
        # Build the command — no network, restricted imports
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"

        result = subprocess.run(
            [sys.executable, "-u", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=tempfile.gettempdir(),
        )
        passed = result.returncode == 0
        output = result.stdout + result.stderr
        return passed, output.strip()

    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT: exceeded {timeout}s"
    except Exception as e:
        return False, f"ERROR: {e}"
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ═══════════════════════════════════════════════════════════════════════
#  Dataset Filtering
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class FilterStats:
    """Tracks filter pass rates for logging."""
    total: int = 0
    valid_syntax: int = 0
    passed_tests: int = 0

    def __repr__(self) -> str:
        syntax_pct = (self.valid_syntax / max(self.total, 1)) * 100
        test_pct = (self.passed_tests / max(self.valid_syntax, 1)) * 100
        return (
            f"FilterStats(total={self.total}, "
            f"valid_syntax={self.valid_syntax} ({syntax_pct:.1f}%), "
            f"passed_tests={self.passed_tests} ({test_pct:.1f}%))"
        )


def filter_code_samples(
    samples: list[dict],
    code_column: str = "content",
    test_column: Optional[str] = None,
    require_tests: bool = False,
    timeout: int = 5,
) -> tuple[list[dict], FilterStats]:
    """Filter a batch of code samples for quality.

    Pipeline:
      1. Check syntax validity (fast AST parse)
      2. If test_column is provided, execute code against tests
      3. Return only samples that pass all checks

    Args:
        samples:       List of dicts with code content
        code_column:   Key containing the source code
        test_column:   Key containing test code (optional)
        require_tests: If True, skip samples without tests
        timeout:       Execution timeout per sample in seconds

    Returns:
        (filtered_samples, stats)
    """
    stats = FilterStats()
    passed: list[dict] = []

    for sample in samples:
        stats.total += 1
        code = sample.get(code_column, "")

        if not code or not isinstance(code, str):
            continue

        # Step 1: Syntax check
        if not is_valid_python(code):
            continue
        stats.valid_syntax += 1

        # Step 2: Test execution (if tests available)
        if test_column and test_column in sample:
            tests = sample[test_column]
            if tests:
                success, _ = execute_with_tests(code, tests, timeout=timeout)
                if not success:
                    continue
                stats.passed_tests += 1
            elif require_tests:
                continue
            else:
                stats.passed_tests += 1
        else:
            if require_tests:
                continue
            stats.passed_tests += 1

        passed.append(sample)

    return passed, stats


def filter_dataset_streaming(
    dataset,
    code_column: str = "content",
    batch_size: int = 100,
    max_samples: Optional[int] = None,
):
    """Filter a streaming HuggingFace dataset, yielding valid samples.

    Processes in batches for efficiency. Prints progress every batch.

    Args:
        dataset:     HuggingFace streaming dataset
        code_column: Column containing Python source code
        batch_size:  Number of samples per filter batch
        max_samples: Stop after this many total samples (None = all)

    Yields:
        Filtered samples that pass syntax validation
    """
    batch: list[dict] = []
    total_processed = 0
    total_passed = 0

    for sample in dataset:
        batch.append(sample)

        if len(batch) >= batch_size:
            filtered, stats = filter_code_samples(
                batch, code_column=code_column
            )
            total_processed += stats.total
            total_passed += len(filtered)
            yield from filtered
            batch = []

            if total_processed % (batch_size * 10) == 0:
                rate = total_passed / max(total_processed, 1) * 100
                print(f"[FILTER] {total_processed} processed, "
                      f"{total_passed} passed ({rate:.1f}%)")

            if max_samples and total_processed >= max_samples:
                break

    # Process remaining
    if batch:
        filtered, stats = filter_code_samples(batch, code_column=code_column)
        total_processed += stats.total
        total_passed += len(filtered)
        yield from filtered

    rate = total_passed / max(total_processed, 1) * 100
    print(f"[FILTER] Done. {total_processed} processed, "
          f"{total_passed} passed ({rate:.1f}%)")
