#!/usr/bin/env python3
"""
Runs every offline test file in tests/ and reports a combined summary.
No pytest required -- only pandas and numpy (already in requirements.txt).

Usage: python3 run_all_tests.py
"""
import subprocess
import sys
from pathlib import Path

TEST_DIR = Path(__file__).parent / "tests"


def main():
    test_files = sorted(TEST_DIR.glob("test_*.py"))
    if not test_files:
        print("No test files found in tests/")
        sys.exit(1)

    total_passed = 0
    total_failed = 0
    any_crashed = False

    for test_file in test_files:
        print(f"\n{'=' * 60}")
        print(f"Running {test_file.name}")
        print("=" * 60)
        result = subprocess.run([sys.executable, str(test_file)], capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)
            any_crashed = True

        # Parse the LAST "N passed, M failed" summary line each test file prints
        # (using the last match, not the first, avoids accidentally matching
        # an individual PASS/FAIL line that happens to contain both words).
        summary_lines = [
            line for line in result.stdout.splitlines()
            if line.strip().endswith("failed") and "passed" in line
        ]
        if summary_lines:
            try:
                parts = summary_lines[-1].strip().split(",")
                passed = int(parts[0].split()[0])
                failed = int(parts[1].split()[0])
                total_passed += passed
                total_failed += failed
            except (ValueError, IndexError):
                print(f"WARNING: could not parse summary line for {test_file.name}: {summary_lines[-1]!r}")
                any_crashed = True
        else:
            print(f"WARNING: no summary line found for {test_file.name}")
            any_crashed = True

    print(f"\n{'=' * 60}")
    print(f"TOTAL: {total_passed} passed, {total_failed} failed")
    print("=" * 60)

    if total_failed > 0 or any_crashed:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
