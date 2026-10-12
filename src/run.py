"""Run the pipeline end to end.

python -m src.run          data -> targets -> features -> validation baselines -> validation model selection
python -m src.run --test   the same, then the walk-forward evaluation on the test window
"""
import argparse
import subprocess
import sys

STAGES = [
    ["src.data"],
    ["src.target"],
    ["src.features"],
    ["src.evaluate", "--split", "val"],
    ["src.train"],
]
TEST_STAGE = ["src.evaluate", "--split", "test"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test", action="store_true", help="also score the test window (walk-forward)")
    args = parser.parse_args()
    for stage in STAGES + ([TEST_STAGE] if args.test else []):
        print(f"\n===== python -m {' '.join(stage)} =====", flush=True)
        subprocess.run([sys.executable, "-m", *stage], check=True)


if __name__ == "__main__":
    main()
