"""Generate the synthetic customer dataset."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_generator import save_demo_dataset


def main():
    parser = argparse.ArgumentParser(description="Generate Customer360 AI demo data")
    parser.add_argument("--rows", type=int, default=2500, help="Number of customer rows (minimum 100)")
    parser.add_argument("--output", type=str, default=None, help="Optional output CSV path")
    args = parser.parse_args()
    path, frame = save_demo_dataset(output_path=args.output, n_customers=args.rows)
    print(f"Created {len(frame):,} synthetic customer records: {path}")
    print(f"Synthetic churn rate: {frame['churned'].mean():.1%}")


if __name__ == "__main__":
    main()
