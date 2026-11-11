"""Command-line entry point."""
import argparse
from .config import ExperimentConfig
from .experiment import run_experiment


def main():
    parser = argparse.ArgumentParser(description="Run or resume a bounded distilled-RNN pilot")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    report = run_experiment(ExperimentConfig.from_file(args.config), args.output, args.device)
    if report["status"] != "complete":
        print(report["message"])
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
