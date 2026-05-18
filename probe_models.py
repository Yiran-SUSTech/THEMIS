import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remote model probing has been removed.")
    parser.add_argument("image", nargs="?", help="Unused. Kept for backward CLI compatibility.")
    parser.add_argument("models", nargs="*", help="Unused.")
    parser.add_argument("--output", default="model_probe_matrix.json", help="Unused.")
    return parser.parse_args()


def main() -> None:
    parse_args()
    raise SystemExit("Remote model probing has been removed. The project now runs local models only.")


if __name__ == "__main__":
    main()
