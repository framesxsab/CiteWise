import argparse
import json
from .dataset import prepare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--limit-per-split", type=int, default=None, help="Only for pipeline smoke tests, never final experiment")
    args = parser.parse_args()
    print(json.dumps(prepare(args.out, args.limit_per_split), indent=2))


if __name__ == "__main__":
    main()
