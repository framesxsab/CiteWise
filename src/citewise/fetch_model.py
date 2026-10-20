"""Stage one pinned, CPU-feasible model during image build or explicit setup."""
import argparse

from .retrieval import MODEL_IDS, MODEL_REVISIONS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(MODEL_REVISIONS))
    model = parser.parse_args().model
    from huggingface_hub import snapshot_download
    path = snapshot_download(MODEL_IDS[model], revision=MODEL_REVISIONS[model])
    print(f"Staged {model} at {MODEL_REVISIONS[model]}: {path}")


if __name__ == "__main__":
    main()
