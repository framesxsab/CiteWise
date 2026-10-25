# CiteWise online deployment status

Updated 2026-10-25. **No online deployment has been created or verified.** The GitHub repository publication is complete; it does not establish a running application endpoint.

## Current evidence

| Check | Result |
|---|---|
| Deployment target in repository | Hugging Face Docker Space template, port 8080 |
| Docker client | 29.5.3 installed |
| Docker Engine | Unavailable: `npipe:////./pipe/docker_engine` does not exist; image build not run |
| Hugging Face CLI/token | No CLI installed and no Hugging Face token configured in this execution environment |
| Space account/name | No Hugging Face namespace or Space ID configured |
| Local app data | `data/processed/validation.jsonl` exists locally (8,027,113 bytes); it is excluded from the Docker build context and must be mounted at `/data/processed` |
| Public HTTPS health/search flow | Not tested; no endpoint was supplied or created |

The target platform's current documentation says Docker Spaces need a paid plan to create, although CPU Basic hardware itself has no hourly compute charge. Protected visibility also requires PRO, Team, or Enterprise. Confirm plan eligibility and current terms in the account before creating a Space. [Spaces overview and current hardware/plan requirements](https://huggingface.co/docs/hub/spaces-overview). The app's Docker metadata uses `sdk: docker` and `app_port: 8080`, as supported by [Docker Spaces documentation](https://huggingface.co/docs/hub/spaces-sdks-docker).

## Existing deployment configuration

- `Dockerfile` builds one model at a pinned Hugging Face revision, uses CPU, sets `HF_HUB_OFFLINE=1` for runtime, starts one Uvicorn worker on port 8080, and health-checks `/api/health`.
- `deploy/huggingface/README.md` contains the Space metadata template and setup instructions.
- `CITEWISE_DATA_DIR=/data/processed` expects `validation.jsonl` at runtime. The prepared validation JSONL is real QASPER data; keep its attribution and revision with any hosted copy.
- `CITEWISE_EMBED_CACHE_DIR=/data/cache/embeddings` should point to a persistent, writable bucket volume. Hugging Face documents attached bucket volumes for Spaces; `/data` volumes are runtime-only and unavailable during image build. [Spaces storage](https://huggingface.co/docs/hub/main/spaces-storage), [volume management](https://huggingface.co/docs/huggingface_hub/main/guides/manage-spaces).
- Run `python -m citewise.prepare_embeddings --model e5-base-v2 --split validation --data-dir /data/processed --cache-dir /data/cache/embeddings --output /data/cache/embedding-precompute.json` against the mounted dataset and bucket before accepting traffic. This step has not been run in the hosted environment.

## Required to verify deployment

1. In the Hugging Face account that will own the app, create or identify a Docker Space and confirm an eligible plan. Choose public visibility only if publishing the source is intended; protected app access requires a supported paid organization/account plan.
2. Use the metadata in `deploy/huggingface/README.md` in the Space's root `README.md`; preserve its YAML front matter (`sdk: docker`, `app_port: 8080`). Include the app source, Dockerfile, and Python project metadata in that Space repository.
3. Attach the prepared validation data at `/data/processed` and a writable Storage Bucket at `/data/cache`. Set runtime variables `CITEWISE_DATA_DIR=/data/processed` and `CITEWISE_EMBED_CACHE_DIR=/data/cache/embeddings`. Build argument `CITEWISE_DEFAULT_MODEL=e5-base-v2` is the validation-selected quality-first choice; MiniLM is the speed-oriented alternative.
4. Wait for the platform build and runtime to report ready. Verify `https://<space-subdomain>.hf.space/api/health`, then run an actual search using a validation paper/question and confirm the evidence IDs/text match the mounted source. Verify the public browser flow at desktop and mobile widths, keyboard use, and axe before claiming deployment readiness.

No Hugging Face token, Space name, paid-plan eligibility, or Docker Engine is available here, so I stopped before creating a hosted resource or incurring a subscription/compute charge. No model, dataset split, frozen benchmark artifact, or held-out result was changed.
