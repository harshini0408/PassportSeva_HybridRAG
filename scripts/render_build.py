"""Render build step for the Passport Seva backend.

Render build command (service root = repository root):

    python scripts/render_build.py

What it does, in order:
  1. checks the committed dataset (manifest, seed records, appointment PDF) is present;
  2. installs the CPU-only torch wheel, then backend/requirements-deploy.txt;
  3. downloads the embedding model and the (optional) reranker into a model cache
     inside the project directory, which Render keeps after the build;
  4. builds the BM25 + FAISS index from the committed dataset;
  5. verifies, with Hugging Face forced OFFLINE, that the index and both models load
     from that cache and that a hybrid query returns evidence.

The model cache is ``$HF_HOME`` if set, else ``<repo>/.cache/huggingface``. Set the same
``HF_HOME`` for the web service at runtime (see README, "Deploying to Render").
Nothing here reads or needs ``backend/.env`` or the Groq key.

Only the standard library is imported at module level so the script can run before
any dependency is installed.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
REQUIREMENTS = BACKEND_DIR / "requirements-deploy.txt"
TORCH_VERSION = "2.15.0"
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
REQUIRED_DATA = (
    "source_manifest.json",
    "seed_records.jsonl",
    "raw/appointment.pdf",
)


class BuildError(RuntimeError):
    pass


def model_cache_dir(root: Path = REPO_ROOT, environ: dict[str, str] | None = None) -> Path:
    env = os.environ if environ is None else environ
    return Path(env.get("HF_HOME") or root / ".cache" / "huggingface")


def data_dir(root: Path = REPO_ROOT, environ: dict[str, str] | None = None) -> Path:
    env = os.environ if environ is None else environ
    return Path(env.get("PASSPORT_DATA_DIR") or root / "data" / "passport_seva")


def check_dataset(data: Path) -> list[str]:
    """Return the committed dataset files that are missing (empty list = OK)."""
    return [name for name in REQUIRED_DATA if not (data / name).is_file()]


def pip_commands(python: str = sys.executable, requirements: Path = REQUIREMENTS) -> list[list[str]]:
    return [
        [python, "-m", "pip", "install", f"torch=={TORCH_VERSION}", "--index-url", TORCH_CPU_INDEX],
        [python, "-m", "pip", "install", "-r", str(requirements)],
    ]


def build_env(cache: Path, environ: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if environ is None else environ)
    env["HF_HOME"] = str(cache)
    env.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.pop("HF_HUB_OFFLINE", None)  # the download step needs the network
    return env


def run(cmd: list[str], env: dict[str, str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, env=env, cwd=cwd)
    if result.returncode != 0:
        raise BuildError(f"command failed with exit code {result.returncode}: {' '.join(cmd)}")


def download_models(env: dict[str, str]) -> None:
    """Fetch the embedder and reranker into HF_HOME (names come from the app's own settings)."""
    code = (
        "from app.rag.config import load_settings;"
        "from sentence_transformers import SentenceTransformer, CrossEncoder;"
        "s = load_settings();"
        "SentenceTransformer(s.embedding_model);"
        "CrossEncoder(s.reranker_model);"
        "print('models cached:', s.embedding_model, '+', s.reranker_model)"
    )
    run([sys.executable, "-c", code], env, cwd=BACKEND_DIR)


def build_index(env: dict[str, str]) -> None:
    run([sys.executable, "-m", "app.rag.cli", "build"], env, cwd=BACKEND_DIR)


def verify(cache: Path) -> dict[str, object]:
    """Load everything with Hugging Face OFFLINE; raise if anything is missing."""
    os.environ["HF_HOME"] = str(cache)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if str(BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_DIR))
    from app.rag.config import load_settings  # noqa: PLC0415
    from app.rag.retrieve import make_retriever  # noqa: PLC0415

    r = make_retriever(load_settings())
    result = r.search("I can't make it on the day I booked. Can I move it?", mode="hybrid")
    if not result.hits:
        raise BuildError("verification query returned no evidence")
    r.reranker.predict([("warm up", "warm up")])  # proves the reranker is in the cache too
    top = result.hits[0].chunk
    return {"chunks": r.idx.meta["chunk_count"], "top_hit": top.chunk_id,
            "embedding_model": r.idx.meta["embedding_model"]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-install", action="store_true", help="skip pip (dependencies already installed)")
    ap.add_argument("--verify-only", action="store_true", help="only run the offline verification")
    args = ap.parse_args(argv)

    cache = model_cache_dir()
    missing = check_dataset(data_dir())
    if missing:
        raise BuildError(f"committed dataset files missing from the checkout: {missing}")
    env = build_env(cache)
    print(f"repo root : {REPO_ROOT}\nmodel cache: {cache}", flush=True)

    if not args.verify_only:
        if not args.skip_install:
            for cmd in pip_commands():
                run(cmd, env)
        download_models(env)
        build_index(env)
    info = verify(cache)
    print("build verified offline:", info, flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BuildError as exc:
        print(f"BUILD FAILED: {exc}", file=sys.stderr)
        sys.exit(1)

