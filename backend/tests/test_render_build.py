"""Deployment readiness: the Render build script, startup warm-up and Git hygiene."""
from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from app.main import create_app
from app.service import AskService
from app.settings import AppSettings, load_app_settings

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("render_build", REPO / "scripts" / "render_build.py")
rb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rb)  # type: ignore[union-attr]


# ------------------------------------------------------------------ build script
def test_model_cache_defaults_inside_the_project_and_honours_hf_home(tmp_path):
    assert rb.model_cache_dir(tmp_path, {}) == (tmp_path / ".cache" / "huggingface").resolve()
    assert rb.model_cache_dir(tmp_path, {"HF_HOME": str(Path("/x/y"))}) == Path("/x/y").resolve()
    assert rb.model_cache_dir(tmp_path, {"HF_HOME": ""}) == (tmp_path / ".cache" / "huggingface").resolve()
    # Relative path is resolved to absolute relative to root:
    assert rb.model_cache_dir(tmp_path, {"HF_HOME": ".cache/huggingface"}) == (tmp_path / ".cache" / "huggingface").resolve()


def test_default_cache_is_not_the_user_profile_cache():
    cache = rb.model_cache_dir(rb.REPO_ROOT, {})
    assert rb.REPO_ROOT in cache.parents


def test_build_env_forces_cache_and_clears_offline():
    env = rb.build_env(Path("/c"), {"HF_HUB_OFFLINE": "1", "HF_HOME": "/other", "PATH": "p"})
    assert env["HF_HOME"] == str(Path("/c").resolve())
    assert "HF_HUB_OFFLINE" not in env
    assert env["PATH"] == "p"


def test_pip_installs_cpu_torch_before_the_pinned_requirements():
    cmds = rb.pip_commands("py")
    assert cmds[0][:4] == ["py", "-m", "pip", "install"]
    assert f"torch=={rb.TORCH_VERSION}" in cmds[0]
    assert cmds[0][cmds[0].index("--index-url") + 1] == "https://download.pytorch.org/whl/cpu"
    assert cmds[1][-2:] == ["-r", str(rb.REQUIREMENTS)]


def test_deploy_requirements_are_pinned_and_runtime_only():
    lines = [l.strip() for l in rb.REQUIREMENTS.read_text().splitlines()
             if l.strip() and not l.lstrip().startswith("#")]
    assert lines and all(re.fullmatch(r"[A-Za-z0-9_.-]+==[0-9][^\s]*", l) for l in lines), lines
    names = {l.split("==")[0].lower() for l in lines}
    assert not names & {"pytest", "torch"}
    assert {"fastapi", "uvicorn", "faiss-cpu", "sentence-transformers", "groq", "pypdf"} <= names


def test_committed_dataset_is_complete_and_missing_files_are_reported(tmp_path):
    assert rb.check_dataset(rb.data_dir(rb.REPO_ROOT, {})) == []
    assert rb.check_dataset(tmp_path) == list(rb.REQUIRED_DATA)


def test_build_stops_on_a_failing_step():
    with pytest.raises(rb.BuildError):
        rb.run([rb.sys.executable, "-c", "raise SystemExit(3)"], {})


# ---------------------------------------------------------------- startup warm-up
class _FakeIndex:
    meta = {"chunk_count": 1, "embedding_model": "m", "similarity_metric": "s"}


class _FakeRetriever:
    def __init__(self):
        self.idx = _FakeIndex()
        self.semantic_calls = 0
        self.reranker_loads = 0

    def semantic(self, q, k):
        self.semantic_calls += 1
        return []

    @property
    def reranker(self):
        self.reranker_loads += 1

        class _CE:
            def predict(self, pairs):
                return [0.0]
        return _CE()


class _NoLLM:
    def answer(self, *a, **k):  # pragma: no cover - never called
        raise AssertionError


@pytest.mark.parametrize("warm_reranker,expected", [(False, 0), (True, 1)])
def test_startup_loads_the_embedder_but_the_reranker_only_when_requested(warm_reranker, expected):
    fake = _FakeRetriever()
    svc = AskService(AppSettings(warm_reranker=warm_reranker), _NoLLM(), retriever_factory=lambda _s: fake)
    assert svc.load(warm=True) is True
    assert fake.semantic_calls == 1          # hybrid needs the embedder at startup
    assert fake.reranker_loads == expected   # hybrid never needs the cross-encoder


def test_warm_reranker_setting_comes_from_the_environment(monkeypatch):
    monkeypatch.delenv("PASSPORT_WARM_RERANKER", raising=False)
    assert load_app_settings().warm_reranker is False
    monkeypatch.setenv("PASSPORT_WARM_RERANKER", "1")
    assert load_app_settings().warm_reranker is True


def test_default_mode_stays_hybrid_and_health_path_is_unchanged():
    assert AppSettings().default_mode == "hybrid"
    paths = {r.path for r in create_app(service=AskService(AppSettings(), _NoLLM(),
                                                          retriever_factory=lambda _s: _FakeRetriever()),
                                        warm=False).routes}
    assert {"/api/health", "/api/ask"} <= paths


# ------------------------------------------------------------------- Git hygiene
git = shutil.which("git")
pytestmark_git = pytest.mark.skipif(git is None or not (REPO / ".git").exists(), reason="no git checkout")


def _git(*args):
    return subprocess.run([git, *args], cwd=REPO, capture_output=True, text=True)


@pytestmark_git
@pytest.mark.parametrize("path", [
    "backend/.env", "backend/.venv/Scripts/python.exe", "backend/.env.production",
    "Passport-Seva-Application-Guide-debug.apk", "frontend/android/app/build/outputs/apk/debug/app-debug.apk",
    "frontend/android/build/x", "frontend/android/.gradle/x", "frontend/android/local.properties",
    "frontend/node_modules/x", "frontend/dist/index.html", "data/passport_seva/index/faiss.index",
    ".cache/huggingface/hub/x", "release.aab",
])
def test_secrets_builds_and_caches_are_ignored(path):
    assert _git("check-ignore", "-q", path).returncode == 0, f"{path} is not ignored"


@pytestmark_git
@pytest.mark.parametrize("path", [
    "data/passport_seva/raw/appointment.pdf", "data/passport_seva/seed_records.jsonl",
    "data/passport_seva/source_manifest.json", "data/passport_seva/eval_questions.jsonl",
    "backend/.env.example", "backend/requirements-deploy.txt", "scripts/render_build.py",
])
def test_deployment_inputs_are_not_ignored(path):
    assert _git("check-ignore", "-q", path).returncode == 1, f"{path} would be excluded from Git"

