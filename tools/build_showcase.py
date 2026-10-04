"""Create a separate Git root from an explicit source-only allowlist. Never push."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
TOP_LEVEL = (
    ".gitignore", "README.md", "config.py", "main.py", "demo.py", "run_demo.sh",
    "requirements.txt", "requirements-demo.txt", "requirements-demo-lock.txt",
)
PATTERNS = (
    "pipeline/*.py", "utils/*.py", "scripts/*.py", "rules/*.yml",
    "demo_support/*.py", "tests/*.py", "docs/*.md",
)
EXTRA = (
    "validation/before.txt", "validation/after.txt", "validation/environment.txt",
    "examples/synthetic_reviews.json", "examples/sample-result.json",
    "examples/sample-report.html", "docs/publication-audit.json",
    "tools/reproduce_legacy_bugs.py", "tools/build_showcase.py",
    "tools/merge_all.py", "tools/resolve_rules.py",
)


def build(destination: Path) -> None:
    destination = destination.resolve()
    if destination == ROOT or ROOT in destination.parents:
        raise ValueError("Showcase must be separate from the source checkout")
    if destination.exists():
        raise FileExistsError("Destination already exists; refusing to overwrite it")
    candidates = {ROOT / name for name in TOP_LEVEL + EXTRA}
    for pattern in PATTERNS:
        candidates.update(ROOT.glob(pattern))
    required = [ROOT / name for name in TOP_LEVEL]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")
    files = sorted(p for p in candidates if p.is_file())
    destination.mkdir(parents=True)
    manifest = {}
    for source in files:
        if source.is_symlink():
            raise ValueError("Symlinks cannot enter the showcase")
        relative = source.relative_to(ROOT)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        manifest[relative.as_posix()] = hashlib.sha256(target.read_bytes()).hexdigest()
    (destination / "run_demo.sh").chmod(0o755)
    (destination / "showcase-manifest.json").write_text(
        json.dumps({"history": "new independent root", "sha256": manifest}, indent=2) + "\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-b", "main", str(destination)], check=True)
    for key, value in (
        ("user.name", "bglassm"),
        ("user.email", "99392414+bglassm@users.noreply.github.com"),
    ):
        subprocess.run(["git", "-C", str(destination), "config", key, value], check=True)
    subprocess.run(["git", "-C", str(destination), "add", "--all"], check=True)
    subprocess.run([
        "git", "-C", str(destination), "-c", "commit.gpgsign=false", "commit", "-m",
        "Create synthetic Korean review showcase with independent history",
    ], check=True)
    print("Independent root committed; no original data, original Git objects, or push performed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    build(parser.parse_args().destination)
