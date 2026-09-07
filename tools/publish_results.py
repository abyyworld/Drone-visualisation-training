#!/usr/bin/env python3
"""Push a finished training run back to the repo, so nothing needs downloading by hand.

WHY
    A Kaggle committed run survives your laptop, but its output sits on the Version page
    until someone clicks it. If the run finishes while you are offline, the model is safe
    but not anywhere useful. This commits the exported model and its metrics straight to
    the working branch, so when you come back online it is already in the repo.

WHAT IT PUSHES
    web/models/<name>.onnx, web/models/manifest.json, and docs/metrics-<name>.json.
    Not the .pt checkpoint - that stays on the Version page, since it is large and only
    the ONNX is what the site loads.

SAFETY
    Pushes to the current branch, never to main. GitHub Pages deploys only from main
    (see .github/workflows/pages.yml), so a model landing here CANNOT change the live
    site. Merging is a decision you make after looking at the numbers.

    Without a token it prints what to do and exits 0, so it never fails a training run.

SETUP (once)
    1. github.com -> Settings -> Developer settings -> Personal access tokens
       -> Fine-grained tokens -> Generate new token
       Repository access: only this repo.  Permissions: Contents -> Read and write.
    2. Kaggle notebook -> Add-ons -> Secrets -> add GITHUB_TOKEN with that value.

Usage:
    python3 tools/publish_results.py --name turbine --metrics /kaggle/working/runs/eval_v2
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = "github.com/abyyworld/Drone-visualisation-training.git"


def get_token() -> str | None:
    """Kaggle Secrets first, then the environment, so this also runs outside Kaggle."""
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("GITHUB_TOKEN")
    except Exception:
        return os.environ.get("GITHUB_TOKEN") or None


def scrub(text: str, secret: str) -> str:
    """Never let a token reach stdout - git puts the remote URL in its error messages."""
    return text.replace(secret, "***") if secret else text


def run(cmd: list[str], token: str = "", check: bool = True) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = scrub(result.stdout + result.stderr, token)
    if check and result.returncode != 0:
        raise SystemExit(f"failed: {' '.join(cmd[:3])}...\n{output}")
    return output.strip()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default="turbine", help="model name in the manifest")
    parser.add_argument("--metrics", type=Path, help="evaluate.py --out directory")
    args = parser.parse_args()

    model = Path("web/models") / f"{args.name}.onnx"
    if not model.exists():
        raise SystemExit(f"{model} not found - run tools/export_onnx.py first")

    token = get_token()
    if not token:
        print("No GITHUB_TOKEN found, so nothing was pushed. The model is still in this")
        print(f"run's Output at {model} and can be downloaded by hand.")
        print("\nTo have future runs publish themselves, see the setup notes in")
        print("tools/publish_results.py.")
        return 0

    paths = [str(model), "web/models/manifest.json"]

    # Metrics belong in the repo too: a model whose score lives only in a scrolled-past log
    # is a model nobody can honestly quote later.
    source = args.metrics / "metrics.json" if args.metrics else None
    if source and source.exists():
        target = Path("docs") / f"metrics-{args.name}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source, target)
        paths.append(str(target))
        summary = json.loads(source.read_text())
        per_class = summary.get("per_class", {})
        detail = ", ".join(f"{k} {v['mAP50']:.3f}" for k, v in per_class.items())
    else:
        detail = "metrics not captured"

    branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    if branch == "main":
        raise SystemExit("refusing to push to main - that deploys the live site")

    run(["git", "config", "user.name", "abyyworld"])
    run(["git", "config", "user.email", "annolieberto@gmail.com"])

    # Credentials in a file outside the working tree, removed immediately after, so the
    # token never enters the repo, the remote URL, or the run log.
    cred = Path("/tmp/.git-credentials")
    cred.write_text(f"https://x-access-token:{token}@github.com\n")
    try:
        run(["git", "config", "credential.helper", f"store --file={cred}"])
        run(["git", "remote", "set-url", "origin", f"https://{REPO}"])
        run(["git", "add"] + paths)

        if not run(["git", "status", "--porcelain"] + paths):
            print("Nothing changed - the committed model is already identical.")
            return 0

        size = model.stat().st_size / 1e6
        run(["git", "commit", "-m",
             f"Publish {args.name} model from a Kaggle run ({size:.1f} MB)\n\n"
             f"Per-class mAP50: {detail}\n\n"
             f"Pushed automatically by tools/publish_results.py. This branch is not the\n"
             f"Pages source, so the live site is unchanged until this is merged."])
        print(scrub(run(["git", "push", "origin", branch], token), token))
    finally:
        cred.unlink(missing_ok=True)
        run(["git", "config", "--unset", "credential.helper"], check=False)

    print(f"\nPushed to {branch}. Pages deploys from main only, so the live site is unchanged.")
    print(f"Per-class mAP50: {detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
