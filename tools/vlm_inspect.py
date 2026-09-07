#!/usr/bin/env python3
"""Inspect a folder of drone images with a vision model, and keep the result as training data.

WHY THIS EXISTS
    The trained detector only knows the defect classes that were in its dataset, and only at
    the framing distance that dataset was shot at. It returned nothing on a turbine with a
    blade snapped in half, because "blade severed" was not a class and every defect image it
    learned from was a close-up. A general vision model has no class list to fall outside of.

    That buys recall today. It does not buy a model. The second half of this tool is the
    half that matters in a year: every inspection is written out as a label file in the
    layout tools/vlm_to_yolo.py expects, so the photographs an operator actually takes
    accumulate into a dataset that matches the job, instead of one bought off a website.
    Reviewed and corrected, that is the training set the detector should have had.

WHAT IT IS NOT
    A replacement for a detector in anything real-time. This makes one network round trip
    per image and costs money per image. The wildfire and crowd stations decode video at
    25 frames a second on a network with no internet at all (docs/DEPLOYMENT.md) - an API
    cannot be in that loop, and no amount of accuracy changes that.

    Its boxes are also approximate. A vision model localises well enough to point an
    engineer at the right part of the blade; it does not place a box the way a detector
    trained on boxes does. If you need tight boxes, the detector is still the answer.

PROVIDERS
    anthropic   ANTHROPIC_API_KEY   https://console.anthropic.com  -> API keys
    gemini      GEMINI_API_KEY      https://aistudio.google.com    -> Get API key
    openai      OPENAI_API_KEY      https://platform.openai.com    -> API keys

    Stdlib HTTP only, so this adds no dependency to the repo. Pillow is optional and only
    used to write annotated copies; without it everything else still runs.

USAGE
    export ANTHROPIC_API_KEY=...
    python3 tools/vlm_inspect.py photos/ --provider anthropic --domain turbine

    python3 tools/vlm_inspect.py photos/ --provider gemini --model gemini-2.5-pro \\
        --out inspections/2026-09-07-site-a

    python3 tools/vlm_inspect.py --list-models --provider openai

OUTPUT
    <out>/inspection_summary.json   the schema report_generator.py and the web app share
    <out>/raw/<image>.json          the provider's untouched reply, per image
    <out>/labels/<image>.json       normalised findings, the input to vlm_to_yolo.py
    <out>/annotated/<image>         boxes drawn on, when Pillow is installed

    Then:  python3 report_generator.py <out>/inspection_summary.json <out>/annotated out.pdf

COST
    One image is a few thousand input tokens. A hundred-image inspection is cents to low
    single-digit dollars depending on the model. --limit exists so you can price a run on
    five images before committing to five hundred.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROMPT_FILE = ROOT / "web" / "prompts" / "inspection.json"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Mirrors web/js/severity.js and score_to_label() in report_generator.py. Three copies of
# these numbers is two too many, but the web one has to run in a browser with no Python and
# the report one predates both, so for now they are kept identical and cross-referenced.
MINOR_BELOW = 2.0
MODERATE_BELOW = 5.0

ENV_KEYS = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "openai": ("OPENAI_API_KEY",),
}

DEFAULT_MODEL = {
    "anthropic": "claude-opus-5",
    "gemini": "gemini-2.5-pro",
    "openai": "gpt-5",
}


# ---------------------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------------------

class ProviderError(RuntimeError):
    """A provider said no, with a message that says what to do about it."""


def _request(url, *, method="GET", headers=None, payload=None, timeout=180):
    body = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        raise ProviderError(_explain(error)) from None
    except urllib.error.URLError as error:
        raise ProviderError(f"Could not reach the provider: {error.reason}") from None


def _explain(error: urllib.error.HTTPError) -> str:
    """Turn an HTTP status into something a person can act on.

    A bare "HTTP Error 401" sends people to check their network. Naming the actual fix -
    the key is wrong, the model ID is wrong, the account is out of credit - is the
    difference between a two-minute fix and an afternoon.
    """
    try:
        detail = json.loads(error.read().decode())
        detail = detail.get("error", {}).get("message") or detail.get("message") or str(detail)
    except Exception:
        detail = error.reason
    detail = str(detail)[:400]

    if error.code in (401, 403):
        return (
            f"The provider rejected the API key ({error.code}). Check it is the whole key, "
            f"that it belongs to this provider and not another, and that it is not revoked. {detail}"
        )
    if error.code == 404:
        return (
            f"The provider does not recognise that model ID (404). Run --list-models to see "
            f"what this key can reach. {detail}"
        )
    if error.code == 429:
        return (
            f"Rate-limited or out of credit (429). Wait and retry, or top up the account. {detail}"
        )
    if error.code >= 500:
        return f"Provider server error ({error.code}). Their side, not yours - retry shortly. {detail}"
    return f"Request failed ({error.code}). {detail}"


# ---------------------------------------------------------------------------------------
# Provider adapters
# ---------------------------------------------------------------------------------------

def _infer_anthropic(model, key, prompt, image):
    body = _request(
        "https://api.anthropic.com/v1/messages",
        method="POST",
        headers={
            "content-type": "application/json",
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
        },
        payload={
            "model": model,
            "max_tokens": 2000,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": image["media_type"], "data": image["b64"]}},
                    {"type": "text", "text": prompt},
                ],
            }],
        },
    )
    return "".join(b.get("text", "") for b in body.get("content", []) if b.get("type") == "text")


def _list_anthropic(key):
    body = _request(
        "https://api.anthropic.com/v1/models?limit=100",
        headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
    )
    return [(m["id"], m.get("display_name", m["id"])) for m in body.get("data", [])]


def _infer_gemini(model, key, prompt, image):
    body = _request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        method="POST",
        headers={"content-type": "application/json", "x-goog-api-key": key},
        payload={
            "contents": [{"role": "user", "parts": [
                {"inline_data": {"mime_type": image["media_type"], "data": image["b64"]}},
                {"text": prompt},
            ]}],
            "generationConfig": {"responseMimeType": "application/json", "maxOutputTokens": 2000},
        },
    )
    candidates = body.get("candidates") or []
    if not candidates:
        raise ProviderError(f"Gemini returned no candidate. Full response: {json.dumps(body)[:400]}")
    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts)


def _list_gemini(key):
    body = _request(
        "https://generativelanguage.googleapis.com/v1beta/models?pageSize=200",
        headers={"x-goog-api-key": key},
    )
    return [
        (m["name"].removeprefix("models/"), m.get("displayName", m["name"]))
        for m in body.get("models", [])
        if "generateContent" in m.get("supportedGenerationMethods", [])
    ]


def _infer_openai(model, key, prompt, image):
    body = _request(
        "https://api.openai.com/v1/chat/completions",
        method="POST",
        headers={"content-type": "application/json", "authorization": f"Bearer {key}"},
        payload={
            "model": model,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:{image['media_type']};base64,{image['b64']}", "detail": "high"}},
            ]}],
        },
    )
    choices = body.get("choices") or []
    if not choices:
        raise ProviderError(f"OpenAI returned no choice. Full response: {json.dumps(body)[:400]}")
    return choices[0].get("message", {}).get("content", "")


def _list_openai(key):
    body = _request("https://api.openai.com/v1/models",
                    headers={"authorization": f"Bearer {key}"})
    skip = ("audio", "realtime", "tts", "whisper", "embedding", "moderation", "image", "dall-e")
    return sorted(
        (m["id"], m["id"]) for m in body.get("data", [])
        if m["id"].startswith(("gpt-", "o1", "o3", "o4", "chatgpt"))
        and not any(word in m["id"] for word in skip)
    )


ADAPTERS = {
    "anthropic": (_infer_anthropic, _list_anthropic),
    "gemini": (_infer_gemini, _list_gemini),
    "openai": (_infer_openai, _list_openai),
}


# ---------------------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------------------

def parse_reply(text: str) -> dict:
    """Pull the JSON object out of a reply that may be fenced or padded with prose."""
    trimmed = (text or "").strip()
    if not trimmed:
        raise ValueError("The model returned an empty response.")

    if trimmed.startswith("```"):
        trimmed = trimmed.split("```")[1]
        if trimmed.startswith("json"):
            trimmed = trimmed[4:]
        trimmed = trimmed.strip()

    try:
        return json.loads(trimmed)
    except json.JSONDecodeError:
        pass

    start, end = trimmed.find("{"), trimmed.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(trimmed[start:end + 1])
        except json.JSONDecodeError:
            pass
    raise ValueError(f"The model did not return usable JSON. It said: {trimmed[:200]}")


def to_pixels(box, width, height):
    """Normalised fractions -> pixel box, clamped, with degenerate boxes rejected.

    Models answer in fractions, in 0-1000, or in raw pixels regardless of what they were
    asked for. Rescaling is better than discarding: the box is the useful part, and the
    convention it came in is guessable from its magnitude.
    """
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        values = [float(v) for v in box]
    except (TypeError, ValueError):
        return None
    if any(v != v or v in (float("inf"), float("-inf")) for v in values):
        return None

    largest = max(abs(v) for v in values)
    if 1.5 < largest <= 1000:
        values = [v / 1000 for v in values]
    elif largest > 1000:
        values = [values[0] / width, values[1] / height, values[2] / width, values[3] / height]

    clamp = lambda v: min(1.0, max(0.0, v))  # noqa: E731
    x0 = clamp(min(values[0], values[2])) * width
    x1 = clamp(max(values[0], values[2])) * width
    y0 = clamp(min(values[1], values[3])) * height
    y1 = clamp(max(values[1], values[3])) * height

    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)]


def tidy_label(raw) -> str:
    cleaned = "".join(c if c.isalnum() else "_" for c in str(raw or "defect").lower())
    cleaned = "_".join(part for part in cleaned.split("_") if part)[:40]
    return cleaned or "defect"


def score_to_label(score: float) -> str:
    if score <= 0:
        return "No defects found"
    if score < MINOR_BELOW:
        return "Minor wear"
    if score < MODERATE_BELOW:
        return "Moderate damage"
    return "Severe damage"


# ---------------------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------------------

def _pillow():
    try:
        from PIL import Image, ImageDraw  # noqa: F401
        return True
    except ImportError:
        return False


def read_image(path: Path, max_edge: int):
    """Base64 the image, downscaled if Pillow is available, and report its dimensions.

    Downscaling matters twice. Providers resize above roughly max_edge before the model
    sees the pixels, so a 48-megapixel original buys nothing and costs upload time and
    tokens; and re-encoding strips EXIF, including the GPS coordinates of the asset, which
    is not something to hand to a third party by accident.

    Without Pillow the original bytes are sent as they are. That still works - it is just
    slower, dearer, and it keeps the EXIF. The tool says so rather than failing.
    """
    media_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"

    if not _pillow():
        return {
            "b64": base64.b64encode(path.read_bytes()).decode(),
            "media_type": media_type,
            "width": None, "height": None, "resized": False,
        }

    import io
    from PIL import Image

    with Image.open(path) as source:
        source = source.convert("RGB")
        width, height = source.size
        scale = min(1.0, max_edge / max(width, height))
        if scale < 1.0:
            source = source.resize(
                (max(1, round(width * scale)), max(1, round(height * scale))), Image.LANCZOS)
        buffer = io.BytesIO()
        source.save(buffer, format="JPEG", quality=90)

    return {
        "b64": base64.b64encode(buffer.getvalue()).decode(),
        "media_type": "image/jpeg",
        "width": width, "height": height, "resized": scale < 1.0,
    }


PALETTE = ["#E5484D", "#F76B15", "#FFB224", "#30A46C", "#0091FF", "#8E4EC6", "#E93D82", "#00A2C7"]


def annotate(path: Path, detections, destination: Path) -> bool:
    """Draw the boxes onto a copy. Same palette as web/js/render.js, keyed the same way."""
    if not _pillow():
        return False
    from PIL import Image, ImageDraw

    with Image.open(path) as source:
        canvas = source.convert("RGB")
        draw = ImageDraw.Draw(canvas)
        line = max(2, round(max(canvas.size) / 250))
        for detection in detections:
            colour = PALETTE[detection["class_id"] % len(PALETTE)]
            draw.rectangle(detection["box"], outline=colour, width=line)
            draw.text((detection["box"][0] + line, detection["box"][1] + line),
                      f"{detection['label']} ({detection['certainty']})", fill=colour)
        destination.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(destination, quality=92)
    return True


def colour_index(label: str) -> int:
    """Same hash as colourIndex() in web/js/vlm.js, so a class keeps its colour in both."""
    value = 0
    for char in label:
        value = (value * 31 + ord(char)) % 4096
    return value


# ---------------------------------------------------------------------------------------
# One image
# ---------------------------------------------------------------------------------------

def inspect_one(path: Path, *, provider, model, key, prompt, doc, retries=2):
    infer = ADAPTERS[provider][0]
    image = read_image(path, doc["maxEdge"])

    last_error = None
    for attempt in range(retries + 1):
        try:
            raw = infer(model, key, prompt, image)
            break
        except ProviderError as error:
            last_error = error
            # A wrong key or a wrong model ID will not fix itself, and retrying a 429
            # immediately makes it worse. Only transient failures are worth a second go.
            if "rejected the API key" in str(error) or "does not recognise" in str(error):
                raise
            if attempt == retries:
                raise
            time.sleep(2 ** attempt * 2)
    else:  # pragma: no cover - the loop always breaks or raises
        raise last_error

    parsed = parse_reply(raw)

    # Pillow gives real dimensions; without it, fall back to the fractions themselves so
    # boxes stay usable and the caller can scale them later.
    width = image["width"] or 1000
    height = image["height"] or 1000

    detections, unlocated = [], []
    for item in parsed.get("findings") or []:
        label = tidy_label(item.get("label"))
        certainty = str(item.get("certainty", "medium")).lower()
        entry = {
            "label": label,
            "class_id": colour_index(label),
            "certainty": certainty,
            "confidence": doc["certainty"].get(certainty, doc["certainty"]["medium"]),
            "note": item.get("note") if isinstance(item.get("note"), str) else "",
        }
        box = to_pixels(item.get("box"), width, height)
        if box:
            detections.append({**entry, "box": box})
        else:
            unlocated.append(entry)

    asset = parsed.get("asset")
    return {
        "raw": raw,
        "asset": asset if asset in ("turbine", "solar", "crowd", "neither") else "neither",
        "asset_reason": parsed.get("asset_reason", "") or "",
        "overall": parsed.get("overall", "") or "",
        "detections": detections,
        "unlocated": unlocated,
        "width": image["width"],
        "height": image["height"],
    }


# ---------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------

def resolve_key(provider: str, supplied: str | None) -> str:
    if supplied:
        return supplied.strip()
    for name in ENV_KEYS[provider]:
        value = os.environ.get(name)
        if value:
            return value.strip()
    raise SystemExit(
        f"No API key for {provider}. Set {' or '.join(ENV_KEYS[provider])}, or pass --api-key.\n"
        f"Do not paste a key into a file that gets committed."
    )


def severity_weights(domain: str) -> dict:
    """Class weights from the web manifest, so a score means the same in both places.

    A vision model invents its own labels, so most of them will not be in the manifest and
    fall back to 1.0. The entries that do match - a crack outranking soiling - still apply,
    and structural words are weighted here because no manifest built for the old three-class
    detector has ever heard of a severed blade.
    """
    weights = {"broken": 4.0, "severed": 4.0, "missing": 4.0, "structural": 4.0,
               "hole": 3.0, "puncture": 3.0, "crack": 3.0, "lightning": 3.0,
               "corrosion": 2.0, "erosion": 2.0, "delamination": 2.0, "peeling": 2.0,
               "discoloration": 1.5, "dropping": 1.5, "dirt": 1.0, "soiling": 1.0}
    manifest = ROOT / "web" / "models" / "manifest.json"
    if manifest.exists():
        try:
            spec = json.loads(manifest.read_text()).get(domain, {})
            weights.update(spec.get("severityWeights") or {})
        except (json.JSONDecodeError, OSError):
            pass
    return weights


def weight_for(label: str, weights: dict) -> float:
    """Exact match first, then any weighted word contained in the label."""
    if label in weights:
        return weights[label]
    for word, weight in weights.items():
        if word in label:
            return weight
    return 1.0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See the module docstring for the full picture.",
    )
    parser.add_argument("images", nargs="?", type=Path,
                        help="an image, or a folder of images")
    parser.add_argument("--provider", choices=sorted(ADAPTERS), required=True)
    parser.add_argument("--model", help="defaults to this provider's most capable model")
    parser.add_argument("--api-key", help="prefer the environment variable")
    parser.add_argument("--domain", choices=["auto", "turbine", "solar", "crowd"], default="auto")
    parser.add_argument("--out", type=Path, help="output directory (default: inspections/<date>)")
    parser.add_argument("--limit", type=int, help="stop after N images, to price a run first")
    parser.add_argument("--list-models", action="store_true",
                        help="ask the provider what this key can reach, then exit")
    args = parser.parse_args(argv)

    key = resolve_key(args.provider, args.api_key)
    model = args.model or DEFAULT_MODEL[args.provider]

    if args.list_models:
        for identifier, label in ADAPTERS[args.provider][1](key):
            print(f"{identifier}\t{label}")
        return 0

    if not args.images:
        parser.error("give an image or a folder, or pass --list-models")

    if not PROMPT_FILE.exists():
        raise SystemExit(f"Missing {PROMPT_FILE}. It holds the prompt this tool and the web app share.")
    doc = json.loads(PROMPT_FILE.read_text())
    prompt = f"{doc['domains'].get(args.domain, doc['domains']['auto'])}\n\n{doc['schema']}"

    if args.images.is_dir():
        paths = sorted(p for p in args.images.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    elif args.images.is_file():
        paths = [args.images]
    else:
        raise SystemExit(f"No such path: {args.images}")
    if not paths:
        raise SystemExit(f"No images under {args.images} (looked for {', '.join(sorted(IMAGE_SUFFIXES))}).")
    if args.limit:
        paths = paths[:args.limit]

    out = args.out or ROOT / "inspections" / time.strftime("%Y-%m-%d-%H%M%S")
    for sub in ("raw", "labels", "annotated"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    if not _pillow():
        print("Pillow is not installed: images are sent at full size with EXIF intact, and no "
              "annotated copies are written. pip install Pillow", file=sys.stderr)

    weights_cache = {}
    results = []
    counts = {"severe": 0, "moderate": 0, "minor": 0, "healthy": 0}

    for index, path in enumerate(paths, 1):
        print(f"[{index}/{len(paths)}] {path.name}", flush=True)
        record = {"image": path.name, "source": str(path)}

        try:
            outcome = inspect_one(path, provider=args.provider, model=model, key=key,
                                  prompt=prompt, doc=doc)
        except (ProviderError, ValueError) as error:
            print(f"    error: {error}", file=sys.stderr)
            results.append({**record, "status": "error", "message": str(error),
                            "detections": [], "severity_label": None, "severity_score": None})
            continue

        (out / "raw" / f"{path.stem}.json").write_text(
            json.dumps({"model": model, "provider": args.provider, "reply": outcome["raw"]}, indent=2))

        if outcome["asset"] == "neither":
            print(f"    rejected: {outcome['asset_reason']}")
            results.append({**record, "status": "rejected",
                            "message": outcome["asset_reason"]
                            or "Not a turbine, a solar array or a crowd.",
                            "detections": [], "severity_label": None, "severity_score": None})
            continue

        domain = outcome["asset"]
        weights = weights_cache.setdefault(domain, severity_weights(domain))
        score = sum(weight_for(d["label"], weights) * d["confidence"] for d in outcome["detections"])
        label = score_to_label(score)

        counts["severe" if "Severe" in label else
               "moderate" if "Moderate" in label else
               "minor" if "Minor" in label else "healthy"] += 1

        annotated = annotate(path, outcome["detections"], out / "annotated" / path.name)

        # The label sidecar is deliberately separate from the summary: it is the training
        # record, it holds the image dimensions the boxes are relative to, and it survives
        # the summary being regenerated or the report being rebuilt.
        (out / "labels" / f"{path.stem}.json").write_text(json.dumps({
            "image": path.name,
            "source": str(path),
            "width": outcome["width"],
            "height": outcome["height"],
            "domain": domain,
            "provider": args.provider,
            "model": model,
            "prompt_version": doc.get("version"),
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "reviewed": False,
            "overall": outcome["overall"],
            "detections": outcome["detections"],
            "unlocated": outcome["unlocated"],
        }, indent=2))

        results.append({
            **record,
            "status": "analysed",
            "domain": domain,
            "severity_label": label,
            "severity_score": round(score, 3),
            "engine": {"provider": args.provider, "model": model},
            "notes": outcome["overall"],
            "annotated": annotated,
            "detections": [{"class": d["label"], "confidence": d["confidence"],
                            "certainty": d["certainty"], "note": d["note"], "bbox": d["box"]}
                           for d in outcome["detections"]],
            "unlocated": [{"class": d["label"], "certainty": d["certainty"], "note": d["note"]}
                          for d in outcome["unlocated"]],
        })

        found = len(outcome["detections"]) + len(outcome["unlocated"])
        print(f"    {domain}: {found} finding{'' if found == 1 else 's'} - {label} (score {score:.2f})")

    summary_path = out / "inspection_summary.json"
    summary_path.write_text(json.dumps({
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "engine": {"provider": args.provider, "model": model},
        "prompt_version": doc.get("version"),
        "summary": {"images": len(results), "counts": counts},
        "results": results,
    }, indent=2))

    analysed = sum(1 for r in results if r["status"] == "analysed")
    print(f"\n{analysed} of {len(results)} analysed. {counts}")
    print(f"Written to {out}")
    print(f"\nReport:  python3 report_generator.py {summary_path} {out / 'annotated'} report.pdf")
    print(f"Dataset: python3 tools/vlm_to_yolo.py {out.parent} --domain turbine")
    return 0 if analysed else 1


if __name__ == "__main__":
    sys.exit(main())
