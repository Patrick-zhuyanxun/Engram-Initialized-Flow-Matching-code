#!/usr/bin/env python3
"""Build the HFRVLA experiment briefing through open-slide.

The editable deck lives in:
  docs/presentations/hfrvla-training-open-slide/slides/hfrvla-training/index.tsx

This script first checks that the open-slide workspace builds, then bundles a
small standalone viewer for the same open-slide deck source into
docs/hfrvla_experiment_briefing.html. The standalone viewer avoids open-slide's
BrowserRouter so the final HTML also works from file://. The old
docs/training_presentation.html path is retained as a compatibility redirect.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import mimetypes
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_HTML = REPO_ROOT / "docs/hfrvla_experiment_briefing.html"
LEGACY_OUTPUT_HTML = REPO_ROOT / "docs/training_presentation.html"
OPEN_SLIDE_ROOT = REPO_ROOT / "docs/presentations/hfrvla-training-open-slide"
SLIDE_SOURCE = OPEN_SLIDE_ROOT / "slides/hfrvla-training/index.tsx"
EVAL_MASTER = REPO_ROOT / "experiments/eval_registry/eval_results_master.csv"
OPEN_SLIDE_CHECK_BUILD_DIR = OPEN_SLIDE_ROOT / "dist-open-slide-check"
STANDALONE_BUILD_DIR = OPEN_SLIDE_ROOT / "dist-experiment-briefing"


def _ensure_open_slide_dependencies() -> None:
    if not (OPEN_SLIDE_ROOT / "node_modules/@open-slide/core").exists():
        raise RuntimeError(
            f"open-slide dependencies are missing; run `npm install` in {OPEN_SLIDE_ROOT}"
        )


def _run_open_slide_build_check() -> None:
    _ensure_open_slide_dependencies()
    shutil.rmtree(OPEN_SLIDE_CHECK_BUILD_DIR, ignore_errors=True)
    subprocess.run(
        ["npm", "run", "build", "--", "--out-dir", OPEN_SLIDE_CHECK_BUILD_DIR.name],
        cwd=OPEN_SLIDE_ROOT,
        check=True,
    )


def _run_standalone_build() -> None:
    _ensure_open_slide_dependencies()
    shutil.rmtree(STANDALONE_BUILD_DIR, ignore_errors=True)

    build_script = f"""
import path from 'node:path';
import react from '@vitejs/plugin-react';
import {{ build }} from 'vite';

await build({{
  root: process.cwd(),
  configFile: false,
  plugins: [react()],
  resolve: {{
    alias: {{
      '@assets': path.resolve(process.cwd(), 'assets'),
    }},
  }},
  build: {{
    outDir: path.resolve(process.cwd(), {str(STANDALONE_BUILD_DIR.name)!r}),
    emptyOutDir: true,
    rollupOptions: {{
      input: path.resolve(process.cwd(), 'standalone/index.html'),
      output: {{
        inlineDynamicImports: true,
      }},
    }},
  }},
}});
"""

    subprocess.run(
        ["node", "--input-type=module", "-e", build_script],
        cwd=OPEN_SLIDE_ROOT,
        check=True,
    )


def _asset_data_url(asset_path: Path) -> str:
    mime = mimetypes.guess_type(asset_path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(asset_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _inline_css_assets(css: str, assets_dir: Path) -> str:
    def replace_url(match: re.Match[str]) -> str:
        raw = match.group("url").strip("\"'")
        if raw.startswith("data:") or raw.startswith("http://") or raw.startswith("https://"):
            return match.group(0)
        if raw.startswith("/assets/"):
            asset_name = raw.removeprefix("/assets/")
        elif raw.startswith("./"):
            asset_name = raw.removeprefix("./")
        else:
            return match.group(0)
        asset_path = assets_dir / asset_name
        if not asset_path.exists():
            return match.group(0)
        return f"url({_asset_data_url(asset_path)})"

    return re.sub(r"url\((?P<url>[^)]+)\)", replace_url, css)


def _inline_js_assets(js: str, assets_dir: Path) -> str:
    def replace_string_asset(match: re.Match[str]) -> str:
        quote = match.group("quote")
        raw = match.group("url")
        if raw.startswith("/assets/"):
            asset_name = raw.removeprefix("/assets/")
        elif raw.startswith("./assets/"):
            asset_name = raw.removeprefix("./assets/")
        elif raw.startswith("assets/"):
            asset_name = raw.removeprefix("assets/")
        else:
            return match.group(0)
        asset_path = assets_dir / asset_name
        if not asset_path.exists():
            return match.group(0)
        return f"{quote}{_asset_data_url(asset_path)}{quote}"

    return re.sub(r'(?P<quote>["\'])(?P<url>(?:/|\./)?assets/[^"\']+)(?P=quote)', replace_string_asset, js)


def _find_single(pattern: str, text: str, label: str) -> str:
    matches = re.findall(pattern, text)
    if len(matches) != 1:
        raise RuntimeError(f"expected one {label} in open-slide build, found {len(matches)}")
    return matches[0]


def _bundle_html(slide_source: str) -> str:
    source_sha = hashlib.sha256(slide_source.encode("utf-8")).hexdigest()[:12]
    newest_source_mtime = max(SLIDE_SOURCE.stat().st_mtime, EVAL_MASTER.stat().st_mtime)
    source_updated_at = datetime.fromtimestamp(
        newest_source_mtime, timezone.utc
    ).astimezone().isoformat(timespec="seconds")

    index_path = STANDALONE_BUILD_DIR / "standalone/index.html"
    if not index_path.exists():
        index_path = STANDALONE_BUILD_DIR / "index.html"
    index_html = index_path.read_text(encoding="utf-8")
    css_hrefs = re.findall(r'<link rel="stylesheet"[^>]+href="([^"]+)"', index_html)
    js_src = _find_single(r'<script type="module"[^>]+src="([^"]+)"', index_html, "module script")
    favicon_match = re.search(r'<link rel="icon" href="([^"]+)"', index_html)

    assets_dir = STANDALONE_BUILD_DIR / "assets"
    js_path = STANDALONE_BUILD_DIR / js_src.lstrip("/")
    if not js_path.exists():
        raise RuntimeError("standalone presentation build output is missing expected JS asset")

    css_parts = []
    for css_href in css_hrefs:
        css_path = STANDALONE_BUILD_DIR / css_href.lstrip("/")
        if not css_path.exists():
            raise RuntimeError("standalone presentation build output is missing expected CSS asset")
        css_parts.append(_inline_css_assets(css_path.read_text(encoding="utf-8"), assets_dir))
    css = "\n".join(css_parts)
    js = _inline_js_assets(js_path.read_text(encoding="utf-8"), assets_dir)
    favicon = ""
    if favicon_match:
        icon_path = STANDALONE_BUILD_DIR / favicon_match.group(1).lstrip("/")
        if icon_path.exists():
            favicon = f'  <link rel="icon" href="{_asset_data_url(icon_path)}" />\n'

    return f"""<!doctype html>
<html lang="zh-Hant" data-source-sha="{source_sha}" data-built-with="open-slide">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <meta name="generator" content="open-slide @open-slide/core" />
  <meta name="hfrvla-source" content="docs/presentations/hfrvla-training-open-slide/slides/hfrvla-training/index.tsx" />
  <meta name="hfrvla-eval-registry" content="experiments/eval_registry/eval_results_master.csv" />
  <meta name="hfrvla-source-updated-at" content="{html.escape(source_updated_at)}" />
{favicon}  <title>HFRVLA Experiment Briefing</title>
  <style>{css}</style>
</head>
<body>
  <div id="root"></div>
  <script type="module">{js}</script>
</body>
</html>
"""


def _legacy_redirect_html() -> str:
    target = OUTPUT_HTML.name
    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <meta http-equiv="refresh" content="0; url={html.escape(target)}" />
  <title>HFRVLA Experiment Briefing</title>
  <style>
    body {{
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      background: #f7f3ea;
      color: #17211f;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    a {{ color: #0f9f7a; font-weight: 700; }}
  </style>
</head>
<body>
  <main>
    <p>HFRVLA briefing moved to <a href="{html.escape(target)}">{html.escape(target)}</a>.</p>
  </main>
</body>
</html>
"""


def generate(*, check: bool = False) -> None:
    slide_source = SLIDE_SOURCE.read_text(encoding="utf-8")

    _run_open_slide_build_check()
    _run_standalone_build()
    output = _bundle_html(slide_source)
    legacy_output = _legacy_redirect_html()
    if check:
        if not OUTPUT_HTML.exists() or OUTPUT_HTML.read_text(encoding="utf-8") != output:
            raise SystemExit(f"{OUTPUT_HTML} is out of date; run scripts/generate_training_presentation.py")
        if not LEGACY_OUTPUT_HTML.exists() or LEGACY_OUTPUT_HTML.read_text(encoding="utf-8") != legacy_output:
            raise SystemExit(
                f"{LEGACY_OUTPUT_HTML} is out of date; run scripts/generate_training_presentation.py"
            )
        print(f"[experiment-briefing] up to date via open-slide: {OUTPUT_HTML}")
        return

    OUTPUT_HTML.write_text(output, encoding="utf-8")
    LEGACY_OUTPUT_HTML.write_text(legacy_output, encoding="utf-8")
    print(f"[experiment-briefing] wrote open-slide bundle {OUTPUT_HTML}")
    print(f"[experiment-briefing] wrote compatibility redirect {LEGACY_OUTPUT_HTML}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Fail if the HTML is out of date.")
    args = parser.parse_args()
    generate(check=args.check)


if __name__ == "__main__":
    main()
