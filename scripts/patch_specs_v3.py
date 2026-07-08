# -*- coding: utf-8 -*-
"""v3 리팩토링 반영: spec hiddenimports 보강."""
from __future__ import annotations

import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAG_BLOCK = """
# === RAG v3 (2026-07) ===
try:
    from PyInstaller.utils.hooks import collect_submodules as _csm
    hiddenimports += _csm('rag')
except Exception:
    pass
for _m in ('httpx', 'langgraph', 'langgraph.graph'):
    if _m not in hiddenimports:
        hiddenimports.append(_m)
"""

LITE_PATH_PATCHES = {
    "app.services.search": "app.services.search",
    "app.routes.api_files": "app.routes.api_files",
}
LITE_ADDITIONS = [
    "app.services.files",
    "app.routes.api_tags",
    "app.routes.api_revisions",
    "app.routes.files_request",
    "app.services.search.index",
    "app.services.search.hybrid_search",
    "app.services.search.qa_facade",
]


def patch_full_spec(path: str) -> bool:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if "collect_submodules('rag')" in text or '_csm("rag")' in text:
        return False
    if "collect_submodules('app')" not in text and 'collect_submodules("app")' not in text:
        if "hiddenimports = [" not in text:
            return False
    marker = "hiddenimports += collect_submodules('app')"
    if marker not in text:
        marker = 'hiddenimports += collect_submodules("app")'
    if marker in text and "RAG v3" not in text:
        text = text.replace(marker, marker + RAG_BLOCK, 1)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        return True
    # server.spec style: append before Analysis
    if "RAG v3" not in text and path.endswith("server.spec"):
        text = text.replace(
            "a = Analysis(",
            RAG_BLOCK + "\na = Analysis(",
            1,
        )
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        return True
    return False


def patch_lite_spec(path: str) -> bool:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    changed = False
    for mod in LITE_ADDITIONS:
        line = f"    '{mod}',\n"
        if mod not in text:
            # insert before closing ] of hiddenimports list near app.services
            anchor = "    'app.services.parsers.hwp_models',\n"
            if anchor in text:
                text = text.replace(anchor, anchor + line, 1)
                changed = True
    if changed:
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
    return changed


def main() -> None:
    full_specs = [
        "regulation_search_gui.spec",
        "regulation_search.spec",
        "regulation_search_onefile.spec",
        "internal_regulations.spec",
        "server_gui.spec",
        "server.spec",
    ]
    lite_specs = [
        "regulation_search_ultra_lite_gui.spec",
        "regulation_search_ultra_lite.spec",
        "regulation_search_lite.spec",
        "internal_regulations_lite.spec",
    ]
    for name in full_specs:
        p = os.path.join(ROOT, name)
        if os.path.exists(p) and patch_full_spec(p):
            print(f"patched full: {name}")
    for name in lite_specs:
        p = os.path.join(ROOT, name)
        if os.path.exists(p) and patch_lite_spec(p):
            print(f"patched lite: {name}")


if __name__ == "__main__":
    main()