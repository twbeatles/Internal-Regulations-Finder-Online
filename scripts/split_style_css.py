# -*- coding: utf-8 -*-
"""style.css에서 tokens/rag 분할 후 @import 연결."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
style_path = ROOT / "static" / "style.css"
lines = style_path.read_text(encoding="utf-8").splitlines(keepends=True)
header = lines[:5]
rest = lines[91:4365]
new_lines = header + [
    '@import url("css/tokens.css");\n',
    '@import url("css/rag.css");\n',
    "\n",
] + rest
style_path.write_text("".join(new_lines), encoding="utf-8")
print(f"updated {style_path} ({len(new_lines)} lines)")