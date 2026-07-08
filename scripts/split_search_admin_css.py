# -*- coding: utf-8 -*-
"""style.css에서 search.css / admin.css 분할."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
style_path = ROOT / "static" / "style.css"
lines = style_path.read_text(encoding="utf-8").splitlines(keepends=True)

# 0-indexed: 검색 섹션 269-896, 관리자 897-2206 (대략)
search_block = "".join(lines[269:897])
admin_block = "".join(lines[897:2207])

css_dir = ROOT / "static" / "css"
css_dir.mkdir(parents=True, exist_ok=True)
(css_dir / "search.css").write_text(
    "/* 검색 UI */\n" + search_block,
    encoding="utf-8",
)
(css_dir / "admin.css").write_text(
    "/* 관리자 UI */\n" + admin_block,
    encoding="utf-8",
)

header = lines[:7]  # title + existing @imports
rest = lines[269:897]  # will remove search+admin from main - keep header through layout before search
# Rebuild: header imports + body without extracted blocks
before_search = lines[7:269]
after_admin = lines[2207:]
new_lines = header + [
    '@import url("css/search.css");\n',
    '@import url("css/admin.css");\n',
    "\n",
] + before_search + after_admin

style_path.write_text("".join(new_lines), encoding="utf-8")
print(f"search.css + admin.css extracted; style.css -> {len(new_lines)} lines")