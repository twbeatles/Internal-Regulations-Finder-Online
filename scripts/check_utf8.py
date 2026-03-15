from __future__ import annotations

from pathlib import Path


TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
    ".spec",
    ".json",
    ".html",
    ".css",
    ".js",
}

EXCLUDED_PARTS = {"__pycache__", ".git", ".pytest_cache"}


def iter_text_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in EXCLUDED_PARTS for part in path.parts):
            continue
        if path.suffix.lower() in TEXT_SUFFIXES:
            files.append(path)
    return files


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    failures: list[str] = []

    for path in iter_text_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            failures.append(f"{path.relative_to(root)}: invalid UTF-8 ({exc})")
            continue

        if "\ufffd" in text:
            failures.append(f"{path.relative_to(root)}: contains replacement character U+FFFD")

    if failures:
        for failure in failures:
            print(failure)
        return 1

    print("UTF-8 check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
