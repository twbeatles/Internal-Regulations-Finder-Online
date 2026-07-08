# -*- coding: utf-8 -*-
"""document.py → app/services/document/ 패키지 분할 (1회성)."""
from __future__ import annotations

import os
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "app", "services", "document.py")
PKG = os.path.join(ROOT, "app", "services", "document")


def read_src() -> list[str]:
    with open(SRC, "r", encoding="utf-8") as f:
        return f.readlines()


def write(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)


def slice_lines(lines: list[str], start: int, end: int) -> str:
    return "".join(lines[start:end])


def main() -> None:
    lines = read_src()
    os.makedirs(PKG, exist_ok=True)

    patterns_header = textwrap.dedent(
        '''\
        # -*- coding: utf-8 -*-
        """문서 처리 공통 정규식 패턴."""
        import re

        _RE_ARTICLE_SPLIT = re.compile(r'(제\\s*\\d+\\s*조(?:의\\s*\\d+)?[^\\n]*)')
        _RE_ARTICLE_MATCH = re.compile(r'제\\s*(\\d+)\\s*조(?:의\\s*(\\d+))?(.+)?')
        _RE_PARAGRAPH_SPLIT = re.compile(r'([①②③④⑤⑥⑦⑧⑨⑩])')
        _RE_NUMBER_EXTRACT = re.compile(r'\\d+')
        _RE_CHAPTER_SPLIT = re.compile(r'(제\\s*\\d+\\s*장[^\\n]*)')
        _RE_ARTICLE_SPLIT_FULL = re.compile(r'(제\\s*\\d+\\s*조(?:의\\s*\\d+)?[^\\n]*)')
        _RE_KEYWORD_EXTRACT = re.compile(r'[가-힣]{2,}|[a-zA-Z]{3,}')
        '''
    )
    write(os.path.join(PKG, "patterns.py"), patterns_header)

    extractor_header = textwrap.dedent(
        '''\
        # -*- coding: utf-8 -*-
        """다양한 포맷 문서 텍스트 추출."""
        import os
        import re
        import logging
        import importlib
        from typing import Any, Dict, List, Optional, Tuple

        from app.utils import logger, FileUtils
        from app.config import AppConfig
        from app.constants import ErrorMessages, Patterns, Limits
        from app.services.parsers.hwp_adapter import HwpAdapter
        from app.services.parsers.hwpx_adapter import HwpxAdapter
        from app.services.parsers.hwp_models import (
            ExtractedDocument,
            build_basic_metadata,
            build_diagnostics,
        )
        from app.exceptions import (
            DocumentNotFoundError, DocumentExtractionError,
            DocumentTypeError, DocumentEmptyError
        )

        '''
    )
    write(os.path.join(PKG, "extractor.py"), extractor_header + slice_lines(lines, 49, 356))

    article_header = textwrap.dedent(
        '''\
        # -*- coding: utf-8 -*-
        """규정 문서 조문 파서."""
        import re
        from typing import Dict, List, Optional

        from app.services.document.patterns import (
            _RE_ARTICLE_MATCH,
            _RE_ARTICLE_SPLIT,
            _RE_NUMBER_EXTRACT,
            _RE_PARAGRAPH_SPLIT,
        )

        '''
    )
    write(os.path.join(PKG, "article_parser.py"), article_header + slice_lines(lines, 359, 452))

    splitter_header = textwrap.dedent(
        '''\
        # -*- coding: utf-8 -*-
        """문서 청킹·구조 분할."""
        import re
        from typing import Dict, List

        from app.constants import Limits

        '''
    )
    write(os.path.join(PKG, "splitter.py"), splitter_header + slice_lines(lines, 455, 598))

    comparator_header = textwrap.dedent(
        '''\
        # -*- coding: utf-8 -*-
        """문서 diff 비교."""
        from typing import Dict, Tuple

        '''
    )
    write(os.path.join(PKG, "comparator.py"), comparator_header + slice_lines(lines, 602, 650))

    highlighter_header = textwrap.dedent(
        '''\
        # -*- coding: utf-8 -*-
        """검색어 하이라이트."""
        import re
        from typing import List

        from app.services.document.patterns import _RE_KEYWORD_EXTRACT

        '''
    )
    write(os.path.join(PKG, "highlighter.py"), highlighter_header + slice_lines(lines, 653, len(lines)))

    write(
        os.path.join(PKG, "__init__.py"),
        textwrap.dedent(
            '''\
            # -*- coding: utf-8 -*-
            """문서 처리 패키지."""
            from app.services.document.article_parser import ArticleParser
            from app.services.document.comparator import DocumentComparator
            from app.services.document.extractor import DocumentExtractor
            from app.services.document.highlighter import TextHighlighter
            from app.services.document.splitter import DocumentSplitter

            __all__ = [
                "ArticleParser",
                "DocumentComparator",
                "DocumentExtractor",
                "DocumentSplitter",
                "TextHighlighter",
            ]
            '''
        ),
    )

    if os.path.exists(SRC):
        os.remove(SRC)
    print("document package created, old document.py removed")


if __name__ == "__main__":
    main()