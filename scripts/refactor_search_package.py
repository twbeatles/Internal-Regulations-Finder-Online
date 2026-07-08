# -*- coding: utf-8 -*-
"""search.py → app/services/search/ 패키지 분할 (1회성)."""
from __future__ import annotations

import os
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "app", "services", "search.py")
PKG = os.path.join(ROOT, "app", "services", "search")


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

    write(
        os.path.join(PKG, "langchain_lazy.py"),
        '# -*- coding: utf-8 -*-\n"""LangChain 지연 로딩."""\nimport importlib\nfrom typing import Any\n\n'
        + slice_lines(lines, 23, 64),
    )

    write(
        os.path.join(PKG, "bm25.py"),
        '# -*- coding: utf-8 -*-\n'
        + '"""BM25 경량 검색."""\n'
        + "import gc\nimport math\nimport re\nimport threading\nfrom collections import Counter\nfrom typing import Dict, List, Tuple\n\n"
        + "from app.constants import Patterns\n\n"
        + slice_lines(lines, 85, 244),
    )

    write(
        os.path.join(PKG, "cache.py"),
        '# -*- coding: utf-8 -*-\n'
        + '"""검색 LRU 캐시."""\n'
        + "import threading\nimport time\nfrom dataclasses import dataclass\nfrom typing import Any, Dict, Optional\n\n"
        + "from app.config import AppConfig\n"
        + "from app.utils import logger\n\n"
        + slice_lines(lines, 248, 428),
    )

    write(
        os.path.join(PKG, "rate_limiter.py"),
        '# -*- coding: utf-8 -*-\n'
        + '"""IP 기반 Rate Limiter."""\n'
        + "import threading\nimport time\nfrom typing import Any, Dict\n\n"
        + "from app.utils import logger\n\n"
        + slice_lines(lines, 429, 503),
    )

    write(
        os.path.join(PKG, "queue.py"),
        '# -*- coding: utf-8 -*-\n'
        + '"""동시 검색 큐."""\n'
        + "import threading\nfrom typing import Dict\n\n"
        + "from app.utils import logger\n\n"
        + slice_lines(lines, 504, 536),
    )

    write(
        os.path.join(PKG, "history.py"),
        '# -*- coding: utf-8 -*-\n'
        + '"""검색 히스토리."""\n'
        + "from typing import Any, Dict, List\n\n"
        + "from app.services.db import db\n\n"
        + slice_lines(lines, 540, 573),
    )

    qa_header = textwrap.dedent(
        '''\
        # -*- coding: utf-8 -*-
        """RegulationQASystem — 인덱싱·검색 파사드."""
        import os
        import time
        import json
        import threading
        import logging
        import traceback
        import math
        import re
        import gc
        import shutil
        import hashlib
        import importlib
        from dataclasses import dataclass
        from typing import List, Dict, Tuple, Optional, Any
        from collections import Counter
        from concurrent.futures import ThreadPoolExecutor

        from app.config import AppConfig
        from app.utils import logger, FileInfo, FileStatus, TaskResult, FileUtils, get_app_directory
        from app.constants import ErrorMessages, SuccessMessages, InfoMessages, Limits, Weights, Patterns
        from app.exceptions import (
            ModelNotLoadedError, ModelLoadError, ModelOfflineError,
            SearchError, SearchTimeoutError, SearchRateLimitError, SearchQueueFullError,
            DocumentExtractionError, IndexingError, CacheError
        )
        from app.services.db import db
        from app.services.document import DocumentExtractor, TextHighlighter, DocumentSplitter, ArticleParser
        from app.services.metadata import TagManager
        from app.services.file_manager import RevisionTracker

        from app.services.search.langchain_lazy import _lazy_import_langchain, CharacterTextSplitter, Document, HuggingFaceEmbeddings, FAISS
        from app.services.search.bm25 import BM25Light
        from app.services.search.cache import SearchCache
        from app.services.search.rate_limiter import RateLimiter
        from app.services.search.queue import SearchQueue
        from app.services.search.history import SearchHistory

        '''
    )

    qa_body = slice_lines(lines, 577, len(lines))
    # Remove module-level singletons from qa_body end - we'll put in __init__.py
    qa_body = qa_body.replace(
        "rate_limiter = RateLimiter(AppConfig.RATE_LIMIT_PER_MINUTE)\n"
        "search_queue = SearchQueue(AppConfig.MAX_CONCURRENT_SEARCHES)\n"
        "qa_system = RegulationQASystem()\n",
        "",
    )

    write(os.path.join(PKG, "qa_system.py"), qa_header + qa_body)

    write(
        os.path.join(PKG, "__init__.py"),
        textwrap.dedent(
            '''\
            # -*- coding: utf-8 -*-
            """검색 서비스 패키지 (legacy 하이브리드 + 싱글톤)."""
            from app.config import AppConfig
            from app.services.search.bm25 import BM25Light
            from app.services.search.cache import SearchCache
            from app.services.search.history import SearchHistory
            from app.services.search.rate_limiter import RateLimiter
            from app.services.search.queue import SearchQueue
            from app.services.search.qa_system import RegulationQASystem
            from app.services.search.langchain_lazy import _lazy_import_langchain

            rate_limiter = RateLimiter(AppConfig.RATE_LIMIT_PER_MINUTE)
            search_queue = SearchQueue(AppConfig.MAX_CONCURRENT_SEARCHES)
            qa_system = RegulationQASystem()

            __all__ = [
                "BM25Light",
                "SearchCache",
                "SearchHistory",
                "RateLimiter",
                "SearchQueue",
                "RegulationQASystem",
                "rate_limiter",
                "search_queue",
                "qa_system",
                "_lazy_import_langchain",
            ]
            '''
        ),
    )

    # Remove old search.py after package ready
    if os.path.exists(SRC):
        os.remove(SRC)
    print("search package created, old search.py removed")


if __name__ == "__main__":
    main()