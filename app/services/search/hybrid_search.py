# -*- coding: utf-8 -*-
"""하이브리드 검색 (Vector + BM25 융합)."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from app.config import AppConfig
from app.services.search.normalization import prepare_vector_query
from app.utils import FileUtils, TaskResult, logger

if TYPE_CHECKING:
    from app.services.search.qa_system import RegulationQASystem


class HybridSearchService:
    """RegulationQASystem.search() 본문 — SRP 분리."""

    def search(
        self,
        qa: "RegulationQASystem",
        query: str,
        k: int = 5,
        hybrid: bool = True,
        sort_by: str = "relevance",
        filter_file: str | None = None,
        filter_file_id: str | None = None,
    ) -> TaskResult:
        start_time = time.perf_counter()

        if not qa.vector_store and not qa.bm25:
            return TaskResult(False, "문서가 로드되지 않음")

        query = query.strip()
        if len(query) < 2:
            return TaskResult(False, "검색어가 너무 짧습니다 (최소 2자)")

        vector_query = prepare_vector_query(query)

        cached_result = qa._search_cache.get(query, k, hybrid, sort_by, filter_file, filter_file_id)
        if cached_result is not None:
            return TaskResult(True, "검색 완료 (캐시)", cached_result)

        try:
            k = max(1, min(k, AppConfig.MAX_SEARCH_RESULTS))
            results: dict[Any, dict[str, Any]] = {}
            parallel_search = getattr(AppConfig, "PARALLEL_SEARCH", True)

            if hybrid and qa.vector_store and qa.bm25 and parallel_search:
                results = self._parallel_hybrid(qa, query, vector_query, k)
            else:
                results = self._sequential_hybrid(qa, query, vector_query, k, hybrid)

            if filter_file_id:
                results = {key: val for key, val in results.items() if val.get("file_id") == filter_file_id}
            if filter_file:
                results = {key: val for key, val in results.items() if val["source"] == filter_file}

            for item in results.values():
                item["score"] = (
                    AppConfig.VECTOR_WEIGHT * item["vec_score"]
                    + AppConfig.BM25_WEIGHT * item["bm25_score"]
                )

            if sort_by == "filename":
                sorted_res = sorted(results.values(), key=lambda x: x["source"])[:k]
            elif sort_by == "length":
                sorted_res = sorted(results.values(), key=lambda x: len(x["content"]), reverse=True)[:k]
            else:
                sorted_res = sorted(results.values(), key=lambda x: x["score"], reverse=True)[:k]

            qa._search_cache.set(query, k, hybrid, sorted_res, sort_by, filter_file, filter_file_id)

            if len(qa.documents) > 5000:
                from app.utils import MemoryMonitor

                warning = MemoryMonitor.check_memory_warning(threshold_mb=512)
                if warning:
                    logger.warning(f"검색 후 메모리 경고: {warning}")

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            if elapsed_ms > 1000.0:
                logger.warning(
                    "search_slow query_len=%d results=%d elapsed_ms=%.1f",
                    len(query),
                    len(sorted_res),
                    elapsed_ms,
                )
            elif elapsed_ms > 500.0:
                logger.info(
                    "search_done query_len=%d results=%d elapsed_ms=%.1f",
                    len(query),
                    len(sorted_res),
                    elapsed_ms,
                )

            return TaskResult(True, "검색 완료", sorted_res)

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(f"검색 오류 ({elapsed_ms:.1f}ms): {e}")
            return TaskResult(False, f"검색 오류: {e}")

    def _parallel_hybrid(
        self,
        qa: "RegulationQASystem",
        query: str,
        vector_query: str,
        k: int,
    ) -> dict[Any, dict[str, Any]]:
        fetch_k = k * 2
        results: dict[Any, dict[str, Any]] = {}

        def vector_search():
            try:
                return qa.vector_store.similarity_search_with_score(vector_query, k=fetch_k)
            except Exception as e:
                logger.debug(f"Vector 검색 오류: {e}")
                return []

        def bm25_search():
            try:
                return qa.bm25.search(query, top_k=fetch_k)
            except Exception as e:
                logger.debug(f"BM25 검색 오류: {e}")
                return []

        vec_future = qa._search_executor.submit(vector_search)
        bm25_future = qa._search_executor.submit(bm25_search)
        vec_results = vec_future.result(timeout=30)
        bm_res = bm25_future.result(timeout=30)

        if vec_results:
            self._merge_vector_results(qa, vec_results, results)
        if bm_res:
            self._merge_bm25_results(qa, bm_res, results)
        return results

    def _sequential_hybrid(
        self,
        qa: "RegulationQASystem",
        query: str,
        vector_query: str,
        k: int,
        hybrid: bool,
    ) -> dict[Any, dict[str, Any]]:
        results: dict[Any, dict[str, Any]] = {}
        fetch_k = k * 2 if hybrid else k

        if qa.vector_store:
            vec_results = qa.vector_store.similarity_search_with_score(vector_query, k=fetch_k)
            if vec_results:
                self._merge_vector_results(qa, vec_results, results)

        if hybrid and qa.bm25:
            try:
                bm_res = qa.bm25.search(query, top_k=k * 2)
            except Exception as e:
                logger.debug(f"BM25 검색 중 오류 (무시됨): {e}")
                bm_res = []
            if bm_res:
                self._merge_bm25_results(qa, bm_res, results)
        return results

    def _merge_vector_results(self, qa: "RegulationQASystem", vec_results, results: dict) -> None:
        distances = [r[1] for r in vec_results]
        min_d = min(distances)
        max_d = max(distances)
        rng = max_d - min_d if max_d != min_d else 1

        for doc, dist in vec_results:
            meta = doc.metadata or {}
            doc_id = meta.get("doc_id")
            key = doc_id if isinstance(doc_id, int) else doc.page_content[:100]
            score = max(0.1, 1 - ((dist - min_d) / (rng + 0.001)))
            results[key] = {
                "content": qa.documents[doc_id]
                if isinstance(doc_id, int) and 0 <= doc_id < len(qa.documents)
                else doc.page_content,
                "source": meta.get("source", "?"),
                "path": meta.get("path", ""),
                "file_id": meta.get("file_id")
                or (FileUtils.make_file_id(meta.get("path", "")) if meta.get("path") else ""),
                "vec_score": score,
                "bm25_score": 0,
            }

    def _merge_bm25_results(self, qa: "RegulationQASystem", bm_res, results: dict) -> None:
        bm_scores = [r[1] for r in bm_res]
        max_bm = max(bm_scores) if bm_scores else 1
        for idx, sc in bm_res:
            if 0 <= idx < len(qa.documents):
                key = idx
                norm = sc / (max_bm + 0.001)
                if key in results:
                    results[key]["bm25_score"] = norm
                else:
                    meta = qa.doc_meta[idx] if idx < len(qa.doc_meta) else {}
                    results[key] = {
                        "content": qa.documents[idx],
                        "source": meta.get("source", "?"),
                        "path": meta.get("path", ""),
                        "file_id": meta.get("file_id")
                        or (FileUtils.make_file_id(meta.get("path", "")) if meta.get("path") else ""),
                        "vec_score": 0,
                        "bm25_score": norm,
                    }


_hybrid_search_service = HybridSearchService()