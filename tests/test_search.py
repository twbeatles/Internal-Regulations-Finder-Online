# -*- coding: utf-8 -*-
"""
사내 규정 검색기 - 단위 테스트

테스트 실행:
    pytest tests/test_search.py -v
    pytest tests/test_search.py -v --cov=app
"""

import pytest
import time
import threading
from typing import Dict, Any, List
from unittest.mock import MagicMock, patch


class TestSearchCache:
    """SearchCache 클래스 테스트"""
    
    @pytest.fixture
    def cache(self):
        """테스트용 캐시 인스턴스"""
        from app.services.search import SearchCache
        return SearchCache(max_size=5, ttl_seconds=2)
    
    def test_set_and_get(self, cache):
        """기본 set/get 동작 테스트"""
        cache.set("테스트", 5, True, [{"result": 1}])
        result = cache.get("테스트", 5, True)
        
        assert result is not None
        assert result[0]["result"] == 1

    def test_sort_by_cache_key_isolated(self, cache):
        """sort_by가 다른 경우 캐시가 분리되는지 테스트"""
        cache.set("테스트", 5, True, [{"result": "relevance"}], sort_by="relevance")
        cache.set("테스트", 5, True, [{"result": "filename"}], sort_by="filename")

        relevance = cache.get("테스트", 5, True, sort_by="relevance")
        filename = cache.get("테스트", 5, True, sort_by="filename")

        assert relevance is not None
        assert filename is not None
        assert relevance[0]["result"] == "relevance"
        assert filename[0]["result"] == "filename"

    def test_filter_cache_key_isolated(self, cache):
        """파일 필터가 다른 경우 캐시가 분리되는지 테스트"""
        cache.set("policy", 5, True, [{"result": "all"}])
        cache.set("policy", 5, True, [{"result": "id-a"}], filter_file_id="id-a")
        cache.set("policy", 5, True, [{"result": "a.txt"}], filter_file="a.txt")

        all_results = cache.get("policy", 5, True)
        by_id = cache.get("policy", 5, True, filter_file_id="id-a")
        by_name = cache.get("policy", 5, True, filter_file="a.txt")

        assert all_results is not None
        assert by_id is not None
        assert by_name is not None
        assert all_results[0]["result"] == "all"
        assert by_id[0]["result"] == "id-a"
        assert by_name[0]["result"] == "a.txt"
    
    def test_cache_miss(self, cache):
        """캐시 미스 테스트"""
        result = cache.get("없는쿼리", 5, True)
        assert result is None
    
    def test_ttl_expiry(self, cache):
        """TTL 만료 테스트"""
        cache.set("만료테스트", 5, True, [{"data": "test"}])
        
        # TTL 전에는 조회 가능
        assert cache.get("만료테스트", 5, True) is not None
        
        # TTL 후에는 만료
        time.sleep(2.1)
        assert cache.get("만료테스트", 5, True) is None
    
    def test_lru_eviction(self, cache):
        """LRU 교체 정책 테스트"""
        # max_size(5)개 채우기
        for i in range(5):
            cache.set(f"query{i}", 5, True, [{"idx": i}])
        
        assert cache.size() == 5
        
        # 새 항목 추가 시 가장 오래된 항목 제거
        cache.set("query_new", 5, True, [{"idx": "new"}])
        
        # 첫 번째 항목(query0)은 제거됨
        assert cache.get("query0", 5, True) is None
        # 새 항목은 존재
        assert cache.get("query_new", 5, True) is not None
    
    def test_cache_stats(self, cache):
        """캐시 통계 테스트"""
        cache.set("통계테스트", 5, True, [])
        cache.get("통계테스트", 5, True)  # hit
        cache.get("없음", 5, True)  # miss
        
        stats = cache.get_stats()
        
        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['hit_rate'] == 50.0
    
    def test_invalidate_by_file(self, cache):
        """파일별 캐시 무효화 테스트"""
        cache.set("query1", 5, True, [{"source": "file1.txt", "content": "내용1"}])
        cache.set("query2", 5, True, [{"source": "file2.txt", "content": "내용2"}])
        cache.set("query3", 5, True, [{"source": "file1.txt", "content": "내용3"}])
        
        # file1.txt 관련 캐시 무효화
        count = cache.invalidate_by_file("file1.txt")
        
        assert count == 2
        assert cache.get("query1", 5, True) is None
        assert cache.get("query3", 5, True) is None
        assert cache.get("query2", 5, True) is not None


class TestRateLimiter:
    """RateLimiter 클래스 테스트"""
    
    @pytest.fixture
    def limiter(self):
        """테스트용 Rate Limiter (분당 5회 제한)"""
        from app.services.search import RateLimiter
        return RateLimiter(requests_per_minute=5)
    
    def test_allow_under_limit(self, limiter):
        """제한 미만 요청 허용"""
        ip = "192.168.1.1"
        
        # 5회까지 허용
        for _ in range(5):
            assert limiter.is_allowed(ip) is True
    
    def test_block_over_limit(self, limiter):
        """제한 초과 요청 차단"""
        ip = "192.168.1.2"
        
        # 5회 요청
        for _ in range(5):
            limiter.is_allowed(ip)
        
        # 6번째 요청은 차단
        assert limiter.is_allowed(ip) is False
    
    def test_different_ips_independent(self, limiter):
        """다른 IP는 독립적으로 제한"""
        ip1 = "192.168.1.1"
        ip2 = "192.168.1.2"
        
        # IP1 제한 도달
        for _ in range(5):
            limiter.is_allowed(ip1)
        
        # IP2는 여전히 허용
        assert limiter.is_allowed(ip2) is True
    
    def test_stats(self, limiter):
        """통계 테스트"""
        limiter.is_allowed("1.1.1.1")  # allowed
        limiter.is_allowed("1.1.1.1")  # allowed
        
        stats = limiter.get_stats()
        
        assert stats['total_allowed'] == 2
        assert stats['active_ips'] == 1


class TestSearchQueue:
    """SearchQueue 클래스 테스트"""
    
    @pytest.fixture
    def queue(self):
        """테스트용 검색 큐 (최대 2개 동시 실행)"""
        from app.services.search import SearchQueue
        return SearchQueue(max_concurrent=2)
    
    def test_acquire_release(self, queue):
        """기본 acquire/release 동작"""
        assert queue.acquire(timeout=1) is True
        assert queue.get_stats()['active'] == 1
        
        queue.release()
        assert queue.get_stats()['active'] == 0
    
    def test_max_concurrent(self, queue):
        """최대 동시 실행 제한"""
        # 2개 획득
        assert queue.acquire(timeout=0.1) is True
        assert queue.acquire(timeout=0.1) is True
        
        # 3번째는 타임아웃
        assert queue.acquire(timeout=0.1) is False
        
        # 통계 확인
        stats = queue.get_stats()
        assert stats['active'] == 2
        assert stats['rejected'] == 1


class TestBM25Light:
    """BM25Light 클래스 테스트"""
    
    @pytest.fixture
    def bm25(self):
        """테스트용 BM25 인스턴스"""
        from app.services.search import BM25Light
        bm25 = BM25Light()
        docs = [
            "인사규정 제1조 목적",
            "취업규칙 근로시간 휴식",
            "복리후생 건강검진 지원",
            "출장규정 국내출장 정산",
            "보안규정 정보보호 준수사항"
        ]
        bm25.fit(docs)
        return bm25
    
    def test_fit_and_search(self, bm25):
        """인덱싱 및 검색 테스트"""
        results = bm25.search("인사규정", top_k=3)
        
        assert len(results) > 0
        assert results[0][0] == 0  # 첫 번째 문서가 가장 관련성 높음
    
    def test_search_empty_query(self, bm25):
        """빈 쿼리 처리"""
        results = bm25.search("", top_k=3)
        assert results == []
    
    def test_search_no_match(self, bm25):
        """매칭 없는 쿼리"""
        results = bm25.search("xyz123", top_k=3)
        assert len(results) == 0 or all(score == 0 for _, score in results)
    
    def test_thread_safety(self, bm25):
        """스레드 안전성 테스트"""
        results_container = []
        
        def search_task():
            for _ in range(10):
                results = bm25.search("규정", top_k=2)
                results_container.append(len(results))
        
        threads = [threading.Thread(target=search_task) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # 모든 검색이 성공적으로 완료
        assert len(results_container) == 50


class TestMemoryMonitor:
    """MemoryMonitor 클래스 테스트"""
    
    def test_get_memory_usage(self):
        """메모리 사용량 조회"""
        from app.utils import MemoryMonitor
        
        usage = MemoryMonitor.get_memory_usage()
        
        assert 'gc_objects' in usage
        assert usage['gc_objects'] > 0
    
    def test_force_gc(self):
        """강제 GC 수행"""
        from app.utils import MemoryMonitor
        
        # 가비지 생성
        garbage = [list(range(100)) for _ in range(100)]
        del garbage
        
        result = MemoryMonitor.force_gc()
        
        assert 'collected' in result
        assert 'objects_before' in result
        assert 'objects_after' in result


class TestDocumentExtractor:
    """DocumentExtractor 클래스 테스트"""
    
    @pytest.fixture
    def extractor(self):
        """테스트용 문서 추출기"""
        from app.services.document import DocumentExtractor
        return DocumentExtractor()
    
    def test_extract_txt(self, extractor, tmp_path):
        """TXT 파일 추출"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("테스트 내용입니다.", encoding='utf-8')
        
        content, error = extractor.extract(str(test_file))
        
        assert error is None
        assert "테스트 내용" in content
    
    def test_extract_nonexistent(self, extractor):
        """존재하지 않는 파일"""
        content, error = extractor.extract("/nonexistent/file.txt")
        
        assert content == ""
        assert error is not None
    
    def test_extract_unsupported(self, extractor, tmp_path):
        """지원하지 않는 형식"""
        test_file = tmp_path / "test.xyz"
        test_file.write_text("test")
        
        content, error = extractor.extract(str(test_file))
        
        assert content == ""
        assert "지원하지 않는" in error


class TestArticleParser:
    """ArticleParser 클래스 테스트"""
    
    @pytest.fixture
    def parser(self):
        """테스트용 조문 파서"""
        from app.services.document import ArticleParser
        return ArticleParser()
    
    def test_parse_articles(self, parser):
        """조문 파싱"""
        content = """
        제1조 (목적) 이 규정은 인사관리에 관한 사항을 정함을 목적으로 한다.
        ① 첫번째 항
        ② 두번째 항
        
        제2조 (적용범위) 모든 임직원에게 적용한다.
        """
        
        articles = parser.parse_articles(content)
        
        assert len(articles) >= 2
        assert "제1조" in articles[0]['number']
    
    def test_search_article(self, parser):
        """조문 검색"""
        articles = [
            {"number": "제1조", "title": "목적", "content": "인사관리"},
            {"number": "제2조", "title": "범위", "content": "모든 직원"},
        ]
        
        results = parser.search_article(articles, "인사")
        
        assert len(results) > 0
        assert results[0]['number'] == "제1조"


# 통합 테스트
class TestIntegration:
    """통합 테스트"""
    
    def test_search_flow(self):
        """검색 플로우 통합 테스트"""
        from app.services.search import SearchCache, RateLimiter, SearchQueue
        
        cache = SearchCache(max_size=10, ttl_seconds=60)
        limiter = RateLimiter(requests_per_minute=100)
        queue = SearchQueue(max_concurrent=5)
        
        # 시뮬레이션: 여러 검색 요청
        for i in range(10):
            ip = f"192.168.1.{i % 5}"
            
            if limiter.is_allowed(ip):
                if queue.acquire(timeout=1):
                    try:
                        # 캐시 확인
                        result = cache.get(f"query{i}", 5, True)
                        if result is None:
                            # 검색 수행 (시뮬레이션)
                            result = [{"mock": True}]
                            cache.set(f"query{i}", 5, True, result)
                    finally:
                        queue.release()
        
        # 통계 확인
        assert cache.size() == 10
        assert queue.get_stats()['processed'] == 10


class TestLiteMode:
    def test_cache_dir_without_model_id(self):
        from app.services.search import RegulationQASystem

        qa = RegulationQASystem()
        try:
            qa.model_id = None
            cache_dir = qa._get_cache_dir("C:/tmp/folder")
            assert cache_dir
            assert isinstance(cache_dir, str)
        finally:
            qa.cleanup()

    def test_search_filter_file_id(self):
        from app.services.search import RegulationQASystem, BM25Light

        qa = RegulationQASystem()
        try:
            qa.documents = ["vacation policy guide", "security policy guide"]
            qa.doc_meta = [
                {"doc_id": 0, "source": "a.txt", "path": "C:/a.txt", "file_id": "id-a"},
                {"doc_id": 1, "source": "b.txt", "path": "C:/b.txt", "file_id": "id-b"},
            ]
            qa.bm25 = BM25Light()
            qa.bm25.fit(qa.documents)

            all_res = qa.search("vacation", hybrid=True)
            res = qa.search("vacation", hybrid=True, filter_file_id="id-a")
            assert res.success is True
            assert len(res.data) >= 1
            assert all(item.get("file_id") == "id-a" for item in res.data)
            assert all_res.success is True
            assert len(all_res.data) >= 1
        finally:
            qa.cleanup()

    def test_search_filter_file_name_not_polluted_by_unfiltered_cache(self):
        from app.services.search import RegulationQASystem, BM25Light

        qa = RegulationQASystem()
        try:
            qa.documents = ["vacation policy guide", "security policy guide"]
            qa.doc_meta = [
                {"doc_id": 0, "source": "a.txt", "path": "C:/a.txt", "file_id": "id-a"},
                {"doc_id": 1, "source": "b.txt", "path": "C:/b.txt", "file_id": "id-b"},
            ]
            qa.bm25 = BM25Light()
            qa.bm25.fit(qa.documents)

            qa.search("policy", hybrid=True)
            filtered = qa.search("policy", hybrid=True, filter_file="b.txt")

            assert filtered.success is True
            assert len(filtered.data) >= 1
            assert all(item.get("source") == "b.txt" for item in filtered.data)
        finally:
            qa.cleanup()

    def test_process_single_file_uses_app_chunk_policy(self, tmp_path, monkeypatch):
        from app.config import AppConfig
        from app.services.search import RegulationQASystem

        captured = {}

        class _DummySplitter:
            def __init__(self, chunk_size, chunk_overlap):
                captured["chunk_size"] = chunk_size
                captured["chunk_overlap"] = chunk_overlap

            def split(self, text):
                return [text]

        test_file = tmp_path / "sample.txt"
        test_file.write_text("샘플 텍스트", encoding="utf-8")

        qa = RegulationQASystem()
        try:
            monkeypatch.setattr("app.services.search.DocumentSplitter", _DummySplitter)
            monkeypatch.setattr(qa.extractor, "extract", lambda _: ("샘플 텍스트", None), raising=False)

            result = qa.process_single_file(str(test_file))

            assert result.success is True
            assert captured["chunk_size"] == AppConfig.CHUNK_SIZE
            assert captured["chunk_overlap"] == AppConfig.CHUNK_OVERLAP
        finally:
            qa.cleanup()

    def test_cache_entry_key_distinguishes_duplicate_basenames(self, tmp_path):
        from app.services.search import RegulationQASystem

        file1 = tmp_path / "A" / "same.txt"
        file2 = tmp_path / "B" / "same.txt"
        file1.parent.mkdir(parents=True, exist_ok=True)
        file2.parent.mkdir(parents=True, exist_ok=True)
        file1.write_text("alpha", encoding="utf-8")
        file2.write_text("beta", encoding="utf-8")

        qa = RegulationQASystem()
        try:
            key1 = qa._get_cache_entry_key(str(tmp_path), str(file1))
            key2 = qa._get_cache_entry_key(str(tmp_path), str(file2))

            assert key1 != key2
            assert key1.endswith("a/same.txt")
            assert key2.endswith("b/same.txt")
        finally:
            qa.cleanup()

    def test_remove_file_rebuilds_vector_store(self, tmp_path, monkeypatch):
        from app.services import search as search_module

        class _DummyDocument:
            def __init__(self, page_content, metadata):
                self.page_content = page_content
                self.metadata = metadata

        class _DummyVectorStore:
            def __init__(self, docs):
                self.docs = list(docs)

            def similarity_search_with_score(self, query, k=5):
                return [(doc, float(i + 1)) for i, doc in enumerate(self.docs[:k])]

            def add_documents(self, docs):
                self.docs.extend(docs)

        class _DummyFAISS:
            @staticmethod
            def from_documents(docs, embedding_model):
                return _DummyVectorStore(docs)

        qa = search_module.RegulationQASystem()
        file1 = tmp_path / "a.txt"
        file2 = tmp_path / "b.txt"
        file1.write_text("vacation policy guide", encoding="utf-8")
        file2.write_text("security policy guide", encoding="utf-8")

        try:
            monkeypatch.setattr(search_module, "Document", _DummyDocument, raising=False)
            monkeypatch.setattr(search_module, "FAISS", _DummyFAISS, raising=False)

            file1_id = search_module.FileUtils.make_file_id(str(file1))
            file2_id = search_module.FileUtils.make_file_id(str(file2))

            qa.embedding_model = object()
            qa.documents = ["vacation policy guide", "security policy guide"]
            qa.doc_meta = [
                {"doc_id": 0, "source": "a.txt", "path": str(file1), "file_id": file1_id},
                {"doc_id": 1, "source": "b.txt", "path": str(file2), "file_id": file2_id},
            ]
            qa.file_infos = {
                str(file1): search_module.FileInfo(path=str(file1), size=file1.stat().st_size, chunks=1),
                str(file2): search_module.FileInfo(path=str(file2), size=file2.stat().st_size, chunks=1),
            }
            qa._build_bm25()
            qa._rebuild_vector_store_locked()

            removed = qa.remove_file_from_index(str(file1), "a.txt", file1_id)
            search_result = qa.search("policy", hybrid=False)

            assert removed is True
            assert search_result.success is True
            assert len(search_result.data) == 1
            assert search_result.data[0]["file_id"] == file2_id
            assert search_result.data[0]["source"] == "b.txt"

            removed_last = qa.remove_file_from_index(str(file2), "b.txt", file2_id)
            final_result = qa.search("policy", hybrid=False)

            assert removed_last is True
            assert qa.vector_store is None
            assert qa.bm25 is None
            assert final_result.success is False
        finally:
            qa.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
