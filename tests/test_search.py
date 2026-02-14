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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
