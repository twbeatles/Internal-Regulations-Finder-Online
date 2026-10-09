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

class ArticleParser:
    """규정 문서의 조문 구조 파싱
    
    성능 최적화:
    - 사전 컴파일된 정규식 사용 (모듈 레벨에 정의)
    - 반복 패턴 매칭 최소화
    """
    
    # 클래스 레벨 패턴 (사전 컴파일된 모듈 레벨 패턴 참조)
    ARTICLE_PATTERNS = [
        (re.compile(r'제\s*(\d+)\s*장[^\n]*'), 'chapter'),
        (re.compile(r'제\s*(\d+)\s*절[^\n]*'), 'section'),
        (re.compile(r'제\s*(\d+)\s*조[^\n]*'), 'article'),
        (re.compile(r'제\s*(\d+)\s*조의\s*(\d+)[^\n]*'), 'article_sub'),
    ]
    
    ITEM_PATTERNS = [
        (re.compile(r'①|②|③|④|⑤|⑥|⑦|⑧|⑨|⑩'), 'paragraph'),
        (re.compile(r'^\s*(\d+)\.\s*'), 'numbered'),
        (re.compile(r'^\s*[가-하]\.\s*'), 'korean'),
    ]
    
    def parse_articles(self, content: str) -> List[Dict]:
        """조문별로 분리된 구조 반환 (사전 컴파일된 패턴 사용)"""
        articles = []
        # 사전 컴파일된 패턴 사용
        parts = _RE_ARTICLE_SPLIT.split(content)
        
        current_article = None
        for part in parts:
            match = _RE_ARTICLE_MATCH.match(part.strip())
            if match:
                if current_article:
                    articles.append(current_article)
                
                article_num = match.group(1)
                sub_num = match.group(2) or ""
                title = match.group(3) or ""
                
                current_article = {
                    'number': f"제{article_num}조" + (f"의{sub_num}" if sub_num else ""),
                    'title': title.strip().strip('()（）[]'),
                    'content': "",
                    'paragraphs': []
                }
            elif current_article and part.strip():
                current_article['content'] += part
                
                # 사전 컴파일된 패턴으로 항 분리
                para_split = _RE_PARAGRAPH_SPLIT.split(part)
                for i in range(1, len(para_split), 2):
                    if i+1 < len(para_split):
                        current_article['paragraphs'].append({
                            'marker': para_split[i],
                            'content': para_split[i+1].strip()
                        })
        
        if current_article:
            articles.append(current_article)
        
        return articles
    
    def search_article(self, articles: List[Dict], query: str) -> List[Dict]:
        """조문에서 검색"""
        results = []
        query_lower = query.lower()
        
        for article in articles:
            score = 0
            if query_lower in article.get('title', '').lower():
                score += 3
            if query_lower in article.get('content', '').lower():
                score += 1
            if query_lower in article.get('number', '').lower():
                score += 5
            
            if score > 0:
                results.append({**article, 'score': score})
        
        return sorted(results, key=lambda x: x['score'], reverse=True)
    
    def get_article_by_number(self, articles: List[Dict], number: str) -> Optional[Dict]:
        """조문 번호로 조문 찾기 (사전 컴파일된 패턴 사용)"""
        num_match = _RE_NUMBER_EXTRACT.search(number)
        if not num_match:
            return None
        
        target_num = num_match.group()
        for article in articles:
            if target_num in article.get('number', ''):
                return article
        return None

