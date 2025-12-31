# -*- coding: utf-8 -*-
from typing import List, Dict
from app.services.db import db
from app.utils import logger

class TagManager:
    """파일별 태그 관리 (SQLite 기반)"""
    
    PREDEFINED_CATEGORIES = [
        '인사', '회계', '보안', '복리후생', '근태', 
        '윤리', '조직', '계약', '기술', '기타'
    ]
    
    CATEGORY_KEYWORDS = {
        '인사': ['인사', '채용', '퇴직', '승진', '평가', '인력', '직원', '사원'],
        '회계': ['회계', '경비', '예산', '결산', '세금', '세무', '비용', '지출'],
        '보안': ['보안', '비밀', '정보보호', '접근', '암호', '인증', '개인정보'],
        '복리후생': ['복리', '후생', '건강', '보험', '연금', '지원금', '복지'],
        '근태': ['근태', '휴가', '출퇴근', '연차', '병가', '출장', '재택'],
        '윤리': ['윤리', '청렴', '공정', '부정', '비위', '행동강령'],
        '조직': ['조직', '부서', '팀', '직제', '직무', '업무분장'],
        '계약': ['계약', '협약', '협정', '입찰', '구매', '조달'],
        '기술': ['기술', '개발', 'IT', '시스템', '소프트웨어', '하드웨어']
    }
    
    def __init__(self):
        # DB 연결은 db 객체 사용
        pass
    
    def add_tag(self, filename: str, tag: str) -> bool:
        """태그 추가"""
        try:
            db.execute("INSERT INTO tags (filename, tag) VALUES (?, ?)", (filename, tag))
            return True
        except Exception:
            return False
    
    def remove_tag(self, filename: str, tag: str) -> bool:
        """태그 제거"""
        try:
            db.execute("DELETE FROM tags WHERE filename=? AND tag=?", (filename, tag))
            return True
        except Exception:
            return False
    
    def get_tags(self, filename: str) -> List[str]:
        """파일의 태그 목록 반환"""
        rows = db.fetchall("SELECT tag FROM tags WHERE filename=?", (filename,))
        return [r['tag'] for r in rows]
    
    def set_tags(self, filename: str, tags: List[str]):
        """파일의 태그 설정 (덮어쓰기)"""
        try:
            db.execute("DELETE FROM tags WHERE filename=?", (filename,))
            for tag in set(tags):
                try:
                    db.execute("INSERT INTO tags (filename, tag) VALUES (?, ?)", (filename, tag))
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"태그 설정 실패: {e}")
    
    def search_by_tag(self, tag: str) -> List[str]:
        """태그로 파일 검색"""
        rows = db.fetchall("SELECT DISTINCT filename FROM tags WHERE tag=?", (tag,))
        return [r['filename'] for r in rows]
    
    def get_all_tags(self) -> List[str]:
        """사용된 모든 태그 반환"""
        rows = db.fetchall("SELECT DISTINCT tag FROM tags ORDER BY tag")
        return [r['tag'] for r in rows]
    
    def auto_categorize(self, content: str, filename: str = "") -> List[str]:
        """키워드 기반 자동 카테고리 추천"""
        content_lower = content.lower()
        suggested = []
        
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in content_lower:
                    suggested.append(category)
                    break
        
        if filename:
            filename_lower = filename.lower()
            for category, keywords in self.CATEGORY_KEYWORDS.items():
                if category not in suggested:
                    for keyword in keywords:
                        if keyword in filename_lower:
                            suggested.append(category)
                            break
        
        return suggested if suggested else ['기타']
