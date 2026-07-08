# -*- coding: utf-8 -*-
"""문서 청킹·구조 분할."""
import re
from typing import Dict, List

from app.constants import Limits

class DocumentSplitter:
    """대용량 규정집을 개별 규정 파일로 분할
    
    Usage:
        splitter = DocumentSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split(text)
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Args:
            chunk_size: 각 청크의 최대 문자 수
            chunk_overlap: 청크 간 겹치는 문자 수
        """
        self.chunk_size = max(chunk_size, Limits.MIN_CHUNK_SIZE)
        self.chunk_overlap = min(chunk_overlap, chunk_size // 2)
    
    def split(self, text: str) -> List[str]:
        """텍스트를 청크로 분할
        
        Args:
            text: 분할할 전체 텍스트
            
        Returns:
            분할된 청크 리스트
        """
        if not text or not text.strip():
            return []
        
        # 문단 단위로 먼저 분리
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # 현재 청크 + 새 문단이 chunk_size 이하면 추가
            if len(current_chunk) + len(para) + 2 <= self.chunk_size:
                current_chunk += ('\n\n' + para) if current_chunk else para
            else:
                # 현재 청크 저장
                if current_chunk:
                    chunks.append(current_chunk)
                
                # 새 문단이 chunk_size보다 크면 강제 분할
                if len(para) > self.chunk_size:
                    for i in range(0, len(para), self.chunk_size - self.chunk_overlap):
                        chunk = para[i:i + self.chunk_size]
                        if chunk.strip():
                            chunks.append(chunk)
                    current_chunk = ""
                else:
                    # overlap 적용
                    if chunks and self.chunk_overlap > 0:
                        overlap_text = chunks[-1][-self.chunk_overlap:]
                        current_chunk = overlap_text + '\n\n' + para
                    else:
                        current_chunk = para
        
        # 마지막 청크 저장
        if current_chunk.strip():
            chunks.append(current_chunk)
        
        return chunks
    
    def split_by_chapters(self, content: str, filename: str = "") -> List[Dict]:
        pattern = r'(제\s*\d+\s*장[^\n]*)'
        return self._split_by_pattern(content, pattern, 'chapter', filename)
    
    def split_by_articles(self, content: str, filename: str = "") -> List[Dict]:
        pattern = r'(제\s*\d+\s*조(?:의\s*\d+)?[^\n]*)'
        return self._split_by_pattern(content, pattern, 'article', filename)
    
    def _split_by_pattern(self, content: str, pattern: str, split_type: str, filename: str) -> List[Dict]:
        parts = re.split(pattern, content)
        results = []
        
        current_title = None
        current_content = ""
        
        for part in parts:
            if re.match(pattern, part):
                if current_title:
                    results.append({
                        'title': current_title.strip(),
                        'content': current_content.strip(),
                        'type': split_type,
                        'source': filename
                    })
                current_title = part
                current_content = ""
            else:
                current_content += part
        
        if current_title:
            results.append({
                'title': current_title.strip(),
                'content': current_content.strip(),
                'type': split_type,
                'source': filename
            })
        
        return results
    
    def split_by_size(self, content: str, max_size: int = 5000) -> List[Dict]:
        results = []
        paragraphs = content.split('\n\n')
        
        current_chunk = ""
        chunk_num = 1
        
        for para in paragraphs:
            if len(current_chunk) + len(para) > max_size:
                if current_chunk:
                    results.append({
                        'title': f'파트 {chunk_num}',
                        'content': current_chunk.strip(),
                        'type': 'size_split'
                    })
                    chunk_num += 1
                    current_chunk = para
                else:
                    results.append({
                        'title': f'파트 {chunk_num}',
                        'content': para[:max_size],
                        'type': 'size_split'
                    })
                    chunk_num += 1
            else:
                current_chunk += '\n\n' + para if current_chunk else para
        
        if current_chunk:
            results.append({
                'title': f'파트 {chunk_num}',
                'content': current_chunk.strip(),
                'type': 'size_split'
            })
        
        return results
