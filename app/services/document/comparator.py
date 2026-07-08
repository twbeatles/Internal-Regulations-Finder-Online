# -*- coding: utf-8 -*-
"""문서 diff 비교."""
from typing import Dict, Tuple

class DocumentComparator:
    """두 문서 간 차이점 비교"""
    
    def compare(self, doc1: str, doc2: str) -> Dict:
        import difflib
        
        lines1 = doc1.splitlines(keepends=True)
        lines2 = doc2.splitlines(keepends=True)
        
        differ = difflib.unified_diff(lines1, lines2, lineterm='')
        diff_lines = list(differ)
        
        added = sum(1 for line in diff_lines if line.startswith('+') and not line.startswith('+++'))
        removed = sum(1 for line in diff_lines if line.startswith('-') and not line.startswith('---'))
        
        html_diff = difflib.HtmlDiff()
        html_table = html_diff.make_table(lines1, lines2, context=True)
        
        return {
            'added_lines': added,
            'removed_lines': removed,
            'total_changes': added + removed,
            'diff_text': ''.join(diff_lines),
            'diff_html': html_table,
            'similarity': difflib.SequenceMatcher(None, doc1, doc2).ratio()
        }
    
    def highlight_changes(self, doc1: str, doc2: str) -> Tuple[str, str]:
        import difflib
        
        matcher = difflib.SequenceMatcher(None, doc1, doc2)
        
        result1 = []
        result2 = []
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                result1.append(doc1[i1:i2])
                result2.append(doc2[j1:j2])
            elif tag == 'delete':
                result1.append(f'<del>{doc1[i1:i2]}</del>')
            elif tag == 'insert':
                result2.append(f'<ins>{doc2[j1:j2]}</ins>')
            elif tag == 'replace':
                result1.append(f'<del>{doc1[i1:i2]}</del>')
                result2.append(f'<ins>{doc2[j1:j2]}</ins>')
        
        return ''.join(result1), ''.join(result2)
