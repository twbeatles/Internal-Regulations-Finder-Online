# -*- coding: utf-8 -*-
"""
정적 자원 다운로드 스크립트 (오프라인/폐쇄망 환경용)
jsPDF, AutoTable, 글꼴 등을 다운로드하여 static 폴더에 저장합니다.
"""

import os
import requests
import time

def download_file(url, save_path):
    print(f"다운로드 중: {url} -> {save_path}")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # 디렉토리 생성
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print("✅ 성공")
        return True
    except Exception as e:
        print(f"❌ 실패: {e}")
        return False

def main():
    print("=" * 60)
    print("🚀 사내 규정 검색기 - 정적 자원 다운로드 도구")
    print("   (오프라인/폐쇄망 환경 사전 준비용)")
    print("=" * 60)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    static_dir = os.path.join(base_dir, 'static')
    vendor_dir = os.path.join(static_dir, 'vendor')
    
    resources = [
        {
            "name": "jsPDF (v2.5.1)",
            "url": "https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js",
            "path": os.path.join(vendor_dir, "jspdf.umd.min.js")
        },
        {
            "name": "jsPDF AutoTable (v3.8.1)",
            "url": "https://cdnjs.cloudflare.com/ajax/libs/jspdf-autotable/3.8.1/jspdf.plugin.autotable.min.js",
            "path": os.path.join(vendor_dir, "jspdf.plugin.autotable.min.js")
        },
        # 필요한 경우 폰트 파일도 여기에 추가 가능
        # Google Fonts는 CSS와 WOFF2 파일 구조가 복잡하여 여기서는 제외
    ]
    
    success_count = 0
    for res in resources:
        if download_file(res['url'], res['path']):
            success_count += 1
            
    print("-" * 60)
    if success_count == len(resources):
        print("🎉 모든 정적 자원 다운로드 완료!")
        print(f"📂 저장 위치: {vendor_dir}")
    else:
        print(f"⚠️ 일부 다운로드 실패 ({success_count}/{len(resources)} 성공)")

if __name__ == "__main__":
    main()
