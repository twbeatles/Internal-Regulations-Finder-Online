# -*- mode: python ; coding: utf-8 -*-
"""
사내 규정 검색기 v2.1 - 경량 빌드 Spec
빌드: pyinstaller internal_regulations_lite.spec

경량화 특징:
- AI/LLM 관련 라이브러리 최소화
- 불필요한 모듈 제외
- UPX 압축 활성화
"""

import os
import sys
from PyInstaller.utils.hooks import collect_submodules

block_cipher = None
project_dir = os.path.dirname(os.path.abspath(SPEC))

# 데이터 파일 (최소한만 포함)
datas = [
    (os.path.join(project_dir, 'templates'), 'templates'),
    (os.path.join(project_dir, 'static'), 'static'),
]

# Hidden Imports - 최소 필수 항목만
hiddenimports = [
    # Flask 코어
    'flask',
    'flask.json',
    'flask_cors',
    'werkzeug',
    'jinja2',
    
    # DB & 유틸리티
    'sqlite3',
    'threading',
    
    # 문서 처리 (선택적)
    'docx',           # python-docx
    'pypdf',          # PDF
    'openpyxl',       # Excel
    'olefile',        # HWP
    
    # 시스템
    'psutil',
]

# app 패키지 서브모듈
hiddenimports += collect_submodules('app')

# 경량 모드: langchain/torch 제외 시 주석 해제
# AI 기능 필요시 아래 주석 해제
# hiddenimports += [
#     'langchain', 'langchain_community',
#     'faiss', 'sentence_transformers',
# ]

# 제외 목록 (경량화 핵심)
excludes = [
    # GUI 프레임워크
    'tkinter', 'tk',
    'PyQt5', 'PyQt6', 'PySide2', 'PySide6',
    'wx', 'wxPython',
    
    # 데이터 분석 (불필요)
    'matplotlib', 'pandas', 'scipy',
    'seaborn', 'plotly',
    
    # 개발/테스트 도구
    'IPython', 'notebook', 'jupyter',
    'pytest', 'unittest',
    'setuptools', 'pkg_resources',
    
    # 대용량 AI 라이브러리 (오프라인 모드용 - 필요시 제거)
    'torch', 'tensorflow', 'keras',
    'transformers', 'huggingface_hub',
    
    # 기타
    'cv2', 'PIL.ImageTk',
    'email.test', 'test',
]

a = Analysis(
    ['run.py'],
    pathex=[project_dir],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# 불필요한 바이너리 필터링
a.binaries = [b for b in a.binaries if not any(
    x in b[0].lower() for x in [
        'qt5', 'qt6', 'opencv', 'torch', 'cuda',
        'mkl_', 'libiomp', 'libtorch',
    ]
)]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='사내규정검색기',
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,  # 디버그 심볼 제거
    upx=True,    # UPX 압축
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # 아이콘 있으면: 'icon.ico'
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=True,
    upx=True,
    upx_exclude=['vcruntime140.dll', 'python*.dll'],
    name='사내규정검색기',
)
