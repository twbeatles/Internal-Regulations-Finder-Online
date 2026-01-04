# -*- mode: python ; coding: utf-8 -*-
"""
사내 규정 검색기 - GUI 버전 PyInstaller Spec (경량화)
빌드: pyinstaller server_gui.spec

버전: 1.8 (2025-12-31)
- v1.4 GUI 업데이트 반영 (툴팁, UI/UX 개선)
- 경량화 최적화: 불필요한 모듈 제외 강화
- UPX 압축 활성화
- 바이너리 스트리핑 활성화
"""

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None
project_dir = os.path.dirname(os.path.abspath(SPEC))

# ============================================================================
# 데이터 파일 (최소한으로 유지)
# ============================================================================
datas = [
    (os.path.join(project_dir, 'templates'), 'templates'),
    (os.path.join(project_dir, 'static'), 'static'),
]

# config 폴더 (존재 시 포함)
config_dir = os.path.join(project_dir, 'config')
if os.path.exists(config_dir):
    datas.append((config_dir, 'config'))

# ============================================================================
# Hidden Imports (필수 항목만)
# ============================================================================
hiddenimports = [
    # === Flask & Web ===
    'flask', 'flask.sessions', 'flask_cors',
    'werkzeug', 'werkzeug.utils', 'werkzeug.security',
    'jinja2', 'markupsafe',
    'waitress', 'waitress.server',
    
    # === PyQt6 (GUI) ===
    'PyQt6', 'PyQt6.QtCore', 'PyQt6.QtGui', 'PyQt6.QtWidgets', 'PyQt6.sip',
    
    # === LangChain (Core only) ===
    'langchain', 'langchain.schema',
    'langchain_community', 'langchain_community.vectorstores',
    'langchain_huggingface', 'langchain_huggingface.embeddings',
    'langchain_text_splitters',
    'langchain_core', 'langchain_core.documents',
    
    # === AI/ML (필수) ===
    'transformers', 'sentence_transformers',
    'huggingface_hub', 'tokenizers',
    'faiss',
    'torch', 'torch.nn', 'torch.utils',
    
    # === 문서 처리 ===
    'docx', 'docx.document', 'docx.opc', 'docx.oxml',
    'pypdf', 'pypdf.generic',
    'lxml', 'lxml.etree',
    'openpyxl', 'olefile',
    
    # === 유틸리티 ===
    'psutil', 'numpy', 'tqdm', 'requests',
    'hashlib', 'json', 'dataclasses', 'sqlite3',
    'concurrent', 'concurrent.futures',
    'watchdog', 'watchdog.observers', 'watchdog.events',
]

# ============================================================================
# 동적 모듈 수집 (최소화)
# ============================================================================
try:
    hiddenimports += collect_submodules('PyQt6')
except: pass

try:
    hiddenimports += collect_submodules('docx')
    hiddenimports += collect_submodules('pypdf')
except: pass

# ============================================================================
# 제외 목록 (경량화 - 대폭 확장)
# ============================================================================
excludes = [
    # GUI 라이브러리 (PyQt6 제외 모두)
    'tkinter', 'PyQt5', 'PySide2', 'PySide6', 'wx', 'kivy',
    
    # 대용량 과학 라이브러리
    'matplotlib', 'scipy', 'pandas', 'sklearn', 'scikit-learn',
    'cv2', 'opencv', 'PIL', 'pillow',
    
    # 개발/테스트 도구
    'IPython', 'notebook', 'jupyter', 'pytest', 'unittest',
    'sphinx', 'docutils', 'pydoc',
    
    # 불필요한 서브모듈
    'torch.distributions', 'torch.testing',
    'transformers.data', 'transformers.pipelines',
    
    # 기타
    'email.test', 'test', 'tests',
    'setuptools', 'pkg_resources',
    'cython', 'numba',
]

# ============================================================================
# Analysis
# ============================================================================
a = Analysis(
    ['server_gui.py'],
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

# ============================================================================
# PYZ Archive
# ============================================================================
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# ============================================================================
# Executable
# ============================================================================
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='사내규정검색기',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,              # Windows에서는 strip 도구 없음 (Linux 전용)
    upx=True,                 # 경량화: UPX 압축
    console=False,            # GUI 모드 (콘솔 숨김)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

# ============================================================================
# Collect (최종 패키지)
# ============================================================================
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,              # Windows 호환성 (strip은 Linux 전용)
    upx=True,                 # 경량화: UPX 압축
    upx_exclude=[             # UPX 제외 (손상 방지)
        'python*.dll',
        'vcruntime*.dll',
        'Qt*.dll',
    ],
    name='사내규정검색기'
)
