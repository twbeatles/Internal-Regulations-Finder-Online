# -*- mode: python ; coding: utf-8 -*-
"""
사내 규정 검색기 - GUI 버전 PyInstaller Spec
빌드: pyinstaller server_gui.spec

버전: 1.4 (2024-12-19)
수정 사항:
- v1.3 다중 사용자 최적화 반영
- Rate Limiter, Search Queue 지원
- 병렬 문서 처리 (ThreadPoolExecutor) 지원
- 파일 작업 락 추가
- concurrent.futures 모듈 추가
"""

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# 프로젝트 경로
project_dir = os.path.dirname(os.path.abspath(SPEC))

# 데이터 파일 수집
datas = [
    (os.path.join(project_dir, 'templates'), 'templates'),
    (os.path.join(project_dir, 'static'), 'static'),
]

# config 폴더가 있으면 포함
config_dir = os.path.join(project_dir, 'config')
if os.path.exists(config_dir):
    datas.append((config_dir, 'config'))

# 숨겨진 import 수집
hiddenimports = [
    # Flask
    'flask', 'flask.sessions', 'flask_cors',
    'werkzeug', 'werkzeug.utils', 'werkzeug.security',
    'jinja2', 'markupsafe',
    
    # Waitress
    'waitress', 'waitress.server',
    
    # PyQt6
    'PyQt6', 'PyQt6.QtCore', 'PyQt6.QtGui', 'PyQt6.QtWidgets', 'PyQt6.sip',
    
    # LangChain
    'langchain', 'langchain.schema',
    'langchain_community', 'langchain_community.vectorstores',
    'langchain_huggingface', 'langchain_huggingface.embeddings',
    'langchain_text_splitters',
    'langchain_core', 'langchain_core.documents',
    
    # HuggingFace
    'transformers', 'sentence_transformers',
    'huggingface_hub', 'tokenizers',
    
    # FAISS
    'faiss',
    
    # PyTorch
    'torch', 'torch.nn', 'torch.utils',
    
    # 문서 처리
    'docx', 'docx.document', 'docx.opc', 'docx.oxml',
    'pypdf', 'pypdf.generic',
    'lxml', 'lxml.etree',
    
    # 유틸리티
    'psutil', 'numpy', 'tqdm', 'requests',
    'hashlib', 'json', 'dataclasses',
    
    # 멀티스레딩 (v1.3 추가)
    'concurrent', 'concurrent.futures',
]

# 동적 모듈 수집
try:
    hiddenimports += collect_submodules('sentence_transformers')
    hiddenimports += collect_submodules('transformers')
    hiddenimports += collect_submodules('PyQt6')
    hiddenimports += collect_submodules('langchain')
    hiddenimports += collect_submodules('langchain_community')
    hiddenimports += collect_submodules('langchain_huggingface')
    hiddenimports += collect_submodules('langchain_text_splitters')
    hiddenimports += collect_submodules('langchain_core')
    hiddenimports += collect_submodules('docx')
    hiddenimports += collect_submodules('pypdf')
    hiddenimports += collect_submodules('lxml')
except Exception:
    pass

# 데이터 파일 수집
try:
    datas += collect_data_files('torch')
    datas += collect_data_files('transformers')
    datas += collect_data_files('sentence_transformers')
except Exception:
    pass

# Analysis
a = Analysis(
    ['server_gui.py'],
    pathex=[project_dir],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter', 'matplotlib', 'PIL', 'cv2', 'scipy',
        'pandas', 'IPython', 'notebook', 'jupyter', 'pytest',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# PYZ Archive
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Executable
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='사내규정검색기',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

# Collect
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='사내규정검색기'
)
