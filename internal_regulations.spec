# -*- mode: python ; coding: utf-8 -*-
"""
사내 규정 검색기 v2.0 - PyInstaller Spec
빌드: pyinstaller internal_regulations.spec
"""

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None
project_dir = os.path.dirname(os.path.abspath(SPEC))

# 데이터 파일 수집
datas = [
    (os.path.join(project_dir, 'templates'), 'templates'),
    (os.path.join(project_dir, 'static'), 'static'),
    # config 폴더는 실행 시 생성되므로 포함하지 않음 (필요 시 기본 설정 파일 포함)
]

# Hidden Imports
hiddenimports = [
    # Flask & Extensions
    'flask', 'flask.json', 'flask_cors',
    'werkzeug', 'jinja2',
    
    # Core Dependencies
    'watchdog', 'openpyxl', 'olefile', 'pytesseract', 'pdf2image', 'PIL', 'sqlite3',
    'numpy', 'tqdm', 'requests', 'psutil', 'lxml',
    
    # AI & LangChain
    'langchain', 'langchain_community', 'langchain_huggingface',
    'langchain_text_splitters', 'langchain_core',
    'faiss', 'torch', 'sentence_transformers', 'transformers',
]

# 동적 모듈 수집
hiddenimports += collect_submodules('app')
hiddenimports += collect_submodules('langchain')
hiddenimports += collect_submodules('langchain_community')
hiddenimports += collect_submodules('langchain_huggingface')

# 외부 라이브러리 데이터 파일 수집
try:
    datas += collect_data_files('langchain_community')
except: pass

a = Analysis(
    ['run.py'],
    pathex=[project_dir],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter', 'matplotlib', 'cv2', 'pandas', 'IPython', 'notebook', 'jupyter', 'pytest',
        'PyQt6', 'PyQt5', 'wx', 'unittest', 'email.test'
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='InternalRegulationsFinder',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True, # 서버 로그 확인을 위해 콘솔 표시
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='InternalRegulationsFinder',
)
