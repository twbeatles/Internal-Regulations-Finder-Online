# -*- mode: python ; coding: utf-8 -*-
"""
사내 규정 검색기 - GUI 버전 PyInstaller Spec
빌드: pyinstaller server_gui.spec
"""

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# 프로젝트 경로
project_dir = os.path.dirname(os.path.abspath(SPEC))

# 데이터 파일 수집
datas = [
    # 템플릿 및 정적 파일
    (os.path.join(project_dir, 'templates'), 'templates'),
    (os.path.join(project_dir, 'static'), 'static'),
]

# 숨겨진 import 수집
hiddenimports = [
    # Flask
    'flask',
    'flask_cors',
    'werkzeug',
    'werkzeug.utils',
    'jinja2',
    'markupsafe',
    
    # Waitress (프로덕션 서버)
    'waitress',
    
    # PyQt6 (GUI)
    'PyQt6',
    'PyQt6.QtCore',
    'PyQt6.QtGui',
    'PyQt6.QtWidgets',
    'PyQt6.sip',
    
    # LangChain 관련
    'langchain',
    'langchain.text_splitter',
    'langchain.docstore',
    'langchain.docstore.document',
    'langchain_community',
    'langchain_community.vectorstores',
    'langchain_community.vectorstores.faiss',
    'langchain_huggingface',
    
    # HuggingFace
    'transformers',
    'sentence_transformers',
    'huggingface_hub',
    'tokenizers',
    
    # FAISS
    'faiss',
    
    # PyTorch
    'torch',
    'torch.nn',
    'torch.utils',
    
    # 문서 처리
    'docx',
    'pypdf',
    
    # 시스템 모니터링 (선택적)
    'psutil',
    
    # 기타
    'numpy',
    'tqdm',
    'requests',
    'urllib3',
    'certifi',
    'charset_normalizer',
    'idna',
    'packaging',
    'regex',
    'safetensors',
    'filelock',
    'fsspec',
    'yaml',
    'pyyaml',
]

# sentence_transformers 하위 모듈 수집
try:
    hiddenimports += collect_submodules('sentence_transformers')
except Exception:
    pass

# transformers 하위 모듈 수집
try:
    hiddenimports += collect_submodules('transformers')
except Exception:
    pass

# PyQt6 하위 모듈 수집
try:
    hiddenimports += collect_submodules('PyQt6')
except Exception:
    pass

# torch 데이터 파일 수집
try:
    datas += collect_data_files('torch')
except Exception:
    pass

# transformers 데이터 파일 수집
try:
    datas += collect_data_files('transformers')
except Exception:
    pass

# sentence_transformers 데이터 파일 수집
try:
    datas += collect_data_files('sentence_transformers')
except Exception:
    pass

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
        # 불필요한 모듈 제외
        'tkinter',
        'matplotlib',
        'PIL',
        'cv2',
        'scipy',
        'pandas',
        'IPython',
        'notebook',
        'jupyter',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=block_cipher
)

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
    console=False,  # GUI 모드: 콘솔 창 숨김
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # 아이콘 경로 (옵션): 'icon.ico'
)

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
