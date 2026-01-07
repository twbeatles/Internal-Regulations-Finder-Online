# -*- mode: python ; coding: utf-8 -*-
"""
사내 규정 검색기 v2.2 - 최적화 빌드 Spec
빌드: pyinstaller regulation_search.spec

경량화 전략:
1. 불필요한 패키지 전면 제외 (matplotlib, scipy 등)
2. UPX 압축 활성화
3. 디버그 심볼 제거
4. 불필요한 바이너리 필터링
5. 하위 모듈 선택적 포함
"""

import os
import sys
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

block_cipher = None
project_dir = os.path.dirname(os.path.abspath(SPEC))

# ============================================================================
# 데이터 파일 (필수 최소화)
# ============================================================================
datas = [
    (os.path.join(project_dir, 'templates'), 'templates'),
    (os.path.join(project_dir, 'static'), 'static'),
    (os.path.join(project_dir, 'app'), 'app'),
]

# ============================================================================
# Hidden Imports - 필수 항목
# ============================================================================
hiddenimports = [
    # Flask 코어
    'flask',
    'flask.json',
    'flask.json.provider',
    'flask_cors',
    'werkzeug',
    'werkzeug.serving',
    'werkzeug.routing',
    'jinja2',
    'jinja2.ext',
    'markupsafe',
    
    # WSGI 서버
    'waitress',
    'waitress.server',
    
    # DB & 스레딩
    'sqlite3',
    'threading',
    'queue',
    'concurrent.futures',
    
    # 문서 처리
    'docx',
    'docx.shared',
    'pypdf',
    'openpyxl',
    'olefile',
    'lxml',
    'lxml.etree',
    
    # AI/ML (벡터 검색용)
    'torch',
    'sentence_transformers',
    'transformers',
    'huggingface_hub',
    
    # LangChain (호환성)
    'langchain',
    'langchain_core',
    'langchain_core.documents',
    'langchain_community',
    'langchain_community.embeddings',
    'langchain_community.vectorstores',
    'langchain_text_splitters',
    'langchain_huggingface',
    
    # FAISS 벡터 스토어
    'faiss',
    
    # 시스템
    'psutil',
    'hashlib',
    'json',
    're',
    'pickle',
    'gc',
    
    # 인코딩
    'encodings.utf_8',
    'encodings.cp949',
    'encodings.euc_kr',
]

# app 패키지 전체
hiddenimports += collect_submodules('app')
hiddenimports += collect_submodules('app.routes')
hiddenimports += collect_submodules('app.services')

# ============================================================================
# 제외 목록 (경량화 핵심)
# ============================================================================
excludes = [
    # GUI 프레임워크 (서버에서 불필요)
    'tkinter', 'tk', '_tkinter',
    'PyQt5', 'PyQt6', 'PySide2', 'PySide6',
    'wx', 'wxPython',
    
    # 데이터 분석/시각화 (불필요)
    'matplotlib', 'matplotlib.pyplot',
    'pandas',
    'scipy', 'scipy.sparse',
    'seaborn',
    'plotly',
    'bokeh',
    
    # 과학 계산 (불필요)
    'numpy.testing',
    'sympy',
    
    # 개발/테스트 도구
    'IPython', 'ipykernel', 'ipywidgets',
    'notebook', 'jupyter', 'jupyter_client',
    'pytest', 'unittest', 'doctest',
    # 주의: distutils, setuptools, pkg_resources는 Python 3.12+에서
    # PyInstaller 훅과 충돌하므로 제외하지 않음
    
    # 네트워크/크롤링
    'scrapy', 'selenium', 'requests_html',
    
    # 이미지/비디오 (OCR 외)
    'cv2', 'opencv',
    'moviepy',
    'imageio',
    
    # 게임/멀티미디어
    'pygame', 'pyglet',
    
    # 기타 불필요
    'email.test',
    'test', 'tests',
    'lib2to3',
    
    # TensorFlow (사용 안함)
    'tensorflow', 'keras', 'tensorboard',
]

# ============================================================================
# Analysis
# ============================================================================
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

# ============================================================================
# 바이너리 필터링 (크기 대폭 감소)
# ============================================================================
excluded_binaries = [
    # Qt 관련 (GUI 없음)
    'qt5', 'qt6', 'pyqt', 'pyside',
    
    # OpenCV (OCR 외 불필요)
    'opencv', 'cv2',
    
    # CUDA (CPU 전용 빌드)
    'cuda', 'cudnn', 'cublas', 'cufft', 'cusparse',
    'nvcuda', 'nvrtc',
    
    # Intel MKL (기본 BLAS 사용)
    'mkl_', 'libiomp', 'libiomk',
    
    # LibTorch 최소화
    'libtorch_cuda', 'torch_cuda',
    'caffe2_nvrtc',
    
    # TensorFlow
    'libtensorflow', 'tensorflow',
    
    # 디버그 심볼
    '.pdb', '_d.dll',
]

a.binaries = [
    b for b in a.binaries 
    if not any(ex in b[0].lower() for ex in excluded_binaries)
]

# ============================================================================
# 불필요한 데이터 필터링
# ============================================================================
excluded_datas = [
    'tcl', 'tk',
    'matplotlib',
    'share/jupyter',
    'share/doc',
]

a.datas = [
    d for d in a.datas
    if not any(ex in d[0].lower() for ex in excluded_datas)
]

# ============================================================================
# PYZ (Pure Python Archive)
# ============================================================================
pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=block_cipher
)

# ============================================================================
# EXE
# ============================================================================
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='사내규정검색기',
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,        # 디버그 심볼 제거
    upx=True,          # UPX 압축 활성화
    console=True,      # 콘솔 모드
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,         # 아이콘: 'static/icons/icon.ico'
    version=None,
)

# ============================================================================
# COLLECT
# ============================================================================
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=True,
    upx=True,
    upx_exclude=[
        'vcruntime140.dll',
        'python*.dll',
        'msvcp*.dll',
        'api-ms-*.dll',  # Windows API DLLs
    ],
    name='사내규정검색기',
)
