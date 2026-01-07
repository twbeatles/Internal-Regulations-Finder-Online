# -*- mode: python ; coding: utf-8 -*-
"""
사내 규정 검색기 v2.2 - 초경량 빌드 Spec
빌드: pyinstaller regulation_search_lite.spec

목표 크기: 100-150MB (현재 1.3GB에서 대폭 감소)

경량화 전략:
1. AI/ML 라이브러리 완전 제외 (BM25 텍스트 검색만)
2. CUDA, MKL, OpenBLAS 바이너리 전면 제거
3. PyTorch/Transformers 완전 제외
4. 불필요한 데이터 파일 필터링
5. UPX 압축 최대화
"""

import os
import sys
from PyInstaller.utils.hooks import collect_submodules

block_cipher = None
project_dir = os.path.dirname(os.path.abspath(SPEC))

# ============================================================================
# 데이터 파일 (필수만)
# ============================================================================
datas = [
    (os.path.join(project_dir, 'templates'), 'templates'),
    (os.path.join(project_dir, 'static'), 'static'),
]

# ============================================================================
# Hidden Imports - 최소 필수만
# ============================================================================
hiddenimports = [
    # Flask 코어
    'flask',
    'flask.json',
    'flask.json.provider',
    'flask_cors',
    'werkzeug',
    'werkzeug.serving',
    'jinja2',
    'markupsafe',
    
    # WSGI
    'waitress',
    'waitress.server',
    
    # DB & 스레딩
    'sqlite3',
    'threading',
    'queue',
    'concurrent.futures',
    
    # 문서 처리 (기본)
    'docx',
    'pypdf',
    'openpyxl',
    'olefile',
    'lxml',
    'lxml.etree',
    
    # 시스템
    'psutil',
    
    # 인코딩
    'encodings.utf_8',
    'encodings.cp949',
    'encodings.euc_kr',
    
    # App 모듈
    'app',
    'app.config',
    'app.utils',
    'app.routes',
    'app.routes.main',
    'app.routes.api_search',
    'app.routes.api_files',
    'app.routes.api_system',
    'app.services',
    'app.services.search',
    'app.services.document',
    'app.services.db',
]

# ============================================================================
# 제외 목록 (대폭 확대 - 핵심 경량화)
# ============================================================================
excludes = [
    # === AI/ML 전체 제외 (가장 큰 용량 차지) ===
    'torch', 'torchvision', 'torchaudio',
    'tensorflow', 'keras', 'tensorboard',
    'transformers', 'huggingface_hub',
    'sentence_transformers',
    'langchain', 'langchain_core', 'langchain_community',
    'langchain_text_splitters', 'langchain_huggingface',
    'faiss', 'faiss_cpu',
    'accelerate',
    'safetensors',
    'tokenizers',
    'optimum',
    'onnx', 'onnxruntime',
    
    # GUI 프레임워크
    'tkinter', 'tk', '_tkinter',
    'PyQt5', 'PyQt6', 'PySide2', 'PySide6',
    'wx', 'wxPython',
    
    # 데이터 분석/시각화
    'matplotlib', 'pandas', 'scipy',
    'seaborn', 'plotly', 'bokeh',
    'numpy.testing',
    
    # 개발 도구
    'IPython', 'notebook', 'jupyter',
    'pytest', 'unittest',
    
    # 이미지/비디오
    'cv2', 'opencv', 'PIL.ImageTk',
    'pytesseract', 'pdf2image',
    'imageio', 'moviepy',
    
    # 네트워크
    'scrapy', 'selenium', 'requests_html',
    
    # 기타
    'email.test', 'test', 'tests', 'lib2to3',
    'sympy', 'numba', 'numexpr',
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
# 바이너리 필터링 (매우 공격적 - 용량 대폭 감소)
# ============================================================================
excluded_binaries = [
    # === PyTorch/LibTorch (가장 큰 용량) ===
    'torch', 'libtorch', 'c10', 'caffe2',
    'torch_cpu', 'torch_cuda',
    'libc10', 'libc10_cuda',
    'libshm', 'libnvfuser',
    'fbgemm', 'asmjit',
    'dnnl', 'ideep',
    
    # CUDA/GPU 관련 (500MB+)
    'cuda', 'cudnn', 'cublas', 'cufft', 'cusparse', 'curand',
    'nvcuda', 'nvrtc', 'nvidia', 'nvjitlink',
    'cudart', 'cublaslt', 'cusolver',
    
    # Intel MKL (200MB+)
    'mkl_', 'mkl-', 'libmkl',
    'libiomp', 'libiomk', 
    'svml', 'libimalloc',
    'tbb', 'libtbb',
    
    # OpenBLAS
    'openblas', 'libopenblas',
    
    # Qt 관련
    'qt5', 'qt6', 'pyqt', 'pyside',
    'libqt', 'Qt5', 'Qt6',
    
    # TensorFlow
    'tensorflow', 'libtensorflow',
    
    # OpenCV
    'opencv', 'cv2', 'libopencv',
    
    # LLVM/Clang
    'llvm', 'clang', 'libllvm', 'libclang',
    
    # Transformers/HuggingFace
    'transformers', 'tokenizers',
    'sentencepiece',
    
    # 기타 대용량
    'triton',
    'nvtx',
    'magma',
    
    # 디버그 심볼
    '.pdb', '_d.dll', '_d.pyd',
]

print(f"필터링 전 바이너리 수: {len(a.binaries)}")
a.binaries = [
    b for b in a.binaries 
    if not any(ex in b[0].lower() for ex in excluded_binaries)
]
print(f"필터링 후 바이너리 수: {len(a.binaries)}")

# ============================================================================
# 데이터 필터링 (캐시, 테스트, 문서 제거)
# ============================================================================
excluded_datas = [
    # GUI
    'tcl', 'tk',
    
    # 시각화
    'matplotlib', 'mpl-data',
    
    # AI/ML
    'torch', 'transformers', 'huggingface',
    'sentence_transformers',
    'faiss',
    
    # 기타
    'share/jupyter', 'share/doc',
    'include/',
    '.dist-info',
    '__pycache__',
    'tests/',
    'test/',
]

print(f"필터링 전 데이터 수: {len(a.datas)}")
a.datas = [
    d for d in a.datas
    if not any(ex in d[0].lower() for ex in excluded_datas)
]
print(f"필터링 후 데이터 수: {len(a.datas)}")

# ============================================================================
# PYZ
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
    upx=True,          # UPX 압축
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
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
        'python3*.dll',
        'msvcp*.dll',
        'api-ms-*.dll',
        'ucrtbase.dll',
    ],
    name='사내규정검색기',
)
