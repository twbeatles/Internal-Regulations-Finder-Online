# -*- mode: python ; coding: utf-8 -*-
"""
사내 규정 검색기 v2.5 - 단일 파일 초경량 빌드 Spec
빌드: pyinstaller regulation_search_onefile.spec --clean

특징:
- 단일 실행 파일 (onefile 모드)
- AI/ML 완전 제외 (BM25 텍스트 검색만)
- 최소 필수 라이브러리만 포함
- USB 휴대 및 배포 용이

예상 크기: 40-60MB (단일 .exe 파일)
"""

import os
import sys
from PyInstaller.utils.hooks import collect_submodules, collect_data_files, collect_dynamic_libs

block_cipher = None
project_dir = os.path.dirname(os.path.abspath(SPEC))

# ============================================================================
# PyQt6 수집 (GUI 필수)
# ============================================================================
pyqt_binaries = []
pyqt_datas = []
pyqt_hiddenimports = []

try:
    pyqt_binaries = collect_dynamic_libs('PyQt6')
    print(f"[INFO] PyQt6 바이너리: {len(pyqt_binaries)}개")
except Exception as e:
    print(f"[WARNING] PyQt6 바이너리 수집 실패: {e}")

try:
    pyqt_datas = collect_data_files('PyQt6')
    print(f"[INFO] PyQt6 데이터: {len(pyqt_datas)}개")
except Exception as e:
    print(f"[WARNING] PyQt6 데이터 수집 실패: {e}")

try:
    pyqt_hiddenimports = collect_submodules('PyQt6')
    print(f"[INFO] PyQt6 서브모듈: {len(pyqt_hiddenimports)}개")
except Exception as e:
    print(f"[WARNING] PyQt6 서브모듈 수집 실패: {e}")

# ============================================================================
# 데이터 파일
# ============================================================================
datas = [
    (os.path.join(project_dir, 'templates'), 'templates'),
    (os.path.join(project_dir, 'static'), 'static'),
]

# PyQt6 데이터 추가
datas += pyqt_datas

# config 폴더가 있으면 포함
config_dir = os.path.join(project_dir, 'config')
if os.path.exists(config_dir):
    datas.append((config_dir, 'config'))

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
    'werkzeug.routing',
    'jinja2',
    'jinja2.ext',
    'markupsafe',
    
    # WSGI
    'waitress',
    'waitress.server',
    
    # DB & 스레딩
    'sqlite3',
    'threading',
    'queue',
    'concurrent.futures',
    'multiprocessing',
    
    # 문서 처리 (기본)
    'docx',
    'docx.shared',
    'pypdf',
    'openpyxl',
    'olefile',
    'lxml',
    'lxml.etree',
    
    # 시스템
    'psutil',
    'hashlib',
    'json',
    're',
    'pickle',
    'gc',
    'logging',
    
    # 인코딩 (한글 지원 필수)
    'encodings',
    'encodings.utf_8',
    'encodings.utf_8_sig',
    'encodings.cp949',
    'encodings.euc_kr',
    'encodings.utf_16',
    'encodings.utf_16_le',
    'encodings.ascii',
    'encodings.latin_1',
    'codecs',
    
    # PyQt6 (GUI 필수)
    'PyQt6',
    'PyQt6.QtWidgets',
    'PyQt6.QtCore',
    'PyQt6.QtGui',
    'PyQt6.sip',
    
    # collections.abc (누락 방지)
    'collections.abc',
    
    # App 모듈
    'app',
    'app.config',
    'app.constants',
    'app.exceptions',
    'app.utils',
    'app.routes',
    'app.routes.main_routes',
    'app.routes.api_search',
    'app.routes.api_files',
    'app.routes.api_system',
    'app.services',
    'app.services.search',
    'app.services.document',
    'app.services.db',
    'app.services.file_manager',
    'app.services.metadata',
]

# PyQt6 서브모듈 추가
hiddenimports += pyqt_hiddenimports

# encodings 전체 (안정성)
hiddenimports += collect_submodules('encodings')

# ============================================================================
# 제외 목록 (대폭 확대 - 핵심 경량화)
# ============================================================================
excludes = [
    # === AI/ML 전체 제외 ===
    'torch', 'torchvision', 'torchaudio',
    'tensorflow', 'keras', 'tensorboard',
    'transformers', 'huggingface_hub',
    'sentence_transformers',
    'langchain', 'langchain_core', 'langchain_community',
    'langchain_text_splitters', 'langchain_huggingface',
    'faiss', 'faiss_cpu',
    'accelerate', 'safetensors', 'tokenizers',
    'optimum', 'onnx', 'onnxruntime',
    
    # === GUI 프레임워크 (PyQt6는 유지) ===
    'tkinter', 'tk', '_tkinter',
    'PyQt5', 'PySide2', 'PySide6',
    'wx', 'wxPython',
    
    # === 데이터 분석/시각화 ===
    'matplotlib', 'pandas', 'scipy',
    'seaborn', 'plotly', 'bokeh',
    'numpy.testing', 'numpy.distutils',
    
    # === 개발 도구 ===
    'IPython', 'notebook', 'jupyter',
    'pytest', 'unittest', 'doctest',
    'setuptools', 'pip', 'wheel',
    
    # === 이미지/비디오 (OCR 제외) ===
    'cv2', 'opencv', 'PIL.ImageTk',
    'pytesseract', 'pdf2image',
    'imageio', 'moviepy',
    
    # === 네트워크 ===
    'scrapy', 'selenium', 'requests_html',
    'aiohttp', 'httpx',
    
    # === 기타 ===
    'email.test', 'test', 'tests', 'lib2to3',
    'sympy', 'numba', 'numexpr',
    'pydoc', 'doctest', 'xmlrpc',
]

# ============================================================================
# Analysis
# ============================================================================
a = Analysis(
    ['server_gui.py'],
    pathex=[project_dir],
    binaries=pyqt_binaries,
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
# 바이너리 필터링 (매우 공격적)
# ============================================================================
excluded_binaries = [
    # PyTorch/LibTorch
    'torch', 'libtorch', 'c10', 'caffe2',
    'torch_cpu', 'torch_cuda',
    'libc10', 'libshm', 'libnvfuser',
    'fbgemm', 'asmjit', 'dnnl', 'ideep',
    
    # CUDA/GPU
    'cuda', 'cudnn', 'cublas', 'cufft', 'cusparse', 'curand',
    'nvcuda', 'nvrtc', 'nvidia', 'nvjitlink',
    'cudart', 'cublaslt', 'cusolver',
    
    # Intel MKL
    'mkl_', 'mkl-', 'libmkl',
    'libiomp', 'svml', 'libimalloc',
    'tbb', 'libtbb',
    
    # OpenBLAS
    'openblas', 'libopenblas',
    
    # TensorFlow
    'tensorflow', 'libtensorflow',
    
    # OpenCV
    'opencv', 'cv2', 'libopencv',
    
    # Qt5 (PyQt6만 유지)
    'qt5', 'Qt5', 'pyside',
    
    # LLVM
    'llvm', 'clang', 'libllvm', 'libclang',
    
    # Transformers
    'transformers', 'tokenizers', 'sentencepiece',
    
    # 디버그
    '.pdb', '_d.dll', '_d.pyd',
]

print(f"[Onefile] 필터링 전 바이너리 수: {len(a.binaries)}")
a.binaries = [
    b for b in a.binaries 
    if not any(ex.lower() in b[0].lower() for ex in excluded_binaries)
]
print(f"[Onefile] 필터링 후 바이너리 수: {len(a.binaries)}")

# ============================================================================
# 데이터 필터링 (PyQt6 locale 등 제거)
# ============================================================================
excluded_datas = [
    'tcl', 'tk',
    'matplotlib', 'mpl-data',
    'torch', 'transformers', 'huggingface',
    'sentence_transformers', 'faiss',
    'share/jupyter', 'share/doc',
    'include/', '.dist-info',
    '__pycache__', 'tests/', 'test/',
    'share/locale',  # 다국어 리소스 제외
    'translations',
]

print(f"[Onefile] 필터링 전 데이터 수: {len(a.datas)}")
a.datas = [
    d for d in a.datas
    if not any(ex.lower() in d[0].lower() for ex in excluded_datas)
]
print(f"[Onefile] 필터링 후 데이터 수: {len(a.datas)}")

# ============================================================================
# PYZ
# ============================================================================
pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=block_cipher
)

# ============================================================================
# EXE - 단일 파일 모드
# ============================================================================
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,       # onefile 모드: 바이너리 포함
    a.zipfiles,       # onefile 모드: zipfiles 포함
    a.datas,          # onefile 모드: 데이터 포함
    [],
    name='사내규정검색기_Portable',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,      # strip 비활성화 (안정성)
    upx=True,         # UPX 활성화 (크기 절감)
    upx_exclude=[
        'vcruntime140.dll',
        'python3*.dll',
        'msvcp*.dll',
        'api-ms-*.dll',
        'ucrtbase.dll',
        'Qt*.dll',     # Qt DLL은 UPX 제외 (안정성)
    ],
    runtime_tmpdir=None,
    console=False,    # GUI 모드
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
