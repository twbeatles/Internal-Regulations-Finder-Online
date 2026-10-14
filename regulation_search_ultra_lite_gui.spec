# -*- mode: python ; coding: utf-8 -*-
"""
사내 규정 검색기 - Ultra Lite GUI 빌드 (PySide6 시스템 트레이)
빌드: pyinstaller regulation_search_ultra_lite_gui.spec
정합성 점검(2026-03-01): config 전체 포함 대신 settings.example.json만 포함
정합성 점검(2026-03-16): HWP/HWPX 파서 서브모듈 hiddenimports 보강
정합성 점검(2026-07-15): files.path_validation 등 hardening 서브모듈 명시

특징:
- GUI 모드 (콘솔 창 없음)
- 시스템 트레이 아이콘 지원
- AI/ML 완전 제외 (BM25 텍스트 검색만)
- UPX 미사용 (GUI 바이너리 손상 방지)
- 목표 크기: 80-100MB
"""

import os
import sys
from PyInstaller.utils.hooks import copy_metadata

block_cipher = None
project_dir = os.path.dirname(os.path.abspath(SPEC))

# ============================================================================
# 데이터 파일 (최소화)
# ============================================================================
datas = [
    (os.path.join(project_dir, 'templates'), 'templates'),
    (os.path.join(project_dir, 'static'), 'static'),
]

# settings.example.json만 포함
settings_example = os.path.join(project_dir, 'config', 'settings.example.json')
if os.path.exists(settings_example):
    datas.append((settings_example, 'config'))

# PySide6 플러그인 (플랫폼 플러그인 필수)
try:
    import PySide6
    pyside_dir = os.path.dirname(PySide6.__file__)
    plugins_dir = os.path.join(pyside_dir, 'Qt', 'plugins')
    
    # platforms 플러그인만 포함 (최소화)
    platforms_dir = os.path.join(plugins_dir, 'platforms')
    if os.path.exists(platforms_dir):
        datas.append((platforms_dir, 'PySide6/Qt/plugins/platforms'))
        print(f"[GUI] PySide6 platforms 플러그인 추가: {platforms_dir}")
        
    # styles 플러그인 추가 (네이티브 룩앤필)
    styles_dir = os.path.join(plugins_dir, 'styles')
    if os.path.exists(styles_dir):
        datas.append((styles_dir, 'PySide6/Qt/plugins/styles'))
except ImportError:
    print("[GUI] 경고: PySide6을 찾을 수 없습니다.")

# ============================================================================
# PySide6 바이너리 수집
# ============================================================================
pyside_binaries = []
pyside_hiddenimports = [
    'PySide6',
    'PySide6.QtCore',
    'PySide6.QtGui',
    'PySide6.QtWidgets',
    'shiboken6',
]

# ============================================================================
# Hidden Imports (필수 패키지만)
# ============================================================================
hiddenimports = [
    # Flask 핵심
    'flask',
    'flask.json',
    'flask.json.provider',
    'flask_cors',
    'flask_compress',
    'werkzeug',
    'werkzeug.serving',
    'werkzeug.routing',
    'jinja2',
    'jinja2.ext',
    'markupsafe',
    
    # 웹 서버
    'waitress',
    'waitress.server',
    
    # 표준 라이브러리
    'sqlite3',
    'threading',
    'queue',
    'concurrent.futures',
    'uuid',
    'datetime',
    'logging',
    'logging.handlers',
    'hashlib',
    'json',
    're',
    'os',
    'sys',
    'pathlib',
    'shutil',
    'tempfile',
    'webbrowser',
    'subprocess',
    
    # 기본 문서 파서 (AI 제외)
    'docx',
    'pypdf',
    'openpyxl',
    'olefile',
    
    # 시스템 유틸리티
    'psutil',
    
    # HWP/HWPX 파서 서브모듈
    'app.services.parsers.hwp_adapter',
    'app.services.parsers.hwpx_adapter',
    'app.services.parsers.hwp_models',
    'app.services.search.qa_facade',
    'app.services.search.hybrid_search',
    'app.services.search.index',
    'app.routes.files_request',
    'app.routes.api_revisions',
    'app.routes.api_tags',
    'app.services.files',
    'app.services.files.path_resolver',
    'app.services.files.path_validation',
    'app.services.files.delete_policy',
    'app.services.files.lock',
    'app.services.files.preview_service',
    'app.services.settings_store',
    'app.auth',
]

# PySide6 서브모듈 추가
hiddenimports += pyside_hiddenimports

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
    
    # === GUI 프레임워크 (PySide6는 제외하지 않음) ===
    'tkinter', 'tk', '_tkinter',
    'PyQt5', 'PyQt6', 'PySide2',  # PySide6 제외됨
    'wx', 'wxPython',
    
    # === 데이터 분석/시각화 ===
    'matplotlib', 'pandas', 'scipy',
    'seaborn', 'plotly', 'bokeh',
    'numpy.testing', 'numpy.distutils',
    
    # === 개발 도구 ===
    'IPython', 'notebook', 'jupyter',
    'pytest', 'unittest', 'doctest',
    'setuptools', 'pip', 'wheel',
    
    # === 이미지/비디오 ===
    'cv2', 'opencv', 'PIL.ImageTk',
    'pytesseract', 'pdf2image',
    'imageio', 'moviepy',
    
    # === 네트워크 ===
    'scrapy', 'selenium', 'requests_html',
    'aiohttp', 'httpx',
    
    # === 기타 ===
    'email.test', 'test', 'tests', 'lib2to3',
    'sympy', 'numba', 'numexpr',
]

# ============================================================================
# Analysis
# ============================================================================
a = Analysis(
    ['server_gui.py'],  # ⭐ GUI 엔트리포인트
    pathex=[project_dir],
    binaries=pyside_binaries,  # PySide6 바이너리 포함
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

# Python 3.15 abi3t: cryptography _rust.pyd links python3t.dll
# (<prefix>\abi3t-compat), which PyInstaller does not auto-collect.
# Without it the frozen exe dies with "DLL load failed while importing _rust".
_py3t_dll = os.path.join(sys.base_prefix, 'abi3t-compat', 'python3t.dll')
_py3t_dest = 'python3t.dll'
if os.path.isfile(_py3t_dll) and not any(
        (b[0] or '').lower() == _py3t_dest for b in a.binaries):
    a.binaries.append((_py3t_dest, _py3t_dll, 'BINARY'))

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
    # NumPy 2.x 실행에 필수이므로 OpenBLAS는 유지 (제거 시 _multiarray_umath 로드 실패)
    # 'openblas', 'libopenblas',
    
    # Qt (PySide6는 필요하므로 제외하지 않음)
    'qt5', 'pyqt',
    'Qt5',
    
    # TensorFlow
    'tensorflow', 'libtensorflow',
    
    # OpenCV
    'opencv', 'cv2', 'libopencv',
    
    # LLVM
    'llvm', 'clang', 'libllvm', 'libclang',
    
    # Transformers
    'transformers', 'tokenizers', 'sentencepiece',
    
    # 디버그
    '.pdb', '_d.dll', '_d.pyd',
]

print(f"[Ultra Lite] 필터링 전 바이너리 수: {len(a.binaries)}")
a.binaries = [
    b for b in a.binaries 
    if not any(ex.lower() in b[0].lower() for ex in excluded_binaries)
]
print(f"[Ultra Lite] 필터링 후 바이너리 수: {len(a.binaries)}")

# ============================================================================
# 데이터 필터링
# ============================================================================
excluded_datas = [
    'tcl', 'tk',
    'matplotlib', 'mpl-data',
    'torch', 'transformers', 'huggingface',
    'sentence_transformers', 'faiss',
    'share/jupyter', 'share/doc',
    'include/', '.dist-info',
    '__pycache__', 'tests/', 'test/',
    'share/locale',
]

print(f"[Ultra Lite] 필터링 전 데이터 수: {len(a.datas)}")
a.datas = [
    d for d in a.datas
    if not any(ex.lower() in d[0].lower() for ex in excluded_datas)
]
print(f"[Ultra Lite] 필터링 후 데이터 수: {len(a.datas)}")

# .dist-info 일괄 제외로 날아간 실행 필수 메타데이터 복원
# (werkzeug serving이 importlib.metadata.version() 호출)
for _pkg in ('werkzeug', 'flask'):
    try:
        for _src, _dest in copy_metadata(_pkg):
            if os.path.isdir(_src):
                for _root, _, _files in os.walk(_src):
                    for _fn in _files:
                        _full = os.path.join(_root, _fn)
                        a.datas.append((os.path.join(_dest, os.path.relpath(_full, _src)), _full, 'DATA'))
            else:
                a.datas.append((_dest, _src, 'DATA'))
    except Exception:
        pass

# ============================================================================
# PYZ
# ============================================================================
pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=block_cipher
)

# ============================================================================
# EXE - GUI 모드
# ============================================================================
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='사내규정검색기_Lite',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,      # ⭐ GUI 모드
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
    strip=False,
    upx=False,
    upx_exclude=[
        'vcruntime140.dll',
        'python3*.dll',
        'msvcp*.dll',
        'api-ms-*.dll',
        'ucrtbase.dll',
    ],
    name='사내규정검색기_Lite',
)
