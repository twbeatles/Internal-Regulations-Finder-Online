# -*- mode: python ; coding: utf-8 -*-
"""
사내 규정 검색기 v2.5 - GUI 빌드 Spec (안정화 버전)
빌드: pyinstaller regulation_search_gui.spec --clean

특징:
- GUI 모드 (콘솔 창 없음)
- AI/Vector 검색 포함
- Python encodings 모듈 완전 포함
- PyQt6 GUI 지원

예상 크기: 500-800MB (AI 포함)
"""

import os
import sys
from PyInstaller.utils.hooks import collect_submodules, collect_data_files, collect_dynamic_libs

block_cipher = None
project_dir = os.path.dirname(os.path.abspath(SPEC))

# ============================================================================
# 동적 라이브러리 수집
# ============================================================================
torch_binaries = []
try:
    torch_binaries = collect_dynamic_libs('torch')
    print(f"[INFO] torch 라이브러리: {len(torch_binaries)}개")
except Exception as e:
    print(f"[WARNING] torch 수집 실패: {e}")

# PyQt6 바이너리
pyqt_binaries = []
try:
    pyqt_binaries = collect_dynamic_libs('PyQt6')
    print(f"[INFO] PyQt6 라이브러리: {len(pyqt_binaries)}개")
except Exception as e:
    print(f"[WARNING] PyQt6 수집 실패: {e}")

all_binaries = torch_binaries + pyqt_binaries

# ============================================================================
# 데이터 파일
# ============================================================================
datas = [
    (os.path.join(project_dir, 'templates'), 'templates'),
    (os.path.join(project_dir, 'static'), 'static'),
]

# config 폴더
config_dir = os.path.join(project_dir, 'config')
if os.path.exists(config_dir):
    datas.append((config_dir, 'config'))

# PyQt6 데이터
try:
    datas += collect_data_files('PyQt6')
except Exception:
    pass

# ============================================================================
# Hidden Imports - 포괄적 목록
# ============================================================================
hiddenimports = [
    # === Python 표준 라이브러리 (encodings 오류 방지) ===
    'encodings',
    'encodings.aliases',
    'encodings.ascii',
    'encodings.base64_codec',
    'encodings.big5',
    'encodings.big5hkscs',
    'encodings.charmap',
    'encodings.cp437',
    'encodings.cp949',
    'encodings.cp950',
    'encodings.euc_jp',
    'encodings.euc_kr',
    'encodings.gb2312',
    'encodings.gbk',
    'encodings.hz',
    'encodings.idna',
    'encodings.iso2022_jp',
    'encodings.iso2022_kr',
    'encodings.iso8859_1',
    'encodings.johab',
    'encodings.koi8_r',
    'encodings.latin_1',
    'encodings.mbcs',
    'encodings.palmos',
    'encodings.punycode',
    'encodings.raw_unicode_escape',
    'encodings.rot_13',
    'encodings.shift_jis',
    'encodings.unicode_escape',
    'encodings.utf_16',
    'encodings.utf_16_be',
    'encodings.utf_16_le',
    'encodings.utf_32',
    'encodings.utf_32_be',
    'encodings.utf_32_le',
    'encodings.utf_7',
    'encodings.utf_8',
    'encodings.utf_8_sig',
    'codecs',
    
    # === PyQt6 (GUI) ===
    'PyQt6',
    'PyQt6.QtWidgets',
    'PyQt6.QtCore',
    'PyQt6.QtGui',
    'PyQt6.sip',
    
    # === Flask 코어 ===
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
    
    # === WSGI 서버 ===
    'waitress',
    'waitress.server',
    
    # === DB & 스레딩 ===
    'sqlite3',
    'threading',
    'queue',
    'concurrent.futures',
    'multiprocessing',
    
    # === 문서 처리 ===
    'docx',
    'docx.shared',
    'pypdf',
    'openpyxl',
    'olefile',
    'lxml',
    'lxml.etree',
    
    # === AI/ML ===
    'torch',
    'sentence_transformers',
    'transformers',
    'huggingface_hub',
    
    # === LangChain ===
    'langchain',
    'langchain_core',
    'langchain_core.documents',
    'langchain_community',
    'langchain_community.embeddings',
    'langchain_community.vectorstores',
    'langchain_text_splitters',
    'langchain_huggingface',
    
    # === FAISS ===
    'faiss',
    
    # === 시스템 ===
    'psutil',
    'hashlib',
    'json',
    're',
    'pickle',
    'gc',
    'logging',
    
    # === pydantic ===
    'pydantic',
    'pydantic.deprecated.decorator',
]

# app 패키지 전체
hiddenimports += collect_submodules('app')
hiddenimports += collect_submodules('app.routes')
hiddenimports += collect_submodules('app.services')

# encodings 전체 (안전을 위해)
hiddenimports += collect_submodules('encodings')

# ============================================================================
# 제외 목록 (최소화 - 안정성 우선)
# ============================================================================
excludes = [
    # 다른 GUI 프레임워크만 제외
    'tkinter', 'tk', '_tkinter',
    'PyQt5', 'PySide2', 'PySide6',
    'wx', 'wxPython',
    
    # 불필요한 대형 패키지
    'matplotlib',
    'pandas',
    'scipy',
    'tensorflow',
    'keras',
    
    # 개발 도구
    'IPython',
    'notebook',
    'jupyter',
    'pytest',
]

# ============================================================================
# Analysis
# ============================================================================
a = Analysis(
    ['server_gui.py'],
    pathex=[project_dir],
    binaries=all_binaries,
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
# 바이너리 필터링 (보수적 - CUDA만 제외)
# ============================================================================
excluded_binaries = [
    # CUDA/GPU만 제외 (CPU 전용 빌드)
    'cuda', 'cudnn', 'cublas', 'cufft', 'cusparse', 'curand',
    'nvcuda', 'nvrtc', 'nvidia', 'nvjitlink',
    'torch_cuda', 'libtorch_cuda',
    'caffe2_nvrtc',
]

print(f"[INFO] 필터링 전 바이너리: {len(a.binaries)}개")
a.binaries = [
    b for b in a.binaries 
    if not any(ex.lower() in b[0].lower() for ex in excluded_binaries)
]
print(f"[INFO] 필터링 후 바이너리: {len(a.binaries)}개")

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
    name='사내규정검색기',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,        # strip 비활성화 (안정성)
    upx=False,          # UPX 비활성화 (안정성)
    console=False,      # GUI 모드
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
    upx_exclude=[],
    name='사내규정검색기',
)
