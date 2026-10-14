# -*- mode: python ; coding: utf-8 -*-
"""
Internal Regulations Finder - lite console build

Build:
    pyinstaller internal_regulations_lite.spec
"""

import sys
import os
from PyInstaller.utils.hooks import collect_submodules

block_cipher = None
project_dir = os.path.dirname(os.path.abspath(SPEC))

datas = [
    (os.path.join(project_dir, "templates"), "templates"),
    (os.path.join(project_dir, "static"), "static"),
]

settings_example = os.path.join(project_dir, "config", "settings.example.json")
if os.path.exists(settings_example):
    datas.append((settings_example, "config"))

hiddenimports = [
    "flask",
    "flask.json",
    "flask.json.provider",
    "flask_cors",
    "flask_compress",
    "werkzeug",
    "jinja2",
    "waitress",
    "waitress.server",
    "sqlite3",
    "threading",
    "docx",
    "pypdf",
    "openpyxl",
    "olefile",
    "psutil",
]

hiddenimports += collect_submodules("app")

excludes = [
    "tkinter",
    "tk",
    "PyQt5",
    "PyQt6",
    "PySide2",
    "PySide6",
    "wx",
    "wxPython",
    "matplotlib",
    "pandas",
    "scipy",
    "seaborn",
    "plotly",
    "IPython",
    "notebook",
    "jupyter",
    "pytest",
    "unittest",
    "setuptools",
    "pkg_resources",
    "torch",
    "tensorflow",
    "keras",
    "transformers",
    "huggingface_hub",
    "cv2",
    "PIL.ImageTk",
    "email.test",
    "test",
]

a = Analysis(
    ["run.py"],
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

# Python 3.15 abi3t: cryptography _rust.pyd links python3t.dll
# (<prefix>\abi3t-compat), which PyInstaller does not auto-collect.
# Without it the frozen exe dies with "DLL load failed while importing _rust".
_py3t_dll = os.path.join(sys.base_prefix, 'abi3t-compat', 'python3t.dll')
_py3t_dest = 'python3t.dll'
if os.path.isfile(_py3t_dll) and not any(
        (b[0] or '').lower() == _py3t_dest for b in a.binaries):
    a.binaries.append((_py3t_dest, _py3t_dll, 'BINARY'))

a.binaries = [
    b for b in a.binaries if not any(
        token in b[0].lower()
        for token in ["qt5", "qt6", "opencv", "torch", "cuda", "mkl_", "libiomp", "libtorch"]
    )
]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="사내규정검색기",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=["vcruntime140.dll", "python*.dll"],
    name="사내규정검색기",
)
