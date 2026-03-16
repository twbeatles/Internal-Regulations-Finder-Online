# -*- mode: python ; coding: utf-8 -*-
"""
Internal Regulations Finder - GUI server build

Build:
    pyinstaller server_gui.spec

Consistency notes:
    2026-03-16 - explicit parser hiddenimports added for HWP/HWPX extraction path
"""

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
    # Flask / web
    "flask",
    "flask.sessions",
    "flask.json",
    "flask.json.provider",
    "flask_cors",
    "flask_compress",
    "werkzeug",
    "werkzeug.utils",
    "werkzeug.security",
    "jinja2",
    "markupsafe",
    "waitress",
    "waitress.server",

    # PyQt6
    "PyQt6",
    "PyQt6.QtCore",
    "PyQt6.QtGui",
    "PyQt6.QtWidgets",
    "PyQt6.sip",

    # AI / search
    "langchain",
    "langchain.schema",
    "langchain_community",
    "langchain_community.vectorstores",
    "langchain_huggingface",
    "langchain_huggingface.embeddings",
    "langchain_text_splitters",
    "langchain_core",
    "langchain_core.documents",
    "transformers",
    "sentence_transformers",
    "huggingface_hub",
    "tokenizers",
    "faiss",
    "torch",
    "torch.nn",
    "torch.utils",
    "onnxruntime",

    # Documents / utilities
    "docx",
    "docx.document",
    "docx.opc",
    "docx.oxml",
    "pypdf",
    "pypdf.generic",
    "lxml",
    "lxml.etree",
    "openpyxl",
    "olefile",
    "psutil",
    "numpy",
    "tqdm",
    "requests",
    "hashlib",
    "json",
    "dataclasses",
    "sqlite3",
    "concurrent",
    "concurrent.futures",
    "watchdog",
    "watchdog.observers",
    "watchdog.events",
]

try:
    hiddenimports += collect_submodules("PyQt6")
except Exception:
    pass

try:
    hiddenimports += collect_submodules("app.services.embeddings_backends")
    hiddenimports += collect_submodules("app.services.parsers")
    hiddenimports += collect_submodules("docx")
    hiddenimports += collect_submodules("pypdf")
except Exception:
    pass

excludes = [
    "tkinter",
    "PyQt5",
    "PySide2",
    "PySide6",
    "wx",
    "kivy",
    "matplotlib",
    "scipy",
    "pandas",
    "sklearn",
    "scikit-learn",
    "cv2",
    "opencv",
    "PIL",
    "pillow",
    "IPython",
    "notebook",
    "jupyter",
    "pytest",
    "unittest",
    "sphinx",
    "docutils",
    "pydoc",
    "torch.distributions",
    "torch.testing",
    "transformers.data",
    "transformers.pipelines",
    "email.test",
    "test",
    "tests",
    "setuptools",
    "pkg_resources",
    "cython",
    "numba",
]

a = Analysis(
    ["server_gui.py"],
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
    console=False,
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
    upx_exclude=["python*.dll", "vcruntime*.dll", "Qt*.dll"],
    name="사내규정검색기",
)
