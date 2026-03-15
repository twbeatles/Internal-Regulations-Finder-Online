# -*- mode: python ; coding: utf-8 -*-
"""
Internal Regulations Finder - full console build

Build:
    pyinstaller internal_regulations.spec
"""

import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

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
    # Flask / server
    "flask",
    "flask.json",
    "flask.json.provider",
    "flask_cors",
    "flask_compress",
    "werkzeug",
    "jinja2",
    "waitress",
    "waitress.server",

    # Core dependencies
    "watchdog",
    "openpyxl",
    "olefile",
    "pytesseract",
    "pdf2image",
    "PIL",
    "sqlite3",
    "numpy",
    "tqdm",
    "requests",
    "psutil",
    "lxml",

    # AI / LangChain
    "langchain",
    "langchain_community",
    "langchain_huggingface",
    "langchain_text_splitters",
    "langchain_core",
    "langchain_core.documents",
    "faiss",
    "torch",
    "sentence_transformers",
    "transformers",
    "onnxruntime",
]

hiddenimports += collect_submodules("app")
hiddenimports += collect_submodules("app.services.embeddings_backends")
hiddenimports += collect_submodules("langchain")
hiddenimports += collect_submodules("langchain_community")
hiddenimports += collect_submodules("langchain_huggingface")

try:
    datas += collect_data_files("langchain_community")
except Exception:
    pass

a = Analysis(
    ["run.py"],
    pathex=[project_dir],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "matplotlib",
        "cv2",
        "pandas",
        "IPython",
        "notebook",
        "jupyter",
        "pytest",
        "PyQt6",
        "PyQt5",
        "wx",
        "unittest",
        "email.test",
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
    name="InternalRegulationsFinder",
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
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="InternalRegulationsFinder",
)
