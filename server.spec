# -*- mode: python ; coding: utf-8 -*-
"""
Internal Regulations Finder - standalone server build

Build:
    pyinstaller server.spec
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
    "flask",
    "flask.json",
    "flask.json.provider",
    "flask_cors",
    "flask_compress",
    "werkzeug",
    "werkzeug.utils",
    "jinja2",
    "markupsafe",
    "waitress",
    "waitress.server",
    "langchain",
    "langchain.text_splitter",
    "langchain_text_splitters",
    "langchain_core",
    "langchain_core.documents",
    "langchain.docstore",
    "langchain.docstore.document",
    "langchain_community",
    "langchain_community.embeddings",
    "langchain_community.vectorstores",
    "langchain_community.vectorstores.faiss",
    "langchain_huggingface",
    "transformers",
    "sentence_transformers",
    "huggingface_hub",
    "tokenizers",
    "faiss",
    "torch",
    "torch.nn",
    "torch.utils",
    "docx",
    "pypdf",
    "psutil",
    "numpy",
    "tqdm",
    "requests",
    "urllib3",
    "certifi",
    "charset_normalizer",
    "idna",
    "packaging",
    "regex",
    "safetensors",
    "filelock",
    "fsspec",
    "yaml",
    "pyyaml",
]

try:
    hiddenimports += collect_submodules("sentence_transformers")
except Exception:
    pass

try:
    hiddenimports += collect_submodules("transformers")
except Exception:
    pass

try:
    datas += collect_data_files("torch")
except Exception:
    pass

try:
    datas += collect_data_files("transformers")
except Exception:
    pass

try:
    datas += collect_data_files("sentence_transformers")
except Exception:
    pass

a = Analysis(
    ["server.py"],
    pathex=[project_dir],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "PyQt5",
        "PyQt6",
        "PySide2",
        "PySide6",
        "matplotlib",
        "PIL",
        "cv2",
        "scipy",
        "pandas",
        "IPython",
        "notebook",
        "jupyter",
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
    name="사내규정검색기서버",
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
    upx_exclude=[],
    name="사내규정검색기서버",
)
