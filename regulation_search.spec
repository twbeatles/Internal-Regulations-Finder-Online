# -*- mode: python ; coding: utf-8 -*-
"""
Internal Regulations Finder - full console build

Build:
    pyinstaller regulation_search.spec
"""

import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None
project_dir = os.path.dirname(os.path.abspath(SPEC))

datas = [
    (os.path.join(project_dir, "templates"), "templates"),
    (os.path.join(project_dir, "static"), "static"),
    (os.path.join(project_dir, "app"), "app"),
]

settings_example = os.path.join(project_dir, "config", "settings.example.json")
if os.path.exists(settings_example):
    datas.append((settings_example, "config"))

hiddenimports = [
    # Flask core
    "flask",
    "flask.json",
    "flask.json.provider",
    "flask_cors",
    "flask_compress",
    "werkzeug",
    "werkzeug.serving",
    "werkzeug.routing",
    "jinja2",
    "jinja2.ext",
    "markupsafe",

    # WSGI
    "waitress",
    "waitress.server",

    # Core
    "sqlite3",
    "threading",
    "queue",
    "concurrent.futures",

    # Document handling
    "docx",
    "docx.shared",
    "pypdf",
    "openpyxl",
    "olefile",
    "lxml",
    "lxml.etree",

    # AI / embeddings
    "torch",
    "sentence_transformers",
    "transformers",
    "huggingface_hub",
    "langchain",
    "langchain_core",
    "langchain_core.documents",
    "langchain_community",
    "langchain_community.embeddings",
    "langchain_community.vectorstores",
    "langchain_text_splitters",
    "langchain_huggingface",
    "faiss",
    "onnxruntime",

    # Misc
    "psutil",
    "hashlib",
    "json",
    "re",
    "pickle",
    "gc",
    "encodings.utf_8",
    "encodings.cp949",
    "encodings.euc_kr",
]

hiddenimports += collect_submodules("app")
hiddenimports += collect_submodules("app.routes")
hiddenimports += collect_submodules("app.services")
hiddenimports += collect_submodules("app.services.embeddings_backends")

excludes = [
    "tkinter",
    "tk",
    "_tkinter",
    "PyQt5",
    "PyQt6",
    "PySide2",
    "PySide6",
    "wx",
    "wxPython",
    "matplotlib",
    "matplotlib.pyplot",
    "pandas",
    "scipy",
    "scipy.sparse",
    "seaborn",
    "plotly",
    "bokeh",
    "numpy.testing",
    "sympy",
    "IPython",
    "ipykernel",
    "ipywidgets",
    "notebook",
    "jupyter",
    "jupyter_client",
    "pytest",
    "unittest",
    "doctest",
    "scrapy",
    "selenium",
    "requests_html",
    "cv2",
    "opencv",
    "moviepy",
    "imageio",
    "pygame",
    "pyglet",
    "email.test",
    "test",
    "tests",
    "lib2to3",
    "tensorflow",
    "keras",
    "tensorboard",
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

excluded_binaries = [
    "qt5",
    "qt6",
    "pyqt",
    "pyside",
    "opencv",
    "cv2",
    "cuda",
    "cudnn",
    "cublas",
    "cufft",
    "cusparse",
    "nvcuda",
    "nvrtc",
    "mkl_",
    "libiomp",
    "libiomk",
    "libtorch_cuda",
    "torch_cuda",
    "caffe2_nvrtc",
    "libtensorflow",
    "tensorflow",
    ".pdb",
    "_d.dll",
]
a.binaries = [b for b in a.binaries if not any(token in b[0].lower() for token in excluded_binaries)]

excluded_datas = ["tcl", "tk", "matplotlib", "share/jupyter", "share/doc"]
a.datas = [d for d in a.datas if not any(token in d[0].lower() for token in excluded_datas)]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="사내규정검색기",
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
    version=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=True,
    upx=True,
    upx_exclude=[
        "vcruntime140.dll",
        "python*.dll",
        "msvcp*.dll",
        "api-ms-*.dll",
    ],
    name="사내규정검색기",
)
