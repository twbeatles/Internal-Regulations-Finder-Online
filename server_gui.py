# -*- coding: utf-8 -*-
"""
사내 규정 검색기 - 서버 GUI (PyQt6)
시스템 트레이 + Windows 시작 프로그램 등록 지원
로그 기능 강화 + 관리자 비밀번호 설정
"""

import sys
import os
import threading
import webbrowser
import winreg
import ctypes
import json
import hashlib
import logging
from typing import Optional
from datetime import datetime

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSystemTrayIcon, QMenu, QMessageBox,
    QCheckBox, QGroupBox, QTextEdit, QFrame, QLineEdit, QDialog,
    QDialogButtonBox, QFormLayout, QFileDialog, QSplashScreen, QProgressBar
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread
from PyQt6.QtGui import QIcon, QAction, QFont, QColor, QPalette, QCloseEvent, QTextCharFormat, QPixmap

# ============================================================================
# 지연 로딩 패턴 - 무거운 모듈은 스플래시 후에 로드
# ============================================================================
# 전역 변수 (나중에 로드됨)
app = None  # Flask app
qa_system = None
logger = None
AppConfig = None
UPLOAD_DIR = None

def _load_heavy_modules():
    """무거운 모듈 로드 (백그라운드에서 실행)"""
    global app, qa_system, logger, AppConfig, UPLOAD_DIR
    
    # 서버 모듈 import
    from app import create_app
    from app.config import AppConfig as _AppConfig
    from app.utils import logger as _logger, get_app_directory
    from app.services.search import qa_system as _qa_system
    
    AppConfig = _AppConfig
    logger = _logger
    qa_system = _qa_system
    
    # Qt 로그 핸들러 연결 (logger 로드 후)
    _setup_qt_log_handler()
    
    # Flask 앱 생성
    app = create_app()
    
    # 업로드 디렉토리
    UPLOAD_DIR = os.path.join(get_app_directory(), 'uploads')
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    return True


def initialize_server():
    """서버 초기화 - 모델 로드 및 문서 처리"""
    # 이 함수는 heavy modules 로드 후 호출됨
    from app.utils import get_app_directory
    
    logger.info("서버 초기화 시작...")
    try:
        # 설정 로드
        settings_path = os.path.join(get_app_directory(), 'config', 'settings.json')
        if os.path.exists(settings_path):
            with open(settings_path, 'r', encoding='utf-8') as f:
                settings = json.load(f)
            
            folder = settings.get('folder', '')
            offline_mode = settings.get('offline_mode', False)
            local_model_path = settings.get('local_model_path', '')
            
            # 오프라인 모드 설정
            AppConfig.OFFLINE_MODE = offline_mode
            AppConfig.LOCAL_MODEL_PATH = local_model_path
            
            if folder and os.path.exists(folder):
                logger.info(f"문서 폴더 초기화: {folder}")
                qa_system.initialize(folder)
            else:
                # 폴더 없어도 모델은 로드 시도 (실패해도 BM25로 fallback)
                try:
                    qa_system.load_model(AppConfig.DEFAULT_MODEL)
                except Exception as e:
                    logger.warning(f"AI 모델 로드 실패 (BM25 모드로 동작): {e}")
        else:
            # 기본 모델 로드 시도 (실패해도 BM25로 fallback)
            try:
                qa_system.load_model(AppConfig.DEFAULT_MODEL)
            except Exception as e:
                logger.warning(f"AI 모델 로드 실패 (BM25 모드로 동작): {e}")
        
        logger.info("서버 초기화 완료")
    except Exception as e:
        logger.error(f"서버 초기화 오류: {e}")

def graceful_shutdown():
    """서버 정리 종료 - 리소스 정리"""
    import gc
    
    if logger:
        logger.info("서버 종료 중...")
    
    try:
        # QA 시스템 정리 (모델, 벡터 스토어, 캐시)
        if qa_system:
            qa_system.cleanup()
        
        # DB 연결 정리
        from app.services.db import DBManager
        DBManager.close_all()
        
        # 가비지 컬렉션
        gc.collect()
        
        if logger:
            logger.info("서버 종료 완료")
    except Exception as e:
        if logger:
            logger.error(f"종료 중 오류: {e}")


# ============================================================================
# 상수
# ============================================================================
APP_NAME = "사내 규정 검색기 서버"
APP_VERSION = "1.4"
REGISTRY_KEY = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Run"
REGISTRY_VALUE_NAME = "RegulationSearchServer"


# ============================================================================
# 설정 관리자
# ============================================================================
class SettingsManager:
    """설정 파일 관리 (비밀번호, 오프라인 모드 등)"""
    
    def __init__(self):
        self.settings_dir = os.path.join(
            os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) 
            else os.path.dirname(os.path.abspath(__file__)),
            'config'
        )
        os.makedirs(self.settings_dir, exist_ok=True)
        self.settings_file = os.path.join(self.settings_dir, 'settings.json')
        self._settings = self._load()
    
    def _load(self) -> dict:
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {
            'admin_password_hash': '', 
            'server_port': 8080,
            'offline_mode': False,
            'local_model_path': ''
        }
    
    def _save(self):
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(self._settings, f, ensure_ascii=False, indent=2)
        except IOError as e:
            # logger가 아직 로드되지 않았을 수 있음
            if logger:
                logger.error(f"설정 저장 실패: {e}")
            else:
                print(f"[ERROR] 설정 저장 실패: {e}")
    
    def get_server_port(self) -> int:
        """서버 포트 반환 (기본값: 8080)"""
        return self._settings.get('server_port', 8080)
    
    def set_server_port(self, port: int):
        """서버 포트 설정"""
        self._settings['server_port'] = port
        self._save()
    
    def set_admin_password(self, password: str):
        """관리자 비밀번호 설정 (해시로 저장)"""
        if password:
            pw_hash = hashlib.sha256(password.encode('utf-8')).hexdigest()
            self._settings['admin_password_hash'] = pw_hash
        else:
            self._settings['admin_password_hash'] = ''
        self._save()
    
    def verify_admin_password(self, password: str) -> bool:
        """관리자 비밀번호 검증"""
        stored_hash = self._settings.get('admin_password_hash', '')
        if not stored_hash:
            return True  # 비밀번호 미설정 시 통과
        pw_hash = hashlib.sha256(password.encode('utf-8')).hexdigest()
        return pw_hash == stored_hash
    
    def has_admin_password(self) -> bool:
        """비밀번호 설정 여부"""
        return bool(self._settings.get('admin_password_hash', ''))
    
    def get_password_hash(self) -> str:
        """저장된 비밀번호 해시 반환"""
        return self._settings.get('admin_password_hash', '')
    
    # ========== 오프라인 모드 설정 (폐쇄망 지원) ==========
    
    def get_offline_mode(self) -> bool:
        """오프라인 모드 활성화 여부 반환"""
        return self._settings.get('offline_mode', False)
    
    def set_offline_mode(self, enabled: bool):
        """오프라인 모드 설정"""
        self._settings['offline_mode'] = enabled
        self._save()
    
    def get_local_model_path(self) -> str:
        """로컬 모델 경로 반환 (빈 문자열이면 기본 경로 사용)"""
        return self._settings.get('local_model_path', '')
    
    def set_local_model_path(self, path: str):
        """로컬 모델 경로 설정"""
        self._settings['local_model_path'] = path
        self._save()


# 전역 설정 관리자 (Lazy Initialization - PyInstaller 호환)
_settings_manager = None

def get_settings_manager():
    """설정 관리자 인스턴스 반환 (lazy init)"""
    global _settings_manager
    if _settings_manager is None:
        _settings_manager = SettingsManager()
    return _settings_manager

# 하위 호환성을 위한 프로퍼티
class _SettingsProxy:
    """settings_manager 지연 접근 프록시"""
    def __getattr__(self, name):
        return getattr(get_settings_manager(), name)

settings_manager = _SettingsProxy()


# ============================================================================
# 비밀번호 설정 다이얼로그
# ============================================================================
class PasswordDialog(QDialog):
    def __init__(self, parent=None, is_change: bool = False):
        super().__init__(parent)
        self.setWindowTitle("관리자 비밀번호 설정")
        self.setMinimumWidth(350)
        
        layout = QVBoxLayout(self)
        form = QFormLayout()
        
        if is_change:
            self.current_pw = QLineEdit()
            self.current_pw.setEchoMode(QLineEdit.EchoMode.Password)
            self.current_pw.setPlaceholderText("현재 비밀번호 입력")
            form.addRow("현재 비밀번호:", self.current_pw)
        else:
            self.current_pw = None
        
        self.new_pw = QLineEdit()
        self.new_pw.setEchoMode(QLineEdit.EchoMode.Password)
        self.new_pw.setPlaceholderText("새 비밀번호 입력")
        form.addRow("새 비밀번호:", self.new_pw)
        
        self.confirm_pw = QLineEdit()
        self.confirm_pw.setEchoMode(QLineEdit.EchoMode.Password)
        self.confirm_pw.setPlaceholderText("비밀번호 확인")
        form.addRow("비밀번호 확인:", self.confirm_pw)
        
        layout.addLayout(form)
        
        # 안내
        hint = QLabel("비밀번호를 비워두면 보호가 해제됩니다.")
        hint.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(hint)
        
        # 버튼
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def validate_and_accept(self):
        # 현재 비밀번호 확인 (변경 시)
        if self.current_pw and settings_manager.has_admin_password():
            if not settings_manager.verify_admin_password(self.current_pw.text()):
                QMessageBox.warning(self, "오류", "현재 비밀번호가 일치하지 않습니다.")
                return
        
        # 새 비밀번호 확인
        if self.new_pw.text() != self.confirm_pw.text():
            QMessageBox.warning(self, "오류", "새 비밀번호가 일치하지 않습니다.")
            return
        
        self.accept()
    
    def get_password(self) -> str:
        return self.new_pw.text()


# ============================================================================
# 포트 설정 다이얼로그
# ============================================================================
class PortDialog(QDialog):
    def __init__(self, parent=None, current_port: int = 8080):
        super().__init__(parent)
        self.setWindowTitle("서버 포트 설정")
        self.setMinimumWidth(300)
        
        layout = QVBoxLayout(self)
        
        form = QFormLayout()
        self.port_edit = QLineEdit(str(current_port))
        self.port_edit.setPlaceholderText("예: 8080")
        form.addRow("서버 포트:", self.port_edit)
        layout.addLayout(form)
        
        # 안내
        hint = QLabel("포트 변경 후에는 서버를 재시작해야 적용됩니다.\n(기본값: 8080)")
        hint.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(hint)
        
        # 버튼
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def validate_and_accept(self):
        try:
            port = int(self.port_edit.text())
            if not (1024 <= port <= 65535):
                raise ValueError("포트는 1024~65535 사이여야 합니다.")
            if is_port_in_use(port) and port != settings_manager.get_server_port():
                QMessageBox.warning(self, "오류", f"포트 {port}는 이미 사용 중입니다.")
                return
            self.accept()
        except ValueError as e:
            QMessageBox.warning(self, "오류", "올바른 포트 번호를 입력하세요 (1024~65535).")
    
    def get_port(self) -> int:
        return int(self.port_edit.text())



# ============================================================================
# 로그 시그널 핸들러
# ============================================================================
class LogSignal(QObject):
    log_received = pyqtSignal(str, str)  # message, level


log_signal = LogSignal()


class QtLogHandler(logging.Handler):
    """Qt 시그널로 로그 전송 (logging.Handler 상속)"""
    
    def __init__(self):
        super().__init__()
        self.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    def emit(self, record):
        try:
            msg = self.format(record)
            log_signal.log_received.emit(msg, record.levelname)
        except Exception:
            pass


# QtLogHandler는 heavy modules 로드 후 연결됨
qt_handler = None

def _setup_qt_log_handler():
    """Qt 로그 핸들러 설정 (heavy modules 로드 후 호출)"""
    global qt_handler
    if logger and qt_handler is None:
        qt_handler = QtLogHandler()
        qt_handler.setLevel(logging.DEBUG)
        logger.addHandler(qt_handler)


# ============================================================================
# 자동 시작 관리
# ============================================================================
class AutoStartManager:
    @staticmethod
    def get_executable_path() -> str:
        """실행 파일 경로 반환"""
        if getattr(sys, 'frozen', False):
            return sys.executable
        return f'"{sys.executable}" "{os.path.abspath(__file__)}"'
    
    @staticmethod
    def is_enabled() -> bool:
        """자동 시작 활성화 여부 확인"""
        try:
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                REGISTRY_KEY,
                0,
                winreg.KEY_READ
            )
            try:
                winreg.QueryValueEx(key, REGISTRY_VALUE_NAME)
                return True
            except FileNotFoundError:
                return False
            finally:
                winreg.CloseKey(key)
        except WindowsError:
            return False
    
    @staticmethod
    def enable() -> bool:
        """자동 시작 활성화"""
        try:
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                REGISTRY_KEY,
                0,
                winreg.KEY_SET_VALUE
            )
            exe_path = AutoStartManager.get_executable_path()
            # --minimized 옵션으로 최소화 시작
            winreg.SetValueEx(
                key,
                REGISTRY_VALUE_NAME,
                0,
                winreg.REG_SZ,
                f'{exe_path} --minimized'
            )
            winreg.CloseKey(key)
            logger.info("자동 시작 등록 완료")
            return True
        except WindowsError as e:
            logger.error(f"자동 시작 등록 실패: {e}")
            return False
    
    @staticmethod
    def disable() -> bool:
        """자동 시작 비활성화"""
        try:
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                REGISTRY_KEY,
                0,
                winreg.KEY_SET_VALUE
            )
            try:
                winreg.DeleteValue(key, REGISTRY_VALUE_NAME)
            except FileNotFoundError:
                pass
            winreg.CloseKey(key)
            logger.info("자동 시작 해제 완료")
            return True
        except WindowsError as e:
            logger.error(f"자동 시작 해제 실패: {e}")
            return False


# ============================================================================
# 포트 체크 함수
# ============================================================================
def is_port_in_use(port: int, host: str = '127.0.0.1') -> bool:
    """포트가 사용 중인지 확인 (connect 기반)"""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    try:
        result = sock.connect_ex((host, port))
        sock.close()
        # 0이면 연결 성공 = 포트 사용 중
        return result == 0
    except Exception:
        return False


# ============================================================================
# 서버 스레드
# ============================================================================
class ServerThread(threading.Thread):
    def __init__(self, host: str, port: int):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.server = None
        self._stop_event = threading.Event()
        self._running = False
        self._error = None
    
    @property
    def is_running(self) -> bool:
        return self._running and self.is_alive()
    
    @property 
    def last_error(self) -> Optional[str]:
        return self._error
    
    def run(self):
        try:
            logger.info(f"🚀 서버 스레드 시작 (포트: {self.port})")
            logger.info(f"🚀 서버 초기화 시작 (백그라운드)...")
            
            # 서버 초기화 (비동기 실행)
            # 모델 로딩 등으로 인해 시간이 걸리므로 별도 스레드로 실행
            # 그래야 서버 포트가 즉시 열림
            init_thread = threading.Thread(target=initialize_server, daemon=True)
            init_thread.start()
            
            self._running = True
            
            # Waitress로 실행
            try:
                from waitress import serve
                logger.info(f"✅ 서버 시작: http://localhost:{self.port}")
                serve(
                    app,
                    host=self.host,
                    port=self.port,
                    threads=16,
                    _quiet=True
                )
            except ImportError:
                logger.warning("Waitress 없음 - Flask 개발 서버 사용")
                app.run(
                    host=self.host,
                    port=self.port,
                    debug=False,
                    threaded=True,
                    use_reloader=False
                )
        except Exception as e:
            import traceback
            self._error = str(e)
            self._running = False
            logger.error(f"❌ 서버 시작 실패: {e}\n{traceback.format_exc()}")
    
    def stop(self):
        self._stop_event.set()
        self._running = False


# ============================================================================
# 스타일
# ============================================================================
DARK_STYLE = """
QMainWindow, QWidget {
    background-color: #1a1a2e;
    color: #eaeaea;
}
QGroupBox {
    border: 1px solid #0f3460;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 10px;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 8px;
    color: #e94560;
}
QPushButton {
    background: #0f3460;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 10px 20px;
    font-weight: bold;
}
QPushButton:hover {
    background: #e94560;
}
QPushButton:disabled {
    background: #2a2a3e;
    color: #666;
}
QPushButton#dangerBtn {
    background: #dc2626;
}
QPushButton#dangerBtn:hover {
    background: #ef4444;
}
QPushButton#smallBtn {
    padding: 6px 12px;
    font-size: 11px;
}
QLabel {
    color: #eaeaea;
}
QLabel#statusLabel {
    color: #10b981;
    font-size: 14px;
    font-weight: bold;
}
QLabel#statusLabel[status="loading"] {
    color: #f59e0b;
}
QLabel#statusLabel[status="error"] {
    color: #ef4444;
}
QCheckBox {
    color: #eaeaea;
    spacing: 8px;
}
QCheckBox::indicator {
    width: 20px;
    height: 20px;
    border-radius: 4px;
    background: #0f3460;
}
QCheckBox::indicator:checked {
    background: #e94560;
}
QTextEdit {
    background: #0f3460;
    border: none;
    border-radius: 6px;
    padding: 8px;
    color: #a0a0b0;
    font-family: Consolas, monospace;
    font-size: 11px;
}
QLineEdit {
    background: #0f3460;
    border: 1px solid #2a2a5e;
    border-radius: 4px;
    padding: 8px;
    color: #eaeaea;
}
QLineEdit:focus {
    border-color: #e94560;
}
QDialog {
    background-color: #1a1a2e;
    color: #eaeaea;
}
"""


# ============================================================================
# 메인 윈도우
# ============================================================================
class ServerWindow(QMainWindow):
    def __init__(self, start_minimized: bool = False):
        super().__init__()
        self.server_thread: Optional[ServerThread] = None
        self.start_minimized = start_minimized
        self.log_buffer: list = []  # 로그 버퍼
        
        self._init_ui()
        self._init_tray()
        self._start_server()
        
        if start_minimized:
            self.hide()
        else:
            self.show()
    
    def _get_local_ip(self) -> str:
        """로컬 네트워크 IP 주소 가져오기"""
        import socket
        try:
            # 외부 연결 시도하여 사용 중인 IP 찾기
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"
    
    def _init_ui(self):
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.setMinimumSize(600, 700)
        self.resize(650, 750)
        
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # 헤더
        header = QHBoxLayout()
        logo = QLabel(f"📚 {APP_NAME}")
        logo.setFont(QFont("", 16, QFont.Weight.Bold))
        header.addWidget(logo)
        header.addStretch()
        
        self.status_label = QLabel("🔄 시작 중...")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setProperty("status", "loading")
        header.addWidget(self.status_label)
        layout.addLayout(header)
        
        # 서버 정보
        info_group = QGroupBox("서버 정보")
        info_layout = QVBoxLayout(info_group)
        
        # 로컬 IP 가져오기
        local_ip = self._get_local_ip()
        
        self.url_label = QLabel(f"🌐 로컬: http://localhost:{settings_manager.get_server_port()}")
        self.url_label.setFont(QFont("", 11))
        info_layout.addWidget(self.url_label)
        
        self.network_label = QLabel(f"🔗 네트워크: http://{local_ip}:{settings_manager.get_server_port()}")
        self.network_label.setFont(QFont("", 11))
        self.network_label.setStyleSheet("color: #3b82f6;")  # 파란색으로 강조
        info_layout.addWidget(self.network_label)
        
        self.admin_label = QLabel(f"⚙️ 관리자: http://localhost:{settings_manager.get_server_port()}/admin")
        info_layout.addWidget(self.admin_label)
        
        layout.addWidget(info_group)
        
        # 버튼
        btn_layout = QHBoxLayout()
        
        self.open_btn = QPushButton("🔍 검색 페이지 열기")
        self.open_btn.setToolTip("브라우저에서 검색 페이지를 엽니다")
        self.open_btn.clicked.connect(self._open_search)
        btn_layout.addWidget(self.open_btn)
        
        self.admin_btn = QPushButton("⚙️ 관리자 페이지")
        self.admin_btn.setToolTip("파일 업로드 및 시스템 설정 페이지를 엽니다")
        self.admin_btn.clicked.connect(self._open_admin)
        btn_layout.addWidget(self.admin_btn)
        
        layout.addLayout(btn_layout)
        
        # 시스템 메트릭
        metrics_group = QGroupBox("시스템 상태")
        metrics_layout = QHBoxLayout(metrics_group)
        
        # CPU 사용량
        cpu_frame = QFrame()
        cpu_layout = QVBoxLayout(cpu_frame)
        cpu_layout.setContentsMargins(10, 5, 10, 5)
        self.cpu_label = QLabel("🖥️ CPU")
        self.cpu_label.setFont(QFont("", 10))
        self.cpu_value = QLabel("0%")
        self.cpu_value.setFont(QFont("", 14, QFont.Weight.Bold))
        self.cpu_value.setStyleSheet("color: #3b82f6;")
        cpu_layout.addWidget(self.cpu_label, alignment=Qt.AlignmentFlag.AlignCenter)
        cpu_layout.addWidget(self.cpu_value, alignment=Qt.AlignmentFlag.AlignCenter)
        metrics_layout.addWidget(cpu_frame)
        
        # 메모리 사용량
        mem_frame = QFrame()
        mem_layout = QVBoxLayout(mem_frame)
        mem_layout.setContentsMargins(10, 5, 10, 5)
        self.mem_label = QLabel("💾 메모리")
        self.mem_label.setFont(QFont("", 10))
        self.mem_value = QLabel("0%")
        self.mem_value.setFont(QFont("", 14, QFont.Weight.Bold))
        self.mem_value.setStyleSheet("color: #f59e0b;")
        mem_layout.addWidget(self.mem_label, alignment=Qt.AlignmentFlag.AlignCenter)
        mem_layout.addWidget(self.mem_value, alignment=Qt.AlignmentFlag.AlignCenter)
        metrics_layout.addWidget(mem_frame)
        
        # 활성 검색 수
        search_frame = QFrame()
        search_layout = QVBoxLayout(search_frame)
        search_layout.setContentsMargins(10, 5, 10, 5)
        self.search_label = QLabel("🔍 활성 검색")
        self.search_label.setFont(QFont("", 10))
        self.search_value = QLabel("0")
        self.search_value.setFont(QFont("", 14, QFont.Weight.Bold))
        self.search_value.setStyleSheet("color: #10b981;")
        search_layout.addWidget(self.search_label, alignment=Qt.AlignmentFlag.AlignCenter)
        search_layout.addWidget(self.search_value, alignment=Qt.AlignmentFlag.AlignCenter)
        metrics_layout.addWidget(search_frame)
        
        layout.addWidget(metrics_group)
        
        # 설정
        settings_group = QGroupBox("설정")
        settings_layout = QVBoxLayout(settings_group)
        
        # 포트 설정 버튼 추가
        port_layout = QHBoxLayout()
        self.port_btn = QPushButton("🔌 포트 설정")
        self.port_btn.setObjectName("smallBtn")
        self.port_btn.setToolTip("서버가 사용할 포트 번호를 변경합니다 (변경 후 재시작 필요)")
        self.port_btn.clicked.connect(self._show_port_dialog)
        port_layout.addWidget(self.port_btn)
        port_layout.addStretch()
        settings_layout.addLayout(port_layout)
        
        self.autostart_check = QCheckBox("Windows 시작 시 자동 실행")
        self.autostart_check.setChecked(AutoStartManager.is_enabled())
        self.autostart_check.stateChanged.connect(self._toggle_autostart)
        settings_layout.addWidget(self.autostart_check)
        
        self.minimize_check = QCheckBox("닫기 버튼 클릭 시 트레이로 최소화")
        self.minimize_check.setChecked(True)
        settings_layout.addWidget(self.minimize_check)
        
        # 비밀번호 설정 버튼
        pw_layout = QHBoxLayout()
        self.pw_btn = QPushButton("🔑 관리자 비밀번호 설정")
        self.pw_btn.setObjectName("smallBtn")
        self.pw_btn.setToolTip("관리자 페이지 접근 시 필요한 비밀번호를 설정합니다")
        self.pw_btn.clicked.connect(self._show_password_dialog)
        pw_layout.addWidget(self.pw_btn)
        
        self.pw_status = QLabel("🔓 비보호" if not settings_manager.has_admin_password() else "🔒 보호됨")
        pw_layout.addWidget(self.pw_status)
        pw_layout.addStretch()
        settings_layout.addLayout(pw_layout)
        
        # ========== 오프라인 모드 설정 (폐쇄망 지원) ==========
        offline_separator = QFrame()
        offline_separator.setFrameShape(QFrame.Shape.HLine)
        offline_separator.setStyleSheet("background-color: #0f3460;")
        settings_layout.addWidget(offline_separator)
        
        offline_label = QLabel("🔌 오프라인 모드 (폐쇄망 지원)")
        offline_label.setStyleSheet("color: #e94560; font-weight: bold; margin-top: 5px;")
        settings_layout.addWidget(offline_label)
        
        # 오프라인 모드 체크박스
        self.offline_check = QCheckBox("오프라인 모드 활성화 (인터넷 없이 로컬 모델 사용)")
        self.offline_check.setChecked(settings_manager.get_offline_mode())
        self.offline_check.stateChanged.connect(self._toggle_offline_mode)
        settings_layout.addWidget(self.offline_check)
        
        # 로컬 모델 경로 설정
        model_path_layout = QHBoxLayout()
        model_path_label = QLabel("모델 경로:")
        model_path_label.setStyleSheet("color: #888; font-size: 11px;")
        model_path_layout.addWidget(model_path_label)
        
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("기본값: ./models (비워두면 기본 경로 사용)")
        self.model_path_edit.setText(settings_manager.get_local_model_path())
        self.model_path_edit.textChanged.connect(self._on_model_path_changed)
        model_path_layout.addWidget(self.model_path_edit)
        
        self.model_path_btn = QPushButton("📂")
        self.model_path_btn.setObjectName("smallBtn")
        self.model_path_btn.setFixedWidth(40)
        self.model_path_btn.clicked.connect(self._browse_model_path)
        model_path_layout.addWidget(self.model_path_btn)
        
        settings_layout.addLayout(model_path_layout)
        
        # 오프라인 모드 안내
        offline_hint = QLabel("※ 오프라인 모드 사용 전, 인터넷 환경에서 download_models.py를 실행하세요")
        offline_hint.setStyleSheet("color: #666; font-size: 10px;")
        settings_layout.addWidget(offline_hint)
        
        layout.addWidget(settings_group)
        
        # 로그 (확장된 영역)
        log_group = QGroupBox("서버 로그")
        log_layout = QVBoxLayout(log_group)
        
        # 로그 버튼들
        log_btn_layout = QHBoxLayout()
        
        self.save_log_btn = QPushButton("💾 저장")
        self.save_log_btn.setObjectName("smallBtn")
        self.save_log_btn.setToolTip("현재 로그를 텍스트 파일로 저장합니다")
        self.save_log_btn.clicked.connect(self._save_log)
        log_btn_layout.addWidget(self.save_log_btn)
        
        self.clear_log_btn = QPushButton("🗑️ 지우기")
        self.clear_log_btn.setObjectName("smallBtn")
        self.clear_log_btn.setToolTip("로그 창의 내용을 모두 지웁니다")
        self.clear_log_btn.clicked.connect(self._clear_log)
        log_btn_layout.addWidget(self.clear_log_btn)
        
        # 로그 필터 체크박스
        log_btn_layout.addSpacing(20)
        filter_label = QLabel("필터:")
        filter_label.setStyleSheet("color: #888; font-size: 11px;")
        log_btn_layout.addWidget(filter_label)
        
        self.log_filter_info = QCheckBox("INFO")
        self.log_filter_info.setChecked(True)
        self.log_filter_info.setStyleSheet("color: #a0a0b0;")
        log_btn_layout.addWidget(self.log_filter_info)
        
        self.log_filter_warning = QCheckBox("WARNING")
        self.log_filter_warning.setChecked(True)
        self.log_filter_warning.setStyleSheet("color: #f59e0b;")
        log_btn_layout.addWidget(self.log_filter_warning)
        
        self.log_filter_error = QCheckBox("ERROR")
        self.log_filter_error.setChecked(True)
        self.log_filter_error.setStyleSheet("color: #ef4444;")
        log_btn_layout.addWidget(self.log_filter_error)
        
        log_btn_layout.addStretch()
        log_layout.addLayout(log_btn_layout)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(150)
        log_layout.addWidget(self.log_text)
        
        layout.addWidget(log_group)
        
        # 서버 제어 버튼
        server_btn_layout = QHBoxLayout()
        
        self.restart_btn = QPushButton("🔄 서버 재시작")
        self.restart_btn.setToolTip("서버를 재시작합니다 (설정 변경 적용)")
        self.restart_btn.clicked.connect(self._restart_server)
        server_btn_layout.addWidget(self.restart_btn)
        
        self.quit_btn = QPushButton("🛑 프로그램 종료")
        self.quit_btn.setObjectName("dangerBtn")
        self.quit_btn.setToolTip("서버를 종료하고 프로그램을 닫습니다")
        self.quit_btn.clicked.connect(self._quit_app)
        server_btn_layout.addWidget(self.quit_btn)
        
        layout.addLayout(server_btn_layout)
        
        # 로그 시그널 연결
        log_signal.log_received.connect(self._append_log)
    
    def _init_tray(self):
        """시스템 트레이 초기화"""
        self.tray_icon = QSystemTrayIcon(self)
        
        # 기본 아이콘 설정 (앱 아이콘 또는 시스템 기본)
        app_icon = QApplication.instance().windowIcon()
        if not app_icon.isNull():
            self.tray_icon.setIcon(app_icon)
        else:
            # 기본 스타일 아이콘 사용
            from PyQt6.QtWidgets import QStyle
            default_icon = QApplication.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon)
            self.tray_icon.setIcon(default_icon)
        
        # 트레이 메뉴
        tray_menu = QMenu()
        
        show_action = QAction("창 열기", self)
        show_action.triggered.connect(self.show)
        tray_menu.addAction(show_action)
        
        tray_menu.addSeparator()
        
        search_action = QAction("검색 페이지", self)
        search_action.triggered.connect(self._open_search)
        tray_menu.addAction(search_action)
        
        admin_action = QAction("관리자 페이지", self)
        admin_action.triggered.connect(self._open_admin)
        tray_menu.addAction(admin_action)
        
        tray_menu.addSeparator()
        
        quit_action = QAction("종료", self)
        quit_action.triggered.connect(self._quit_app)
        tray_menu.addAction(quit_action)
        
        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.setToolTip(APP_NAME)
        self.tray_icon.activated.connect(self._tray_activated)
        self.tray_icon.show()
        
        # 상태 업데이트 타이머
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self._update_status)
        self.status_timer.start(2000)
    
    def _start_server(self):
        """서버 시작"""
        port = settings_manager.get_server_port()
        
        # 포트 체크
        if is_port_in_use(port):
            logger.error(f"❌ 포트 {port}가 이미 사용 중입니다. 다른 프로그램이 사용 중인지 확인하세요.")
            self.status_label.setText("❌ 포트 사용 중")
            self.status_label.setProperty("status", "error")
            return
        
        # AppConfig 업데이트 (서버 모듈의 설정도 변경 필요)
        AppConfig.SERVER_PORT = port
        
        self.server_thread = ServerThread(
            AppConfig.SERVER_HOST,
            port
        )
        self.server_thread.start()
        logger.info(f"🚀 서버 스레드 시작 (포트: {port})")
    
    def _show_port_dialog(self):
        """포트 설정 다이얼로그"""
        current_port = settings_manager.get_server_port()
        dialog = PortDialog(self, current_port)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_port = dialog.get_port()
            if new_port != current_port:
                settings_manager.set_server_port(new_port)
                
                # 라벨 업데이트
                local_ip = self._get_local_ip()
                self.url_label.setText(f"🌐 로컬: http://localhost:{new_port}")
                self.network_label.setText(f"🔗 네트워크: http://{local_ip}:{new_port}")
                self.admin_label.setText(f"⚙️ 관리자: http://localhost:{new_port}/admin")
                
                QMessageBox.information(
                    self, "알림", 
                    "포트가 변경되었습니다.\n서버를 재시작해야 적용됩니다."
                )
    
    def _toggle_offline_mode(self, state):
        """오프라인 모드 토글"""
        enabled = state == Qt.CheckState.Checked.value
        settings_manager.set_offline_mode(enabled)
        
        # AppConfig도 업데이트
        AppConfig.OFFLINE_MODE = enabled
        
        mode_str = "활성화" if enabled else "비활성화"
        logger.info(f"🔌 오프라인 모드 {mode_str}")
        
        if enabled:
            QMessageBox.information(
                self, "오프라인 모드",
                "오프라인 모드가 활성화되었습니다.\n\n"
                "• 인터넷 연결 없이 로컬 모델만 사용합니다.\n"
                "• 모델이 미리 다운로드되어 있어야 합니다.\n"
                "• 서버를 재시작해야 적용됩니다.\n\n"
                "모델 다운로드: python download_models.py"
            )
        else:
            QMessageBox.information(
                self, "오프라인 모드",
                "오프라인 모드가 비활성화되었습니다.\n"
                "서버를 재시작해야 적용됩니다."
            )
    
    def _on_model_path_changed(self, text):
        """모델 경로 변경 시"""
        settings_manager.set_local_model_path(text)
        AppConfig.LOCAL_MODEL_PATH = text
        if text:
            logger.info(f"📂 로컬 모델 경로 변경: {text}")
    
    def _browse_model_path(self):
        """모델 폴더 선택 다이얼로그"""
        current_path = self.model_path_edit.text() or os.path.join(
            os.path.dirname(sys.executable) if getattr(sys, 'frozen', False)
            else os.path.dirname(os.path.abspath(__file__)),
            'models'
        )
        
        folder = QFileDialog.getExistingDirectory(
            self,
            "모델 폴더 선택",
            current_path,
            QFileDialog.Option.ShowDirsOnly
        )
        
        if folder:
            self.model_path_edit.setText(folder)
            logger.info(f"📂 모델 폴더 선택: {folder}")
    
    def _restart_server(self):
        """서버 재시작"""
        logger.info("🔄 서버 재시작 중...")
        self.status_label.setText("🔄 재시작 중...")
        
        # 기존 서버 정리
        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.stop()
            # 스레드 종료 대기 (Waitress는 즉시 종료 불가)
            logger.warning("⚠️ 서버 스레드는 다음 요청 후 종료됩니다")
        
        # 새 서버 스레드 시작
        self._start_server()
    
    def _update_status(self):
        """상태 업데이트"""
        if qa_system.is_loading:
            self.status_label.setText(f"🔄 {qa_system.load_progress}")
            self.status_label.setProperty("status", "loading")
        elif qa_system.is_ready:
            stats = qa_system.get_stats()
            self.status_label.setText(
                f"✅ 준비 완료 | 📄 {stats['files']}개 파일 | 📊 {stats['chunks']} 청크"
            )
            self.status_label.setProperty("status", "ready")
        elif qa_system.load_error:
            # 오류 상태 표시
            error_msg = qa_system.load_error
            if len(error_msg) > 30:
                error_msg = error_msg[:30] + "..."
            self.status_label.setText(f"❌ 오류: {error_msg}")
            self.status_label.setProperty("status", "error")
        else:
            self.status_label.setText("⏳ 대기 중...")
            self.status_label.setProperty("status", "loading")
        
        # 시스템 메트릭 업데이트
        try:
            import psutil
            cpu = psutil.cpu_percent(interval=0)
            mem = psutil.virtual_memory().percent
            self.cpu_value.setText(f"{cpu:.0f}%")
            self.mem_value.setText(f"{mem:.0f}%")
            
            # CPU 색상 변경 (높으면 빨강)
            if cpu > 80:
                self.cpu_value.setStyleSheet("color: #ef4444;")
            elif cpu > 50:
                self.cpu_value.setStyleSheet("color: #f59e0b;")
            else:
                self.cpu_value.setStyleSheet("color: #3b82f6;")
            
            # 메모리 색상 변경
            if mem > 85:
                self.mem_value.setStyleSheet("color: #ef4444;")
            elif mem > 70:
                self.mem_value.setStyleSheet("color: #f59e0b;")
            else:
                self.mem_value.setStyleSheet("color: #10b981;")
        except ImportError:
            pass
        
        # 활성 검색 수 업데이트
        try:
            from server import search_queue
            queue_stats = search_queue.get_stats()
            self.search_value.setText(str(queue_stats.get('active', 0)))
        except (ImportError, AttributeError):
            pass
        
        # 스타일 새로고침 (안전하게)
        style = self.status_label.style()
        if style:
            style.unpolish(self.status_label)
            style.polish(self.status_label)
    
    def _append_log(self, message: str, level: str = "INFO"):
        """로그 추가 (레벨별 색상, 필터 적용)"""
        # 버퍼에 저장 (필터와 관계없이)
        self.log_buffer.append((message, level))
        if len(self.log_buffer) > 1000:
            self.log_buffer = self.log_buffer[-500:]
        
        # 필터 체크
        should_show = False
        if level == "INFO" and hasattr(self, 'log_filter_info') and self.log_filter_info.isChecked():
            should_show = True
        elif level == "WARNING" and hasattr(self, 'log_filter_warning') and self.log_filter_warning.isChecked():
            should_show = True
        elif level == "ERROR" and hasattr(self, 'log_filter_error') and self.log_filter_error.isChecked():
            should_show = True
        elif level == "DEBUG":
            should_show = True  # DEBUG는 항상 표시 (또는 별도 필터 추가 가능)
        elif not hasattr(self, 'log_filter_info'):
            should_show = True  # 필터 UI가 아직 없으면 모두 표시
        
        if not should_show:
            return
        
        # 색상 설정
        color_map = {
            "ERROR": "#ef4444",
            "WARNING": "#f59e0b", 
            "INFO": "#a0a0b0",
            "DEBUG": "#666666"
        }
        color = color_map.get(level, "#a0a0b0")
        
        # HTML 형식으로 추가
        html = f'<span style="color: {color}">{message}</span>'
        self.log_text.append(html)
        
        # 스크롤 맨 아래로
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def _save_log(self):
        """로그 파일로 저장"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "로그 저장",
            f"server_log_{datetime.now():%Y%m%d_%H%M%S}.txt",
            "Text Files (*.txt);;All Files (*)"
        )
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    # log_buffer는 이제 (message, level) 튜플 리스트
                    for item in self.log_buffer:
                        if isinstance(item, tuple):
                            f.write(f"{item[0]}\n")
                        else:
                            f.write(f"{item}\n")
                self.tray_icon.showMessage(
                    APP_NAME, "로그가 저장되었습니다",
                    QSystemTrayIcon.MessageIcon.Information, 2000
                )
            except IOError as e:
                QMessageBox.warning(self, "오류", f"저장 실패: {e}")
    
    def _clear_log(self):
        """로그 지우기"""
        self.log_text.clear()
        self.log_buffer.clear()
    
    def _show_password_dialog(self):
        """비밀번호 설정 다이얼로그"""
        is_change = settings_manager.has_admin_password()
        dialog = PasswordDialog(self, is_change)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            password = dialog.get_password()
            settings_manager.set_admin_password(password)
            
            if password:
                self.pw_status.setText("🔒 보호됨")
                self.tray_icon.showMessage(
                    APP_NAME, "관리자 비밀번호가 설정되었습니다",
                    QSystemTrayIcon.MessageIcon.Information, 2000
                )
            else:
                self.pw_status.setText("🔓 비보호")
                self.tray_icon.showMessage(
                    APP_NAME, "비밀번호 보호가 해제되었습니다",
                    QSystemTrayIcon.MessageIcon.Information, 2000
                )
    
    def _open_search(self):
        webbrowser.open(f"http://localhost:{settings_manager.get_server_port()}")
    
    def _open_admin(self):
        webbrowser.open(f"http://localhost:{settings_manager.get_server_port()}/admin")
    
    def _toggle_autostart(self, state):
        if state == Qt.CheckState.Checked.value:
            if AutoStartManager.enable():
                self.tray_icon.showMessage(
                    APP_NAME,
                    "Windows 시작 시 자동 실행됩니다",
                    QSystemTrayIcon.MessageIcon.Information,
                    2000
                )
        else:
            AutoStartManager.disable()
    
    def _tray_activated(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self.show()
            self.raise_()
            self.activateWindow()
    
    def _quit_app(self):
        reply = QMessageBox.question(
            self,
            "서버 종료",
            "서버를 종료하시겠습니까?\n모든 연결이 끊어집니다.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            # 타이머 정리
            if hasattr(self, 'status_timer') and self.status_timer:
                self.status_timer.stop()
            self.tray_icon.hide()
            qa_system.cleanup()
            QApplication.quit()
    
    def closeEvent(self, event: QCloseEvent):
        if self.minimize_check.isChecked():
            event.ignore()
            self.hide()
            self.tray_icon.showMessage(
                APP_NAME,
                "서버가 백그라운드에서 실행 중입니다",
                QSystemTrayIcon.MessageIcon.Information,
                2000
            )
        else:
            self._quit_app()


# ============================================================================
# 메인
# ============================================================================
# ============================================================================
# 스플래시 스크린
# ============================================================================
class SplashScreen(QSplashScreen):
    """시작 시 표시되는 스플래시 화면"""
    
    def __init__(self):
        # 스플래시 이미지 생성 (코드로)
        pixmap = QPixmap(400, 250)
        pixmap.fill(QColor('#1a1a2e'))
        
        super().__init__(pixmap)
        self.setWindowFlags(Qt.WindowType.SplashScreen | Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        
        # 레이아웃
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # 제목
        self.title = QLabel("📚 사내 규정 검색기")
        self.title.setStyleSheet("color: #e94560; font-size: 24px; font-weight: bold;")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # 상태 메시지
        self.status = QLabel("초기화 중...")
        self.status.setStyleSheet("color: #eaeaea; font-size: 14px;")
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # 프로그레스바
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # 무한 로딩
        self.progress.setTextVisible(False)
        self.progress.setFixedWidth(300)
        self.progress.setStyleSheet("""
            QProgressBar {
                border: none;
                background: #0f3460;
                border-radius: 5px;
                height: 8px;
            }
            QProgressBar::chunk {
                background: #e94560;
                border-radius: 5px;
            }
        """)
        
        # 버전
        self.version = QLabel("v2.4")
        self.version.setStyleSheet("color: #666; font-size: 11px;")
        self.version.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # 중앙 위젯
        central = QWidget(self)
        central.setGeometry(0, 0, 400, 250)
        central_layout = QVBoxLayout(central)
        central_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        central_layout.addStretch()
        central_layout.addWidget(self.title)
        central_layout.addSpacing(20)
        central_layout.addWidget(self.status)
        central_layout.addSpacing(15)
        central_layout.addWidget(self.progress, alignment=Qt.AlignmentFlag.AlignCenter)
        central_layout.addSpacing(20)
        central_layout.addWidget(self.version)
        central_layout.addStretch()
    
    def set_status(self, message: str):
        self.status.setText(message)
        QApplication.processEvents()


# ============================================================================
# 백그라운드 로더 스레드
# ============================================================================
class ModuleLoaderThread(QThread):
    """무거운 모듈을 백그라운드에서 로드하는 스레드"""
    progress = pyqtSignal(str)  # 상태 메시지
    finished_loading = pyqtSignal(bool)  # 로드 완료
    
    def run(self):
        try:
            self.progress.emit("Flask 서버 모듈 로드 중...")
            _load_heavy_modules()
            
            # 모델 자동 다운로드 및 로드
            self.progress.emit("AI 모델 확인 중...")
            
            # 설정 파일에서 오프라인 모드 확인
            from app.utils import get_app_directory
            import json
            settings_path = os.path.join(get_app_directory(), 'config', 'settings.json')
            
            offline_mode = False
            if os.path.exists(settings_path):
                try:
                    with open(settings_path, 'r', encoding='utf-8') as f:
                        settings = json.load(f)
                    offline_mode = settings.get('offline_mode', False)
                except Exception:
                    pass
            
            if not offline_mode:
                # 온라인 모드: 모델 자동 다운로드
                self.progress.emit("AI 모델 다운로드/로드 중...")
                self.progress.emit("(최초 실행 시 500MB+ 다운로드)")
            else:
                self.progress.emit("로컬 모델 로드 중...")
            
            # 모델 로드 실행
            result = qa_system.load_model(AppConfig.DEFAULT_MODEL)
            
            if result.success:
                self.progress.emit("서버 시작 중...")
            else:
                self.progress.emit(f"모델 로드 실패: {result.message}")
            
            self.progress.emit("완료!")
            self.finished_loading.emit(True)
        except Exception as e:
            self.progress.emit(f"오류: {e}")
            self.finished_loading.emit(False)


# ============================================================================
# 메인
# ============================================================================
def main():
    # 명령행 인자 확인
    start_minimized = '--minimized' in sys.argv or '-m' in sys.argv
    
    # Qt 앱 먼저 생성 (빠름)
    qt_app = QApplication(sys.argv)
    qt_app.setStyle('Fusion')
    qt_app.setStyleSheet(DARK_STYLE)
    qt_app.setQuitOnLastWindowClosed(False)
    
    # 스플래시 즉시 표시
    splash = SplashScreen()
    splash.show()
    qt_app.processEvents()
    
    # 백그라운드에서 무거운 모듈 로드
    loader = ModuleLoaderThread()
    
    def on_progress(msg):
        splash.set_status(msg)
    
    def on_finished(success):
        splash.close()
        if success:
            # 메인 윈도우 생성 및 표시
            window = ServerWindow(start_minimized=start_minimized)
            # window를 전역으로 유지 (가비지 컬렉션 방지)
            qt_app._main_window = window
        else:
            QMessageBox.critical(None, "오류", "모듈 로드에 실패했습니다.")
            qt_app.quit()
    
    loader.progress.connect(on_progress)
    loader.finished_loading.connect(on_finished)
    loader.start()
    
    sys.exit(qt_app.exec())


if __name__ == '__main__':
    # PyInstaller 멀티프로세싱 지원 (필수)
    import multiprocessing
    multiprocessing.freeze_support()
    
    main()

