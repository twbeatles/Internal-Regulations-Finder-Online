# -*- coding: utf-8 -*-
"""
사내 규정 검색기 - 서버 GUI (PyQt6)
시스템 트레이 + Windows 시작 프로그램 등록 지원
"""

import sys
import os
import threading
import webbrowser
import winreg
import ctypes
from typing import Optional

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSystemTrayIcon, QMenu, QMessageBox,
    QCheckBox, QGroupBox, QTextEdit, QFrame
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QIcon, QAction, QFont, QColor, QPalette, QCloseEvent

# 서버 모듈 import
from server import (
    app, qa_system, initialize_server, AppConfig, logger, UPLOAD_DIR,
    FileUtils, graceful_shutdown
)

# ============================================================================
# 상수
# ============================================================================
APP_NAME = "사내 규정 검색기 서버"
APP_VERSION = "1.0"
REGISTRY_KEY = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Run"
REGISTRY_VALUE_NAME = "RegulationSearchServer"


# ============================================================================
# 로그 시그널 핸들러
# ============================================================================
class LogSignal(QObject):
    log_received = pyqtSignal(str)


log_signal = LogSignal()


class QtLogHandler:
    """Qt 시그널로 로그 전송"""
    def write(self, message):
        if message.strip():
            log_signal.log_received.emit(message.strip())
    
    def flush(self):
        pass


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
# 서버 스레드
# ============================================================================
class ServerThread(threading.Thread):
    def __init__(self, host: str, port: int):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.server = None
        self._stop_event = threading.Event()
    
    def run(self):
        # 서버 초기화
        initialize_server()
        
        # Waitress로 실행
        try:
            from waitress import serve
            logger.info(f"🚀 서버 시작: http://localhost:{self.port}")
            serve(
                app,
                host=self.host,
                port=self.port,
                threads=8,
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
    
    def stop(self):
        self._stop_event.set()


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
"""


# ============================================================================
# 메인 윈도우
# ============================================================================
class ServerWindow(QMainWindow):
    def __init__(self, start_minimized: bool = False):
        super().__init__()
        self.server_thread: Optional[ServerThread] = None
        self.start_minimized = start_minimized
        
        self._init_ui()
        self._init_tray()
        self._start_server()
        
        if start_minimized:
            self.hide()
        else:
            self.show()
    
    def _init_ui(self):
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.setMinimumSize(500, 400)
        self.resize(550, 450)
        
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
        
        self.url_label = QLabel(f"🌐 URL: http://localhost:{AppConfig.SERVER_PORT}")
        self.url_label.setFont(QFont("", 12))
        info_layout.addWidget(self.url_label)
        
        self.admin_label = QLabel(f"⚙️ 관리자: http://localhost:{AppConfig.SERVER_PORT}/admin")
        info_layout.addWidget(self.admin_label)
        
        layout.addWidget(info_group)
        
        # 버튼
        btn_layout = QHBoxLayout()
        
        self.open_btn = QPushButton("🔍 검색 페이지 열기")
        self.open_btn.clicked.connect(self._open_search)
        btn_layout.addWidget(self.open_btn)
        
        self.admin_btn = QPushButton("⚙️ 관리자 페이지")
        self.admin_btn.clicked.connect(self._open_admin)
        btn_layout.addWidget(self.admin_btn)
        
        layout.addLayout(btn_layout)
        
        # 설정
        settings_group = QGroupBox("설정")
        settings_layout = QVBoxLayout(settings_group)
        
        self.autostart_check = QCheckBox("Windows 시작 시 자동 실행")
        self.autostart_check.setChecked(AutoStartManager.is_enabled())
        self.autostart_check.stateChanged.connect(self._toggle_autostart)
        settings_layout.addWidget(self.autostart_check)
        
        self.minimize_check = QCheckBox("닫기 버튼 클릭 시 트레이로 최소화")
        self.minimize_check.setChecked(True)
        settings_layout.addWidget(self.minimize_check)
        
        layout.addWidget(settings_group)
        
        # 로그
        log_group = QGroupBox("서버 로그")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(120)
        log_layout.addWidget(self.log_text)
        
        layout.addWidget(log_group)
        
        # 종료 버튼
        self.quit_btn = QPushButton("🛑 서버 종료")
        self.quit_btn.setObjectName("dangerBtn")
        self.quit_btn.clicked.connect(self._quit_app)
        layout.addWidget(self.quit_btn)
        
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
        self.server_thread = ServerThread(
            AppConfig.SERVER_HOST,
            AppConfig.SERVER_PORT
        )
        self.server_thread.start()
    
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
        else:
            self.status_label.setText("⏳ 대기 중...")
            self.status_label.setProperty("status", "loading")
        
        # 스타일 새로고침 (안전하게)
        style = self.status_label.style()
        if style:
            style.unpolish(self.status_label)
            style.polish(self.status_label)
    
    def _append_log(self, message: str):
        """로그 추가"""
        self.log_text.append(message)
        # 스크롤 맨 아래로
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def _open_search(self):
        webbrowser.open(f"http://localhost:{AppConfig.SERVER_PORT}")
    
    def _open_admin(self):
        webbrowser.open(f"http://localhost:{AppConfig.SERVER_PORT}/admin")
    
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
def main():
    # 명령행 인자 확인
    start_minimized = '--minimized' in sys.argv or '-m' in sys.argv
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setStyleSheet(DARK_STYLE)
    app.setQuitOnLastWindowClosed(False)  # 트레이로 최소화 지원
    
    window = ServerWindow(start_minimized=start_minimized)
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
