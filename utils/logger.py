import os
import threading
import logging
from datetime import datetime
from pathlib import Path

# 預設結果目錄
model_name = os.environ.get("MODEL_NAME", "default")
results_dir = Path(f"./logger/{model_name}/")
# 建立所有日誌目錄 (若不存在則自動建立)
for level in ['info', 'detail']:
    os.makedirs(results_dir / level, exist_ok=True)

class Logger:
    """
    自訂 Logger，支援以下特性：
      - 寫入不同層級的日誌檔案（info, detail, warning, error, critical）
      - 可設定全域前綴，用於區分不同模組或實驗
      - 提供是否在 console 印出日誌的開關
      - 使用 thread lock 保證多線程下寫入安全
    """
    # 對應日誌層級與 Python logging 模組的層級
    _log_levels = {
        'info': logging.INFO,
        'detail': logging.DEBUG,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }
    # 各層級對應的檔案名稱
    _log_files = {
        'info': 'info.log',
        'detail': 'detail.log',
        'warning': 'warning.log',
        'error': 'error.log',
        'critical': 'critical.log'
    }
    # 全域前綴 (可用於標識模組、實驗等)
    _current_prefix = None
    # 是否印出到 console
    _console_output = True
    # thread lock 保證寫檔安全
    _lock = threading.Lock()

    @staticmethod
    def set_prefix(prefix: str):
        """設定全域的前綴"""
        Logger._current_prefix = prefix

    @staticmethod
    def reset_prefix():
        """重置前綴為 None"""
        Logger._current_prefix = None

    @staticmethod
    def set_console_output(enabled: bool):
        """設定是否在 console 印出訊息"""
        Logger._console_output = enabled

    @staticmethod
    def _log(level: str, timestamp: str, message: str):
        """內部寫入日誌檔案的實作"""
        log_message = f"[{timestamp}] [{level.upper()}] {message}"
        if Logger._console_output:
            print(log_message)

        level_lower = level.lower()
        # 依層級取得檔案名稱與目錄
        if level_lower == 'info':
            log_file = Logger._log_files[level_lower]
            # 若有全域前綴則加在檔名前面
            if Logger._current_prefix:
                log_file = f"{Logger._current_prefix}_{log_file}"
            log_path = results_dir / level_lower / log_file
            # 使用 thread lock 確保寫入安全
            with Logger._lock:
                try:
                    with open(log_path, 'a', encoding='utf-8') as f:
                        f.write(log_message + '\n')
                except Exception as e:
                    print(f"Error writing to log file '{log_path}': {e}")

        log_file = Logger._log_files['detail']
        if Logger._current_prefix:
            log_file = f"{Logger._current_prefix}_{log_file}"

        log_path = results_dir / 'detail' / log_file
        # 使用 thread lock 確保寫入安全
        with Logger._lock:
            try:
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(log_message + '\n')
            except Exception as e:
                print(f"Error writing to log file '{log_path}': {e}")

    @staticmethod
    def __call__(level: str, message: str):
        """記錄日誌訊息"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        Logger._log(level, timestamp, message)

    @staticmethod
    def info(message: str):
        """記錄 info 層級的日誌，並同時寫入 detail"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        Logger._log('info', timestamp, message)

    @staticmethod
    def detail(message: str):
        """記錄 detail 層級的日誌"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        Logger._log('detail', timestamp, message)

    @staticmethod
    def warning(message: str):
        """記錄 warning 層級的日誌"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        Logger._log('warning', timestamp, message)

    @staticmethod
    def error(message: str):
        """記錄 error 層級的日誌"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        Logger._log('error', timestamp, message)
        raise message

    @staticmethod
    def critical(message: str):
        """記錄 critical 層級的日誌"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        Logger._log('critical', timestamp, message)
