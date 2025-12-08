"""
Utility Functions for Crop Yield Prediction
Author: Pushkarjay Ajay
"""

import sys
import os
from datetime import datetime
from contextlib import contextmanager

class Logger:
    """Dual output logger - prints to console and saves to file"""
    
    def __init__(self, log_file=None, mode='w'):
        self.terminal = sys.stdout
        self.log_file = log_file
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            self.log = open(log_file, mode, encoding='utf-8')
        else:
            self.log = None
    
    def write(self, message):
        self.terminal.write(message)
        if self.log:
            self.log.write(message)
            self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        if self.log:
            self.log.flush()
    
    def close(self):
        if self.log:
            self.log.close()


@contextmanager
def log_output(log_file):
    """Context manager to log all output to a file"""
    logger = Logger(log_file)
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = logger
    sys.stderr = logger
    try:
        yield logger
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        logger.close()


def get_timestamp():
    """Get current timestamp for logging"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_log_filename(script_name):
    """Generate log filename with timestamp"""
    from .config import LOGS_DIR
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(LOGS_DIR, f"{script_name}_{timestamp}.log")


def print_header(title, width=70):
    """Print a formatted section header"""
    print("\n" + "=" * width)
    print(f"🌾 {title}")
    print("=" * width)


def print_section(title, width=70):
    """Print a formatted subsection header"""
    print("\n" + "-" * width)
    print(f"📋 {title}")
    print("-" * width)


def print_success(message):
    """Print a success message"""
    print(f"✅ {message}")


def print_warning(message):
    """Print a warning message"""
    print(f"⚠️ {message}")


def print_error(message):
    """Print an error message"""
    print(f"❌ {message}")


def print_info(message):
    """Print an info message"""
    print(f"ℹ️ {message}")
