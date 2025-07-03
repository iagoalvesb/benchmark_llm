import logging
import sys

def init_logger():
    """Simple logger initialization"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True
    )
    return logging.getLogger()