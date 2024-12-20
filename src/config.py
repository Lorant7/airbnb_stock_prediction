from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = '../../data/raw/'
PROCESSED_DATA_DIR = '../../data/processed'
ITERIM_DATA_DIR = '../../data/interim'
BATCH_SIZE = 16