import yfinance as yf
from datetime import datetime
import os

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]  # Adjust the level as needed
sys.path.append(str(project_root))

from src.config import RAW_DATA_DIR

def loadData(load_new = False):
    if len(os.listdir(RAW_DATA_DIR)) == 0 or load_new:
        data = yf.Ticker("ABNB").history(period="5y")
        data.to_csv("../../data/raw/" + datetime.today().strftime('%y_%m_%d') + "-ABNB_5y_history.csv")