import yfinance as yf
from datetime import datetime
import os

from ..config import RAW_DATA_DIR

def loadData(load_new = False):
    if len(os.listdir(RAW_DATA_DIR)) == 0 or load_new:
        data = yf.Ticker("ABNB").history(period="5y")
        data.to_csv("../../data/raw/" + datetime.today().strftime('%y_%m_%d') + "-ABNB_5y_history.csv")