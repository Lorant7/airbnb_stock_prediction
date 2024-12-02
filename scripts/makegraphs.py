import sys
from pathlib import Path

# Add the project root to sys.path
project_root = Path(__file__).resolve().parents[1]  # Adjust the level as needed
sys.path.append(str(project_root))

from src.visualization import graphs

import pandas as pd
df = pd.read_csv('../data/processed/train_df.csv')

# Date vs. Volume
graphs.lineGraph(df, 'Volume', save=True)

# Date vs. Open
graphs.lineGraph(df, 'Open', save=True)

# Date vs. Close
graphs.lineGraph(df, 'Close', save=True)

# Date vs. High
graphs.lineGraph(df, 'High', save=True)

# Date vs. Low
graphs.lineGraph(df, 'Low', save=True)

# Date vs. Price Change
graphs.lineGraph(df, 'price_change', save=True)