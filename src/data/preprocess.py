import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def feature_selection(df):
    
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day

    df['month_sin'] = np.sin(2*np.pi*df['month']/12)
    df['month_cos'] = np.cos(2*np.pi*df['month']/12)

    df['day_sin'] = np.sin(2*np.pi*df['day']/7)
    df['day_cos'] = np.cos(2*np.pi*df['day']/7)

    df['price_change'] = (df['Close'].shift(-1) - df['Close']) / df['Close']
    df['price_change'] = df['price_change'].fillna(0)

    df['change'] = df['price_change'].apply(lambda x: 1 if x > 0 else 0)

    df = df.drop(columns = ['price_change', 'Date', 'month', 'day', 'year', 'Dividends', 'Stock Splits'])

    # Normalizing Data
    scalar = StandardScaler()

    for column in df.columns:
        if column != 'change':
            df[column] = scalar.fit_transform(df[[column]])

    # No need to pass the edited data frame since the df is passed by reference, not by value
