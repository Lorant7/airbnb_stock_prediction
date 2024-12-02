import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def lineGraph(df, yaxis, save = False):
    plt.plot(df['Date'], df[yaxis])
    plt.xlabel('Date')
    plt.ylabel(yaxis)
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=60))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)

    if save:
        plt.savefig('../data/graphs/' + yaxis + '_vs_Date.png')

    plt.show()

def confusionMatrix(y, yhat, save = False):
    # Generate confusion matrix
    cm = confusion_matrix(y, yhat)

    # Display as a heatmap
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")

    if save:
        plt.savefig('../../data/graphs/confusion_matrix.png')

    plt.show()