import torch
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]  # Adjust the level as needed
sys.path.append(str(project_root))

from src.models.base import StockDataModule
from src.config import BATCH_SIZE
from src.visualization import graphs
from src.models.base import LSTM


make_confusion_matrix = True
model_file_name = 'w36wilcc-olive-paper-9.pth'

INVEST_PROB = 0.8
def investment_thresholds(prob):
    if prob > INVEST_PROB:
        return 1
    return 0

def test_investments(y, yhat):
    # True pos., False pos., False neg., True neg
    results = [0,0,0,0]
    for i in range(len(yhat)):
        # True positive
        if y[i] == 1 and yhat[i]  == 1:
            results[0] = results[0] + 1
        # False positive
        elif y[i] == 0 and yhat[i] == 1:
            results[1] = results[1] + 1
        # False negative
        elif y[i] == 1 and yhat[i] == 0:
            results[2] = results[2] + 1
        # True negative
        else:
            results[3] = results[3] + 1

    return results

# what I want to measure is 
# how many times the algorithm invests when it should
# how many times it invests when it should'test
# how many times it doesn't invest when it should

def test(model_file_name, graph = False):
    
    path_to_model = '../../model/LSTM/artifacts/' + model_file_name
    tmodel = torch.load(path_to_model)

    dm = StockDataModule(batch_size=BATCH_SIZE)
    dm.setup('test')
    test_loader = dm.test_dataloader()

    with torch.no_grad():
        totals = {
            "good_invest" : 0, # True positive
            "bad_invest" : 0,  # False positive
            "bad_sell" : 0,    # False negative
            "good_sell" : 0    # True negative
            }
        
        all_y = []
        all_yhat = []

        for batch in test_loader:
            y = batch[1]
            all_y.extend(y.int().tolist())

            x = batch[0]
            yhat =  [investment_thresholds(prediction) for prediction in torch.sigmoid(tmodel(x))]
            all_yhat.extend(yhat)

            results = test_investments(y.tolist(), yhat)
            totals['good_invest'] = totals['good_invest'] + results[0]
            totals['bad_invest'] = totals['bad_invest'] + results[1]
            totals['bad_sell'] = totals['bad_sell'] + results[2]
            totals['good_sell'] = totals['good_sell'] + results[3]
        
        graphs.confusionMatrix(all_y, all_yhat, save = graph)

test(model_file_name, graph = True)