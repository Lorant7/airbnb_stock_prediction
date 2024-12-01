import torch
from .base import StockDataModule
from ..config import BATCH_SIZE


model_file_name = '0h88ygww-graceful-haze-33.pth'
path_to_model = '../../model/LSTM/artifacts/' + model_file_name
tmodel = torch.load(path_to_model)

dm = StockDataModule(batch_size=BATCH_SIZE)
test_loader = dm.test_dataloader()

invest_prob = 0.8
def investment_thresholds(prob):
    if prob > invest_prob:
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

def test():
    with torch.no_grad():
        totals = {
            "good_invest" : 0, # True positive
            "bad_invest" : 0,  # False positive
            "bad_sell" : 0,    # False negative
            "good_sell" : 0    # True negative
            }
        

        for batch in test_loader:
            y = batch[1]
            x = batch[0]
            yhat =  [investment_thresholds(prediction) for prediction in torch.sigmoid(tmodel(x))]
            results = test_investments(y.tolist(), yhat)
            totals['good_invest'] = totals['good_invest'] + results[0]
            totals['bad_invest'] = totals['bad_invest'] + results[1]
            totals['bad_sell'] = totals['bad_sell'] + results[2]
            totals['good_sell'] = totals['good_sell'] + results[3]
        
        print(totals)