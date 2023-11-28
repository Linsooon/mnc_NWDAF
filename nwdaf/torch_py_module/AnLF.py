'''
from Model import *
import os.path

def AnLF(model,data_num):
    if os.path.isfile('model.h5'):
        model = load_model('model.h5')
    prediction = model.predict(x_test[data_num:data_num+1],batch_size = 1)
    print(np.argmax(prediction))
    return np.argmax(prediction)
'''

import torch
import os.path
import torch.nn as nn
import json

# should modify sequence to recently three.
manual_test_data = torch.tensor([[[1.0, 0.0], [1.0, 1.0], [1.0, 1.0]]], dtype=torch.float32)

# define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

#anlf_res = json.dumps([{'data' : None}])
def AnLF():
    load_model = LSTMModel(2, 10, 2)
    ## output size 要看到時候 data.csv 有幾個，以目前我的環境只有 4 個
    if os.path.isfile('model.pt'):
        #load_model = torch.load('model.pt')
        load_model.load_state_dict(torch.load("model.pt"))
    # 預測整個序列
    with torch.no_grad():
        predicted_tac_sequence = torch.argmax(load_model(manual_test_data), dim=1).numpy()
    #prediction = model.predict(x_test[data_num:data_num+1],batch_size = 1)
    # 將預測的 TAC 解碼回原始的字串值
    #predicted_tac_strings = label_encoder.inverse_transform(predicted_tac_sequence)
    #print(np.argmax(prediction))
    print(f'手動提供的資料，預測的TAC序列為: {predicted_tac_sequence}')
    #print(f'解碼後的TAC序列: {predicted_tac_strings}')
    #anlf_res['data'] = predicted_tac_sequence
    #return json.dumps([{'data' : str(predicted_tac_sequence)}])
    return predicted_tac_sequence
AnLF()
