
'''
from Model import *
import os.path

def MTLF(model):
    if os.path.isfile('model.h5'):
        model = load_model('model.h5')
    model.fit(x_train, y_train, epochs=1)
    model.save('model.h5')
    print("trainig finish")
'''

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import sklearn
from Model import *
from sklearn.preprocessing import LabelEncoder

# ======= 讀取CSV檔案 =======
file_path = 'data.csv'  # 請替換為實際檔案路徑
df = pd.read_csv(file_path, header=None, names=['access_type', 'imsi', 'id', 'mcc', 'mnc', 'tac', 'timestamp'])

# 保留需要的三個欄位
selected_columns = ['imsi', 'tac', 'timestamp']
df = df[selected_columns]

# 將時間戳記轉換成Python的datetime對象
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 使用LabelEncoder將非數字的特徵轉換成數字
label_encoder = LabelEncoder()
df['imsi'] = label_encoder.fit_transform(df['imsi'])
df['tac'] = label_encoder.fit_transform(df['tac'])

# 查看編碼對應
encoding_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Encoding Mapping:")
print(encoding_mapping)

# 將資料按ID和時間戳記排序
df = df.sort_values(by=['imsi', 'timestamp'])

# ======= functions ========
# 定義 create_sequences 函數，只選擇特定的 imsi 進行預測
def create_sequences(data, imsi, seq_length):
    data_imsi = data[data['imsi'] == imsi]
    
    # 檢查是否有足夠的資料建立序列
    if len(data_imsi) < seq_length + 1:
        print(f'length of data_imsi = {len(data_imsi)}')
        print(f'Not enough data for IMSI: {imsi}')
        return None, None
    
    sequences = []
    targets = []
    for i in range(len(data_imsi) - seq_length):
        seq = data_imsi.iloc[i:i+seq_length, [0, 1]].values  
        target = data_imsi.iloc[i+seq_length, 1]  

        # 檢查是否有 NaN 或 NaT 值，如果有，跳過這筆資料
        if pd.isna(seq).any() or pd.isna(target):
            continue

        sequences.append(seq)
        targets.append(target)
    return torch.tensor(sequences, dtype=torch.float32), torch.tensor(targets, dtype=torch.long)


# 將資料轉換成模型可用的格式
seq_length = 1  # 時間序列的長度，可以根據實際情況調整
sequences, targets = create_sequences(df, imsi = 0, seq_length = seq_length)  # 這裡的 imsi=0 代表選擇第一個 imsi 進行預測

# 分割資料集為訓練集和測試集
X_train, y_train = sequences, targets
X_test, y_test = None, None  # 留給後面手動提供測試資料

# 印出總共有多少筆training data、testing data
print(f'Total Training Data: {len(X_train)}')
'''
# define RNN model
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

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
'''

# 定義模型參數
input_size = 2  # 每個時間點的特徵數，這裡是兩個欄位
hidden_size = 10  # RNN的隱藏層大小
output_size = len(label_encoder.classes_)  # 預測的目標數，這裡是 TAC 的不同類別數

# 創建模型、損失函數和優化器
model = LSTMModel(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()  # 交叉熵損失用於分類問題
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 訓練模型
def MTLF():
    num_epochs = 1000
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # 將 X_train 調整為正確的形狀
        input_seq = X_train.reshape(-1, seq_length, input_size)

        output = model(input_seq)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

    # save model 
    #torch.save(model, "model.pt")
    torch.save(model.state_dict(), "model.pt")
