# -*- coding: utf-8 -*-
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc, confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import csv
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_tabnet.tab_model import TabNetClassifier
from tabtransformertf import FTTransformerConfig, FTTransformerModel


from sklearn.tree import plot_tree
import warnings
warnings.filterwarnings("ignore")

#測試堆疊法
from sklearn.ensemble import StackingClassifier, VotingClassifier

import argparse

'''
字體
'''

# 指定字體

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt

font_dirs = [r'..\..\font']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

for font_file in font_files:
    font_manager.fontManager.addfont(font_file)
plt.rcParams['font.family'] = "Noto Sans Mono CJK TC"

import os

font_path = r'..\..\font\NotoSansCJK-Regular.ttc'
if os.path.exists(font_path):
    print("字體文件存在於系統中。")
else:
    print("字體文件不存在於系統中。")

import matplotlib.pyplot as plt

# 清除 Matplotlib 緩存
plt.rcParams.update({'font.family': 'Noto Sans Mono CJK TC'})
plt.close('all')


'''
讀取資料
'''

data_all_url = 'data_all_May8_big5.csv'
SK_data_May8 = pd.read_csv(data_all_url,encoding='big5')

'''
編碼
'''

from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()
SK_data_May8[['PTA']] = label_binarizer.fit_transform(SK_data_May8[['PTA']])

cat_cols = ['Gender','部位','廔管種類']
SK_data_May8 = pd.get_dummies(SK_data_May8, columns = cat_cols, drop_first = True)

'''
移除不需要的資料
'''

df_tree = SK_data_May8.drop(columns=['病歷號碼','姓名','日期'])

#其他手術資料
df_tree = df_tree.drop(columns=['PermCath R', 'PermCath I', 'Gortex on',
       'Gortex non', 'Thrombectomy open', 'Thrombectomy cath', 'PTA S',
       'PTA C','surgery', 'previous_surgery'])

#修改名字
df_tree.rename(columns={'condition 2': '本次檢測之A值相較於前一筆下降25%並低於1000'}, inplace=True)
df_tree.rename(columns={'Recommended_Surgery': 'KDOQI guidelines 檢測建議執行手術'}, inplace=True)

df_tree = df_tree.drop(columns=['condition 1'])

'''
填補缺失值
'''

df_tree['術後至檢測的天數差'] = df_tree['術後至檢測的天數差'].replace([np.nan, -np.nan], 0)

'''
訓練+驗證
'''
def split_data(X, y, n_splits=3):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    return kf.split(X, y)
'''
深度學習模型
'''
class MLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim, hidden_dim=128, output_dim=2, epochs=50, lr=0.001):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.epochs = epochs
        self.lr = lr
        self.model = self.build_model()
        self.scaler = StandardScaler()
    
    def build_model(self):
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.Softmax(dim=1)
        )
    
    def fit(self, X, y):
        X = self.scaler.fit_transform(X)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.long)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
    
    def predict_proba(self, X):
        X = self.scaler.transform(X)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            outputs = self.model(X_tensor).numpy()
        return outputs

class TabNetWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.model = TabNetClassifier()
    
    def fit(self, X, y):
        self.model.fit(X.values, y.values, max_epochs=50, patience=10)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X.values)

# 設定 Tabular Transformer
class FTTransformerWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim, hidden_dim=128, output_dim=2):
        self.input_dim = input_dim
        config = FTTransformerConfig(input_dim=self.input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        self.model = FTTransformerModel(config)
        self.scaler = StandardScaler()
    
    def fit(self, X, y):
        X = self.scaler.fit_transform(X)
    
    def predict_proba(self, X):
        X = self.scaler.transform(X)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            outputs = self.model(X_tensor).numpy()
        return outputs

'''迴圈起'''
uncertain_mul = [0.9, 0.95, 1.0]
for mul in tqdm(uncertain_mul, desc=f"uncertain_threshold :"):
    
    df_tree_avf = df_tree[df_tree['廔管種類_自體管']==True]
    df_tree_avg = df_tree[df_tree['廔管種類_自體管']==False]

    X_avf = df_tree_avf.drop(columns = ['PTA'])
    y_avf = df_tree_avf['PTA']
    X_avg = df_tree_avg.drop(columns = ['PTA'])
    y_avg = df_tree_avg['PTA']

    est_times = 10
    """Hy =−[0.5log(0.5)+0.5log(0.5)]=−[2×0.5log(0.5)]=log(2)≈0.693
     Hy 的范围是 [0, log⁡(2)]
    """
    uncertain_threshold = mul*math.log(2)
    fold = 0
    for train_index, test_index in split_data(X_avf, y_avf):
        fold += 1
        # 調整閾值
    #     threshold_output_list = [0.33, 0.28, 0.3]

        #noise
        noise_levels = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    #     noise_column = '本次檢測之A值'
    #     noise_results = []

        X_train, X_test_ori = X_avf.iloc[train_index], X_avf.iloc[test_index]
        y_train, y_test = y_avf.iloc[train_index], y_avf.iloc[test_index]

        print()
        print(len(X_train),len(X_test_ori))
        print()

        # 計算正例和負例的數量
        num_positive_samples = (y_train == 1).sum()#+(y_test['surgery'] == 1).sum()
        num_negative_samples = (y_train == 0).sum()#+(y_test['surgery'] == 0).sum()

        # 計算 scale_pos_weight
        data_scale_pos_weight = num_negative_samples / num_positive_samples

        for noise_level in tqdm(noise_levels, desc=f"Fold {fold}:Processing Noise Levels"):
            file_name = f'./AVF_detail/avf_result_ut_{mul}_level_{noise_level}_fold_{fold}.txt'
            accuracy_list = []
            f1_list = []
            ppv_avf_list = []
            npv_avf_list = []
            thres_list = []
    #         print(f'*********\nAVF :{threshold_output_index+1}/3\n{noise_level}\n*********')
            model = VotingClassifier(estimators=[
                ('mlp', MLPClassifier(input_dim=X_train.shape[1])),
                ('tabnet', TabNetWrapper()),
                ('fttransformer', FTTransformerWrapper(input_dim=X_train.shape[1]))
            ], voting='soft', weights=[1, 1, 1])#
            model.fit(X_train, y_train)

            prob_head = []
            y_pred_with_uncer = []
            hy_list = []

            with open(f'./AVF_detail/avf_result_ut_{mul}_level_{noise_level}_fold_{fold}.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['prob', 'mean_var'])
                
                for index, row in X_test_ori.iterrows():
                    for i in range(est_times):
                        eta = np.random.normal(0, noise_level)# noise
        #                 print(eta)
                        row[1] += eta# A值
                        y_test_proba = model.predict_proba([row]).tolist()[0]#[:, 1] [False, True]
                        prob_head.append(y_test_proba)
                    prob_mean = [np.mean(np.array(prob_head)[:, 0]),np.mean(np.array(prob_head)[:, 1])]#np.mean(np.array(prob_head)[:, 0])
                    prob_variance = [np.var(np.array(prob_head)[:, 0]), np.var(np.array(prob_head)[:, 1])]# np.var(np.array(prob_head)[:, 0])
                    
                    writer.writerow([prob_head, [prob_mean,prob_variance]])
                    prob_head = []
    
                    #Indeterminate-Aware data classification
                    #輸出y 'True' 'False' 'Uncertain'
                    Hy = -(prob_mean[0]*math.log(prob_mean[0]) + prob_mean[1]*math.log(prob_mean[1]))
        #             print(Hy)
                    hy_list.append(Hy)
                    if Hy > uncertain_threshold:
                        y_pred_with_uncer.append('Uncertain')
                    else:
                        if ((prob_mean[1] > prob_mean[0]) and ((prob_mean[1]-math.sqrt(prob_variance[1])) > (prob_mean[0]+math.sqrt(prob_variance[0])))):
                            y_pred_with_uncer.append('True')
                        elif ((prob_mean[1] < prob_mean[0]) and ((prob_mean[0]-math.sqrt(prob_variance[0])) > (prob_mean[1]+math.sqrt(prob_variance[1])))):
                            y_pred_with_uncer.append('False')
                        else:
                            y_pred_with_uncer.append('Uncertain2')

            uncertain_indices = [i for i, x in enumerate(y_pred_with_uncer) if x in ["Uncertain", "Uncertain2"]]

            # 移除不确定值
            y_pred_filtered = [x for i, x in enumerate(y_pred_with_uncer) if i not in uncertain_indices]
            y_test_filtered = [x for i, x in enumerate(y_test) if i not in uncertain_indices]

            # uncertain y_test
            y_test_uncertain = [x for i, x in enumerate(y_test) if i in uncertain_indices]

            # 转换 True/False 为 1/0
            y_pred_filtered = [1 if x == "True" else 0 for x in y_pred_filtered]

            # 计算 TP, FP, TN, FN
            TP = sum(1 for p, t in zip(y_pred_filtered, y_test_filtered) if p == 1 and t == 1)
            FP = sum(1 for p, t in zip(y_pred_filtered, y_test_filtered) if p == 1 and t == 0)
            TN = sum(1 for p, t in zip(y_pred_filtered, y_test_filtered) if p == 0 and t == 0)
            FN = sum(1 for p, t in zip(y_pred_filtered, y_test_filtered) if p == 0 and t == 1)

            # 计算 IP, IN
            IP = sum(1 for i in y_test_uncertain if i == 1)
            IN = sum(1 for i in y_test_uncertain if i == 0)

            # 输出结果
    #       print(f"\nNoise level:{noise_level}\n移除的索引: {uncertain_indices}")
    #         print(f'總共 {len(uncertain_indices)}個')
    #         print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
    #         print(f"IP: {IP}, IN: {IN}")
            if (len(uncertain_indices) < X_test_ori.shape[0]):
                conf_result = f'TP:{TP} TN:{TN} FP:{FP} FN:{FN} IP:{IP} IN:{IN}\n\n'
                if (TP==0 or TN==0 or FP==0 or FN==0):
#                     print(conf_result)
                    metrics_info = "####################\nNo metrics info\n####################"
                else:
                    print(conf_result)
                    accuracy = (TP+TN) / (TP+TN+FP+FN)
                    precision = TP/(TP+FP)
                    recall = TP/(TP+FN)
                    f1 = TP/(TP+0.5*(FP+FN))
                    ppv = TP/(TP+FP)
                    npv = TN/(TN+FN)
                    all_data = TP+TN+FP+FN+IP+IN
                    error = (FN+FP)/all_data
                    leakage = FN/all_data
                    overkill = FP/all_data
                    indeterminate = (IP+IN)/all_data
                    imperfection = error + indeterminate
                    iap = TP/(TP+FP+IN)
                    iar = (TP+IP)/(TP+FN+IP)
                    iaf1 = (2*iap*iar)/(iap+iar)
                    harmonic = ((1/np.exp(overkill))+(1/np.exp(leakage))+(1/np.exp(indeterminate)))/3
                # Using triple quotes for multi-line string
                    metrics_info = f"""
                    accuracy: {accuracy}
                    precision: {precision}
                    recall: {recall}
                    f1: {f1}
                    ppv: {ppv}
                    npv: {npv}
                    error: {error}
                    leakage: {leakage}
                    overkill: {overkill}
                    indeterminate: {indeterminate}
                    imperfection: {imperfection}
                    indeterminate-adjusted precision: {iap}
                    indeterminate-adjusted Recall: {iar}
                    indeterminate-adjusted f1: {iaf1}
                    harmonic score: {harmonic}
                    """
#                     print(metrics_info)

    #             	print(f'y_pred_with_uncer\n{y_pred_with_uncer}')
#                     print(f'hy_list\n{hy_list}')
            else:
                conf_result = "####################\nAll sample uncertain\n####################\n"
                metrics_info = "####################\nNo metrics info\n####################"
#                 print(conf_result)
                
#             write_data = f'fold: {fold} noise_level: {noise_level}\n\n'
            with open(file_name, 'w') as file:
                file.write(conf_result + metrics_info)

'''迴圈末'''
