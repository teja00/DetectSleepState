import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import random
from copy import deepcopy
from functools import partial
from itertools import combinations
from itertools import groupby

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from sklearn.model_selection import cross_validate
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import precision_score
from sklearn.metrics import average_precision_score



import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from catboost import Pool


pd.set_option('display.max_columns', None)

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


train = pd.read_parquet("child-mind-institute-detect-sleep-states/Zzzs_train.parquet")
test  = pd.read_parquet("child-mind-institute-detect-sleep-states/test_series.parquet")

# parse the timestamp and create an "hour" feature
train["timestamp"] = pd.to_datetime(train["timestamp"],utc=True)
train["hour"] = train["timestamp"].dt.hour

test["timestamp"] = pd.to_datetime(test["timestamp"],utc=True)
test["hour"] = test["timestamp"].dt.hour

train.head()

def create_features(df):
    # parse the timestamp and create an "hour" feature
    df["timestamp"] = pd.to_datetime(df["timestamp"],utc=True)
    df["hour"] = (df["timestamp"].dt.hour).astype('int8')
    df['minute'] = df['timestamp'].dt.minute

    # Calculate the half-hour periods
    df['half_hour'] = (df['hour'] * 2 + (df['minute'] // 30)).astype('int8')
    
    df.drop(columns=['minute'], inplace=True)
    
    # feature cross
    df["anglez_times_enmo"] = abs(df["anglez"]) * df["enmo"].astype('float16')
    # "rolling" features
    periods = 50
    df["anglez_diff"] = df.groupby('series_id')['anglez'].diff(periods=periods).fillna(method="bfill").astype('float16')
    df["enmo_diff"]   = df.groupby('series_id')['enmo'].diff(periods=periods).fillna(method="bfill").astype('float16')
    df["anglez_rolling"] = df["anglez"].rolling(periods,center=True).mean().fillna(method="bfill").fillna(method="ffill").astype('float16')
    df["enmo_rolling"]   = df["enmo"].rolling(periods,center=True).mean().fillna(method="bfill").fillna(method="ffill").astype('float16')
    df["anglez_diff_rolling"] = df["anglez_diff"].rolling(periods,center=True).mean().fillna(method="bfill").fillna(method="ffill").astype('float16')
    df["enmo_diff_rolling"]   = df["enmo_diff"].rolling(periods,center=True).mean().fillna(method="bfill").fillna(method="ffill").astype('float16')
    
    return df

features = ["hour","anglez_times_enmo", "half_hour",
           "anglez","anglez_diff","anglez_rolling","anglez_diff_rolling",
           "enmo","enmo_diff","enmo_rolling","enmo_diff_rolling"]
train = create_features(train)
test = create_features(test)

X_train = train[features]
y_train = train["awake"].astype('int8')
X_test = test[features]

del train

class Classifier:
    def __init__(self, n_estimators=100, device="cpu", random_state=42):
        self.n_estimators = n_estimators
        self.device = device
        self.random_state = random_state
        self.models = self._define_model()
        self.models_name = list(self._define_model().keys())
        self.len_models = len(self.models)
        
    def _define_model(self):
        
        xgb_1 = {
            'n_estimators': self.n_estimators,
            'eval_metric': 'map',
            'verbosity': 0,
            'random_state': self.random_state,
            'scale_pos_weight': 2/3
        }
        
       
        models = {
            'xgb_1': xgb.XGBClassifier(**xgb_1),
            'rf': RandomForestClassifier(max_depth=4, min_samples_leaf=100, n_estimators=50, random_state=self.random_state),
            #'lr': LogisticRegression(max_iter=150, random_state=self.random_state, n_jobs=-1),
        }
        
        return models
    

random_state = 42
random_state_list =[42]
n_estimators = 90
device = 'cpu'
early_stopping_rounds = 50
verbose = False