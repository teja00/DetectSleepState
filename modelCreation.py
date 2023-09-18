import numpy as np
import pandas as pd
from itertools import groupby
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from copy import deepcopy
from modelClass import Classifier


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

random_state = 42
random_state_list =[42]
n_estimators = 90
device = 'cpu'
early_stopping_rounds = 50
verbose = False
optuna_lgb = False


X_train_, X_val, y_train_, y_val = train_test_split(X_train, y_train, test_size=0.2)

# Initialize an array for storing test predictions
classifier = Classifier(n_estimators=n_estimators, device=device, random_state=random_state)
test_predss = np.zeros((X_test.shape[0]))
oof_predss = np.zeros((X_train.shape[0]))

del X_train

models_name = [_ for _ in classifier.models_name if ('xgb' in _) or ('lgb' in _) or ('cat' in _) or ('rf' in _) or ('lr' in _)]
score_dict = dict(zip(classifier.models_name, [[] for _ in range(len(classifier.models_name))]))

models = classifier.models

# Store oof and test predictions for each base model
oof_preds = []
test_preds = []

# Loop over each base model and fit it
for name, model in models.items():
    if name in ['xgb']:
        model.fit(X_train_, y_train_, eval_set=[(X_val, y_val)], early_stopping_rounds=early_stopping_rounds, verbose=verbose)
    else:
        model.fit(X_train_, y_train_)

    test_pred = model.predict_proba(X_test)[:, 1]
    y_val_pred = model.predict_proba(X_val)[:, 1]

    score = average_precision_score(y_val, y_val_pred)
    score_dict[name].append(score)
        
    print(f'{name} [SEED-{random_state}] Precision score: {score:.5f}')
        
    oof_preds.append(y_val_pred)
    test_preds.append(test_pred)
    
test_predss = np.average(np.array(test_preds), axis=0)
oof_predss[X_val.index] = np.average(np.array(oof_preds), axis=0)
    
del X_train_, X_val, y_val, y_train_

print(test_predss)
# Add a "not_awake" column as the complement of the "score" column:
test['score'] = test_predss
test["not_awake"] = 1 - test["score"]

# Smoothing of the predictions:
smoothing_length = 400  # Define the length for smoothing
test["smooth"] = test["not_awake"].rolling(smoothing_length, center=True).mean().fillna(method="bfill").fillna(method="ffill")

# Re-binarize the "smooth" column:
test["smooth"] = test["smooth"].round()

# https://stackoverflow.com/questions/73777727/how-to-mark-start-end-of-a-series-of-non-null-and-non-0-values-in-a-column-of-a
def get_event(df):
    lstCV = zip(df.series_id, df.smooth)
    lstPOI = []
    for (c, v), g in groupby(lstCV, lambda cv: 
                            (cv[0], cv[1]!=0 and not pd.isnull(cv[1]))):
        llg = sum(1 for item in g)
        if v is False: 
            lstPOI.extend([0]*llg)
        else: 
            lstPOI.extend(['onset']+(llg-2)*[0]+['wakeup'] if llg > 1 else [0])
    return lstPOI

test["event"] = get_event(test)

print(test['event'])

sample_submission = test.loc[test["event"] != 0]
sample_submission = sample_submission[["series_id", "step", "event", "score"]].copy()
sample_submission = sample_submission.reset_index(drop=True).reset_index(names="row_id")

# Save the sample submission DataFrame to a CSV file:
sample_submission.to_csv('submission.csv', index=False)