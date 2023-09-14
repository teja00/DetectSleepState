import pandas as pd
import numpy as np

train_events = pd.read_csv("child-mind-institute-detect-sleep-states/train_events.csv")

# finding the nan's 
series_has_NaN = train_events.groupby('series_id')['step'].apply(lambda x: x.isnull().any())
print(series_has_NaN.value_counts())
no_NaN_series = series_has_NaN[~series_has_NaN].index.tolist()
print(no_NaN_series)

# also drop these two truncated events series (EDA):
no_NaN_series.remove('31011ade7c0a') # incomplete events data
no_NaN_series.remove('a596ad0b82aa') # incomplete events data

def get_train_series(series):
    train_series = pd.read_parquet("child-mind-institute-detect-sleep-states/train_series.parquet", filters=[('series_id','=',series)])
    train_events = pd.read_csv("child-mind-institute-detect-sleep-states/train_events.csv").query('series_id == @series')
    
    # cleaning etc.
    train_events = train_events.dropna()
    train_events["step"]  = train_events["step"].astype("int")
    train_events["awake"] = train_events["event"].replace({"onset":1,"wakeup":0})

    train = pd.merge(train_series, train_events[['step','awake']], on='step', how='left')
    train["awake"] = train["awake"].bfill(axis ='rows')

    # final section:
    # train_events.groupby('series_id').tail(1)["event"].unique()
    # Result: the last event is always a "wakeup"
    train['awake'] = train['awake'].fillna(1) # awake
    train["awake"] = train["awake"].astype("int")
    return(train)

clean_train_data = []

for series_id in no_NaN_series:
    train = get_train_series(series_id)
    clean_train_data.append(train)


Zzzs_train = pd.concat(clean_train_data).reset_index(drop=True)
Zzzs_train["series_id"].nunique()

Zzzs_train.to_parquet('child-mind-institute-detect-sleep-states/Zzzs_train.parquet')
