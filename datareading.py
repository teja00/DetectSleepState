import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from random import sample

train_events = pd.read_csv('child-mind-institute-detect-sleep-states/train_events.csv')
train_series = pd.read_parquet('child-mind-institute-detect-sleep-states/train_series.parquet')
test_series = pd.read_parquet('child-mind-institute-detect-sleep-states/test_series.parquet')

print(train_events.isnull().sum())


train_events['timestamp'] = pd.to_datetime(train_events['timestamp'], errors='coerce', utc=True)


onset_df = train_events[train_events['event'] == 'onset']
wakeup_df = train_events[train_events['event'] == 'wakeup']


plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.histplot(onset_df['step'], bins=30, color='blue', kde=True)
plt.title('Distribution of "onset" events (step)')
plt.subplot(1, 2, 2)
sns.histplot(wakeup_df['step'], bins=30, color='red', kde=True)
plt.title('Distribution of "wakeup" events (step)')
plt.tight_layout()
plt.show()


plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.histplot(onset_df['timestamp'].dt.hour, bins=24, color='blue', kde=True)
plt.title('Distribution of "onset" event times')
plt.subplot(1, 2, 2)
sns.histplot(wakeup_df['timestamp'].dt.hour, bins=24, color='red', kde=True)
plt.title('Distribution of "wakeup" event times')
plt.tight_layout()
plt.show()

# Group the data by "night" and "event" and count the number of occurrences of each event type for each night
night_event_distribution = train_events.groupby(['night', 'event']).size().unstack(fill_value=0).reset_index()

# Plot the distribution of events across different nights
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.histplot(night_event_distribution['onset'], bins=30, color='blue', kde=True)
plt.title('Distribution of "onset" events across nights')

plt.subplot(1, 2, 2)
sns.histplot(night_event_distribution['wakeup'], bins=30, color='red', kde=True)
plt.title('Distribution of "wakeup" events across nights')

plt.tight_layout()
plt.show()


# Pivot the table to have "onset" and "wakeup" times in separate columns for each series and night
pivot_table = train_events.pivot_table(index=['series_id', 'night'], columns='event', values='timestamp', aggfunc='first').reset_index()

# Calculate the sleep duration as the difference between "wakeup" and "onset" times
pivot_table['sleep_duration'] = (pivot_table['wakeup'] - pivot_table['onset']).dt.total_seconds() / 3600  # Convert seconds to hours

# Display the first few rows of the new dataframe
pivot_table.head()



# Plot the distribution of sleep durations
plt.figure(figsize=(8, 6))
sns.histplot(pivot_table['sleep_duration'], bins=30, color='green', kde=True)
plt.title('Distribution of Sleep Durations')
plt.xlabel('Sleep Duration (hours)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# Select 3 random series_ids
random_series_ids = np.random.choice(train_series['series_id'].unique(), 3)

# Initialize the figure
plt.figure(figsize=(12, 10))

for i, series_id in enumerate(random_series_ids):
    # Get the data for the current series_id
    series_data = train_series[train_series['series_id'] == series_id]
    
    # Create subplots for anglez distribution
    plt.subplot(3, 2, i*2+1)
    sns.histplot(series_data['anglez'], bins=30, kde=True)
    plt.title(f'Distribution of anglez values - {series_id}')
    plt.xlabel('anglez')
    plt.ylabel('Frequency')
    
    # Create subplots for enmo distribution
    plt.subplot(3, 2, i*2+2)
    sns.histplot(series_data['enmo'], bins=30, kde=True, color='orange')
    plt.title(f'Distribution of enmo values - {series_id}')
    plt.xlabel('enmo')
    plt.ylabel('Frequency')

# Display the plots
plt.tight_layout()
plt.show()



# Step 1: Data Preparation

unique_series_ids = train_series['series_id'].unique()
selected_series_ids = np.random.choice(unique_series_ids, 3, replace=False)

selected_series_data = train_series[train_series['series_id'].isin(selected_series_ids)]

train_events['timestamp'] = pd.to_datetime(train_events['timestamp'])
selected_series_data['timestamp'] = pd.to_datetime(selected_series_data['timestamp'])

dataframes = []

for series_id in selected_series_ids:
    
    series_data = selected_series_data[selected_series_data['series_id'] == series_id]
    series_events = train_events[train_events['series_id'] == series_id]
    
    for i in range(len(series_events) - 1):
        
        start_timestamp = series_events.iloc[i]['timestamp']
        end_timestamp = series_events.iloc[i + 1]['timestamp']
        
        mask = (series_data['timestamp'] >= start_timestamp) & (series_data['timestamp'] < end_timestamp)
        series_data.loc[mask, 'event'] = series_events.iloc[i]['event']
    
    dataframes.append(series_data)

labeled_data = pd.concat(dataframes)

labeled_data.head()



import matplotlib.pyplot as plt




fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))


for i, series_id in enumerate(selected_series_ids):
    
    series_data = labeled_data[labeled_data['series_id'] == series_id]
    
    series_data[series_data['event'] == 'onset'].plot(x='timestamp', y='anglez', ax=axes[i, 0], style='.', color='red', label='Onset')
    series_data[series_data['event'] == 'wakeup'].plot(x='timestamp', y='anglez', ax=axes[i, 0], style='.', color='blue', label='Awake')
    
    series_data[series_data['event'] == 'onset'].plot(x='timestamp', y='enmo', ax=axes[i, 1], style='.', color='red', label='Onset')
    series_data[series_data['event'] == 'wakeup'].plot(x='timestamp', y='enmo', ax=axes[i, 1], style='.', color='blue', label='Awake')
    
    axes[i, 0].set_title(f'Series ID: {series_id} - anglez')
    axes[i, 0].set_xlabel('Timestamp')
    axes[i, 0].set_ylabel('anglez')
    axes[i, 0].legend()
    
    axes[i, 1].set_title(f'Series ID: {series_id} - enmo')
    axes[i, 1].set_xlabel('Timestamp')
    axes[i, 1].set_ylabel('enmo')
    axes[i, 1].legend()
plt.tight_layout()
plt.show()