import pandas as pd
import numpy as np

# Load data
data = pd.read_csv('revised_traffic_dataset.csv')

data.dropna(subset=['Crash Severity'], inplace=True)

# Drop rows where 'Crash Severity' is 'Unknown'
data = data[data['Crash Severity'] != 'Unknown']


# Process 'Crash Hour' column to extract hour
def extract_hour(crash_hour_range):
    try:
        start_time = crash_hour_range.split(' to ')[0]
        hour = int(pd.to_datetime(start_time, format='%I:%M%p').hour)
        return hour
    except:
        return np.nan

data['Crash Hour Numeric'] = data['Crash Hour'].apply(extract_hour)

# Fill missing 'Crash Hour Numeric' with median
data['Crash Hour Numeric'] = data['Crash Hour Numeric'].fillna(data['Crash Hour Numeric'].median())

# Convert 'Number of Vehicles' to numeric
data['Number of Vehicles'] = pd.to_numeric(data['Number of Vehicles'], errors='coerce')
data['Number of Vehicles'] = data['Number of Vehicles'].fillna(data['Number of Vehicles'].median())

# Convert 'Speed Limit' to numeric
data['Speed Limit'] = pd.to_numeric(data['Speed Limit'], errors='coerce')
data['Speed Limit'] = data['Speed Limit'].fillna(data['Speed Limit'].median())
# Simplify 'Weather Conditions'
def simplify_weather(condition):
    if pd.isnull(condition) or condition.strip() == '':
        return 'Unknown'
    condition = condition.lower()
    if 'clear' in condition:
        return 'Clear'
    elif 'cloudy' in condition:
        return 'Cloudy'
    elif 'rain' in condition:
        return 'Rain'
    elif 'snow' in condition:
        return 'Snow'
    elif 'sleet' in condition or 'hail' in condition or 'freezing' in condition:
        return 'Sleet/Hail'
    elif 'fog' in condition or 'smog' in condition or 'smoke' in condition:
        return 'Fog/Smog/Smoke'
    elif 'wind' in condition or 'crosswind' in condition:
        return 'Severe Winds'
    elif 'blowing' in condition:
        return 'Blowing Sand/Snow'
    elif 'unknown' in condition or 'not reported' in condition:
        return 'Unknown'
    elif 'other' in condition or 'invalid' in condition:
        return 'Other'
    else:
        return 'Other'

data['Weather Conditions'] = data['Weather Conditions'].apply(simplify_weather)
def simplify_light_conditions(condition):
    if pd.isnull(condition) or condition.strip() == '':
        return 'Unknown'
    condition = condition.lower()
    if 'daylight' in condition:
        return 'Daylight'
    elif 'dark' in condition:
        return 'Darkness'
    elif 'dawn' in condition or 'dusk' in condition:
        return 'Twilight'
    elif 'unknown' in condition or 'not reported' in condition or 'other' in condition:
        return 'Unknown'
    else:
        return 'Unknown'

data['Light Conditions'] = data['Light Conditions'].apply(simplify_light_conditions)

# Simplify 'Road Surface Condition'
def simplify_road_surface(condition):
    if pd.isnull(condition) or condition.strip() == '':
        return 'Unknown'
    condition = condition.lower()
    if 'dry' in condition:
        return 'Dry'
    elif 'wet' in condition or 'water' in condition:
        return 'Wet'
    elif 'snow' in condition:
        return 'Snow'
    elif 'ice' in condition:
        return 'Ice'
    elif 'slush' in condition:
        return 'Slush'
    elif 'sand' in condition or 'mud' in condition or 'dirt' in condition or 'oil' in condition or 'gravel' in condition:
        return 'Sand/Mud/Dirt/Oil/Gravel'
    elif 'unknown' in condition or 'not reported' in condition:
        return 'Unknown'
    elif 'other' in condition or 'invalid' in condition:
        return 'Other'
    else:
        return 'Other'

data['Road Surface Condition'] = data['Road Surface Condition'].apply(simplify_road_surface)

data.to_csv('preprocessed_dataset.csv')