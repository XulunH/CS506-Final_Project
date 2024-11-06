import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read the dataset
df = pd.read_csv('../datasets/preprocessed_dataset.csv')

def age_midpoint(age_range):
    if pd.isnull(age_range):
        return np.nan
    age_range = age_range.strip().replace('+', '').replace('>', '').replace('<', '').lower()
    if age_range == 'unknown':
        return np.nan
    parts = age_range.split('-')
    try:
        if len(parts) == 1:
            return float(parts[0])
        elif len(parts) == 2:
            return (float(parts[0]) + float(parts[1])) / 2
        else:
            return np.nan
    except ValueError:
        return np.nan

# Apply the age_midpoint function to the relevant columns
df['Youngest Driver Age'] = df['Age of Driver - Youngest Known'].apply(age_midpoint)
df['Oldest Driver Age'] = df['Age of Driver - Oldest Known'].apply(age_midpoint)

# Convert 'Crash Hour Numeric' and 'Speed Limit' to numeric, coerce errors to NaN
df['Crash Hour Numeric'] = pd.to_numeric(df['Crash Hour Numeric'], errors='coerce')
df['Speed Limit'] = pd.to_numeric(df['Speed Limit'], errors='coerce')

# Trim whitespace from categorical columns to ensure consistency
categorical_columns = [
    'City Town Name', 'Crash Severity', 'Driver Contributing Circumstances (All Drivers)',
    'Driver Distracted By (All Vehicles)', 'First Harmful Event', 'Light Conditions',
    'Manner of Collision', 'Road Surface Condition', 'Roadway Junction Type',
    'Traffic Control Device Type', 'Trafficway Description', 'Vehicle Actions Prior to Crash (All Vehicles)',
    'Vehicle Configuration (All Vehicles)', 'Vehicle Emergency Use (All Vehicles)',
    'Weather Conditions', 'First Harmful Event Location'
]

for col in categorical_columns:
    df[col] = df[col].astype(str).str.strip()

sns.set_theme(style="whitegrid")

# Plot 2: Crash Severity Distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=df, y='Crash Severity', color='lightcoral', order=df['Crash Severity'].value_counts().index)
plt.title('Distribution of Crash Severity')
plt.xlabel('Number of Crashes')
plt.ylabel('Crash Severity')
plt.tight_layout()
plt.savefig('../result_pics/crash_severity_distribution.png')
plt.show()
plt.close()

# Plot 4: Speed Limit vs. Crash Severity
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Crash Severity', y='Speed Limit', palette='Set2')
plt.title('Speed Limit by Crash Severity')
plt.xlabel('Crash Severity')
plt.ylabel('Speed Limit (mph)')
plt.tight_layout()
plt.savefig('../result_pics/speed_limit_by_crash_severity.png')
plt.show()

plt.close()

# Plot 8: Driver Age Distribution
young_age_df = df[['Youngest Driver Age', 'Crash Severity']].rename(columns={'Youngest Driver Age': 'Driver Age'})
old_age_df = df[['Oldest Driver Age', 'Crash Severity']].rename(columns={'Oldest Driver Age': 'Driver Age'})
combined_df = pd.concat([young_age_df, old_age_df]).dropna(subset=['Driver Age', 'Crash Severity'])
plt.figure(figsize=(10, 6))
sns.histplot(data=combined_df, x='Driver Age', hue='Crash Severity', bins=range(0, 100, 5), stat="density", alpha=0.6, multiple="dodge")
plt.title('Driver Age Distribution by Crash Severity')
plt.xlabel('Age')
plt.ylabel('Density')
plt.tight_layout()
plt.savefig('../result_pics/driver_age_distribution_by_crash_severity.png')
plt.show()
plt.close()

# Group by 'Weather Conditions' and 'Crash Severity' to get counts
severity_counts = df.groupby(['Weather Conditions', 'Crash Severity']).size().reset_index(name='Count')
# Calculate the proportion of each severity within each weather condition
severity_counts['Proportion'] = severity_counts.groupby('Weather Conditions')['Count'].transform(lambda x: x / x.sum())

# Exclude 'unknown' and 'other' categories from 'Weather Conditions'
filtered_severity_counts = severity_counts[~severity_counts['Weather Conditions'].str.lower().isin(['unknown', 'other'])]

# Plot the proportions without 'unknown' and 'other' weather conditions
plt.figure(figsize=(12, 8))
sns.barplot(x='Weather Conditions', y='Proportion', hue='Crash Severity', data=filtered_severity_counts)
plt.title('Proportion of Crash Severity by Weather Conditions (Excluding Unknown and Other)')
plt.ylabel('Proportion')
plt.xlabel('Weather Conditions')
plt.xticks(rotation=45)
plt.legend(title='Crash Severity')
plt.tight_layout()
plt.savefig('../result_pics/proportion_of_crash_severity_by_weather_conditions.png')
plt.show()
plt.close()

