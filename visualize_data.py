import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read the dataset
df = pd.read_csv('preprocessed_dataset.csv')

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

# ----------------------------
# Visualization
# ----------------------------

# Set Seaborn theme
sns.set_theme(style="whitegrid")

# Plot 1: Number of Crashes by Hour
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Crash Hour Numeric', color='skyblue')
plt.title('Number of Crashes by Hour')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Crashes')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot 2: Crash Severity Distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=df, y='Crash Severity', color='lightcoral', order=df['Crash Severity'].value_counts().index)
plt.title('Distribution of Crash Severity')
plt.xlabel('Number of Crashes')
plt.ylabel('Crash Severity')
plt.tight_layout()
plt.show()

# Plot 3: Crashes by Weather Conditions
plt.figure(figsize=(10, 6))
sns.countplot(data=df, y='Weather Conditions', color='lightgreen', order=df['Weather Conditions'].value_counts().index)
plt.title('Number of Crashes by Weather Conditions')
plt.xlabel('Number of Crashes')
plt.ylabel('Weather Conditions')
plt.tight_layout()
plt.show()



# Plot 4: Speed Limit vs. Crash Severity
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Crash Severity', y='Speed Limit', palette='Set2')
plt.title('Speed Limit by Crash Severity')
plt.xlabel('Crash Severity')
plt.ylabel('Speed Limit (mph)')
plt.tight_layout()
plt.show()

# Plot 5: Light Conditions During Crashes
plt.figure(figsize=(10, 6))
sns.countplot(data=df, y='Light Conditions', color='goldenrod', order=df['Light Conditions'].value_counts().index)
plt.title('Number of Crashes by Light Conditions')
plt.xlabel('Number of Crashes')
plt.ylabel('Light Conditions')
plt.tight_layout()
plt.show()

# Plot 6: Road Surface Conditions
plt.figure(figsize=(10, 6))
sns.countplot(data=df, y='Road Surface Condition', color='steelblue', order=df['Road Surface Condition'].value_counts().index)
plt.title('Number of Crashes by Road Surface Condition')
plt.xlabel('Number of Crashes')
plt.ylabel('Road Surface Condition')
plt.tight_layout()
plt.show()

# Plot 7: Number of Vehicles Involved in Crashes
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Number of Vehicles', color='orchid')
plt.title('Number of Vehicles Involved in Crashes')
plt.xlabel('Number of Vehicles')
plt.ylabel('Number of Crashes')
plt.tight_layout()
plt.show()

# Plot 8: Driver Age Distribution
plt.figure(figsize=(10, 6))
# Drop NaN values for plotting
young_age = df['Youngest Driver Age'].dropna()
old_age = df['Oldest Driver Age'].dropna()

sns.histplot(young_age, bins=range(0, 100, 5), kde=True, color='green', label='Youngest Driver Age', stat="density", alpha=0.6)
sns.histplot(old_age, bins=range(0, 100, 5), kde=True, color='red', label='Oldest Driver Age', stat="density", alpha=0.6)
plt.title('Driver Age Distribution')
plt.xlabel('Age')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.show()

# ----------------------------
# Additional Visualizations (Optional)
# ----------------------------

# Crash severity and Weather Condition
severity_counts = df.groupby(['Weather Conditions', 'Crash Severity']).size().reset_index(name='Count')
# Calculate the proportion of each severity within each weather condition
severity_counts['Proportion'] = severity_counts.groupby('Weather Conditions')['Count'].transform(lambda x: x / x.sum())

# Plot the proportions
plt.figure(figsize=(12, 8))
sns.barplot(x='Weather Conditions', y='Proportion', hue='Crash Severity', data=severity_counts)
plt.title('Proportion of Crash Severity by Weather Conditions')
plt.ylabel('Proportion')
plt.xticks(rotation=45)
plt.legend(title='Crash Severity')
plt.tight_layout()
plt.show()


# Example: Crash Severity over Crash Hours
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Crash Hour Numeric', hue='Crash Severity', palette='Set3')
plt.title('Crash Severity by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Crashes')
plt.legend(title='Crash Severity', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
