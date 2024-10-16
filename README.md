# CS506-Final_Project
# Traffic Accident Prediction Based on Weather Conditions and Driver Age

## Project Overview
This project aims to explore and model the relationship between weather conditions, driver age, and the occurrence of traffic accidents. By focusing on how various weather conditions impact different age groups and influence the frequency and severity of accidents, we plan to uncover patterns and risk factors that can lead to informed decisions for enhancing road safety. Such insights are intended to support transportation authorities, policymakers, and the public in implementing preventive and targeted road safety measures, especially during adverse weather conditions.

## Objectives and Goals
1. **Data Cleaning and Preprocessing**: Address potential missing data or Inconsistencies between traffic and weather datasets.
2. **Develop a Predictive Model**: The core goal is to develop a machine learning model capable of predicting both the likelihood of traffic accidents and their severity (lethal vs. non-lethal) based on historical weather and driver age data. This model will serve as a foundation for understanding the relationship between environmental conditions and accident occurrence/severity.
3. **Analyze Age-Based Impact**: A key focus will be analyzing how weather conditions affect drivers across different age groups. By examining age-specific data, we aim to identify vulnerabilities and determine how varying weather conditions disproportionately impact drivers of certain ages.
4. **Identify Key Risk Factors**: This project will identify which weather conditions (e.g., rain, snow, fog) and age most significantly contribute to traffic accidents. The goal is to establish which factors are predictive of both the occurrence and severity of accidents under different weather conditions.

## Data Collection
- **Traffic Accident Data**: The primary sources for traffic accident data will be government databases, specifically  [Massachusetts Government Website](https://www.mass.gov) and [MassDOT](https://www.mass.gov/orgs/massachusetts-department-of-transportation). These sources provide access to historical accident data, which will be foundational for model training and analysis. Inchluding:
  - Date and Time: The specific timestamp of each accident will allow the integration of corresponding weather data.
  - Location: Geographic information will be crucial for aligning accident data with weather conditions.
  - Severity: Information on the severity of each incident (fatal, injury, property damage only) will enable a nuanced analysis of weather's role in both the occurrence and severity of accidents.
  - Collision and Vehicle Types: Understanding the nature of collisions and vehicle types involved provides context for model prediction.
  - Driver Age: Age will be the key demographic factor analyzed, allowing the project to explore how different age groups respond to weather conditions and identify the most at-risk populations.
- **Weather Data**: Weather data will be sourced from reputable providers, such as the National Oceanic and Atmospheric Administration [NOAA](https://www.noaa.gov) or weather APIs (e.g., OpenWeatherMap). These sources provide extensive historical weather data, which will be critical in correlating environmental conditions with traffic incidents. Including:
  - Date and Time: Obtain the time and location information in order to merge with timestamps of traffic accidents. 
  - Temperature: Fluctuations in temperature may correlate with accident rates or severity, especially during extreme conditions.
  - Precipitation: Information on the type (e.g., rain, snow) and amount of precipitation will help model weather-related risk.
  - Humidity and Wind Speed: These parameters can influence road conditions and visibility, impacting driver behavior and accident risk.
  - Visibility: Low visibility due to weather (e.g., fog, heavy rain) is a known contributing factor to accidents.
  - General Conditions: Broad weather descriptions (e.g., clear, foggy, stormy) will provide context to detailed weather parameters.
- **Data Cleaning and Preprocessing Plan**: To ensure the data integrity, we will employ the following methods for handling potential missing data in traffic accident and weather datasets.
  - Handling Missing Data: Implement strategies such as imputation for missing values in both traffic and weather datasets. For example, use mean or median values for numerical data or the most frequent category for categorical data.
	-	Addressing Inconsistencies: Develop matching algorithms to align traffic accident records with the corresponding weather data based on timestamps and geographic locations. Use spatial and temporal thresholds to handle slight mismatches.
	-	Data Normalization and Encoding: Normalize numerical features and encode categorical variables using techniques like One-Hot Encoding or Label Encoding to prepare data for modeling.

## Models Explored
- **Linear Regression**: For predicting the occurrence of accidents (binary classification of accident vs. no accident), logistic regression can serve as a simple baseline model.For predicting the severity of accidents (regression on severity scale), linear regression can be applied initially as a baseline.
- **Decision Trees**: Given the categorical nature of many features (e.g., age groups, weather conditions), decision trees are a natural choice for capturing complex interactions and non-linear relationships between weather conditions, age, and accident outcomes.
- **Gradient Boosting (XGBoost, LightGBM)**: To improve predictive performance, gradient boosting algorithms like XGBoost or LightGBM will be explored. These models are powerful for both regression and classification tasks and can handle a mix of continuous and categorical variables well.
XGBoost is particularly effective for tabular data and can help identify the most predictive features for both accident occurrence and severity.
- **Multilayer Perceptron (MLP)**: MLPs work well with structured, tabular data, especially when we have continuous (e.g., temperature, wind speed) and categorical (e.g., weather type, age group) variables.
We can apply MLP as the advanced model for improving predictive accuracy, especially when feature relationships are difficult to model with linear methods or decision trees. In our case, the features of the dataset (e.g., weather conditions, age, time of day) are fed into the input layer. The output layer produces the final prediction, which can be binary classification (lethal vs. non-lethal) or regression (accident severity score).
- **Hyperparameter Tuning**: Utilize techniques like Grid Search and Randomized Search to find the optimal hyperparameters for the models, which can effectively enhance model performance.

## Data Visualization
We plan to use the following visualizations:
- **Heatmap**: A heatmap visually represents the correlation between weather conditions, age, and accident frequency. It uses color gradients to show the strength of the relationship between different variables, helping identify the most influential factors impacting accidents.
- **Correlation Matrix**: A correlation matrix displays the relationships between multiple variables, such as age, weather conditions, and accident severity. This matrix will show highly correlated factors, which can guide further analysis and model development.
- **Box Plots**: Show distributions of accident severity across different conditions, such as various weather types or age groups. This helps in understanding the spread and central tendency of the data.
- **Trend Lines**: Added to scatter plots to highlight trends and relationships between variables, such as the relationship between temperature and accident frequency.

Libraries used:
- **Matplotlib**: For basic visualizations like histograms and scatter plots.
- **Seaborn**: For advanced statistical visualizations like heatmaps and correlation matrices.
- **Pandas and NumPy**: For data manipulation and numerical operations.
- **Scikit-learn**: For implementing machine learning models and preprocessing steps.
- **XGBoost**: For advanced gradient boosting algorithms.
We will use Matplotlib and Seaborn for data visualization. Matplotlib is ideal for creating basic visualizations,  such as histograms and scatter plots, thus allowing us to gain a basic understanding of the distribution of accident severity (lethal vs. non-lethal) across different factors including driver age, weather conditions, or time of day. Seaborn will be used for more advanced statistical visualizations, such as heatmaps and correlation matrices, which will help us explore and visualize the relationships between weather conditions, age, and accident severity, offering deeper insights into how these factors interact. We will also use Pandas and numpy for data cleaning. Scikit-learn for implementing machine learning model. XGBoost for advanced gradient boosting algorithms.

## Test Plan
- **Training Set**: We will use two years’ worth of traffic accident data (including weather conditions and driver age information) for training the model. This will help model learn relationships between weather conditions, driver age, and accident severity.
- **Testing Set**: The remaining 20% of the data will be reserved for model evaluation. Specifically, we will use data from a separate 6-month period that follows the training set time frame. This temporal separation ensures that the model is tested on future, unseen data, improving the reliability of performance assessments.

### Prerequisites
- Python 3.9
- Libraries: 
- **Data Manipulation**: `pandas`, `numpy`
- **Data Visualization**: `matplotlib`, `seaborn`
- **Machine Learning**: `scikit-learn`, `xgboost`, `lightgbm`


