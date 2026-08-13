# Spotify Song Popularity Prediction

## Project Overview
This project focuses on analyzing Spotify song data and building a machine learning model to predict song popularity based on available features. The project includes data exploration, visualization, preprocessing, model training, evaluation, and feature importance analysis.

## Dataset
The dataset contains Spotify track information including:

- Track ID
- Song name
- Genre
- Artist
- Album
- Popularity score
- Duration (milliseconds)
- Explicit content indicator

## Project Objectives

- Analyze patterns in Spotify song data
- Explore relationships between song features and popularity
- Build a machine learning model to predict popularity
- Identify important factors affecting song popularity

## Data Preprocessing

The following preprocessing steps were performed:

- Dataset inspection and understanding
- Checking missing values
- Removing duplicate records
- Handling categorical features
- Preparing data for machine learning

## Exploratory Data Analysis (EDA)

The project includes visual analysis such as:

- Popularity distribution analysis
- Artist and genre-based analysis
- Relationship between song duration and popularity
- Explicit vs non-explicit song popularity comparison
- Feature correlation analysis

## Machine Learning Model

### Random Forest Regressor

A Random Forest Regression model was used to predict song popularity.

Reasons for selecting Random Forest:

- Handles non-linear relationships effectively
- Works well with mixed feature types
- Provides feature importance insights

## Model Evaluation

The model was evaluated using:

- R² Score
- RMSE (Root Mean Square Error)

The model achieved an R² score of approximately 19%–27%, indicating limited predictive performance. The result is influenced by the limited features available in the dataset.

## Feature Importance

Feature importance analysis was performed to understand which features contribute more towards popularity prediction.

## Future Improvements

Model performance can be improved by including additional features such as:

- Audio characteristics (energy, danceability, tempo, etc.)
- Artist popularity metrics
- Streaming-related information
- User engagement features

## Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SQLAlchemy

## Project Structure
