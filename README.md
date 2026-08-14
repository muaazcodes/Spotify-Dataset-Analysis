# 🎧 Spotify Song Popularity Prediction

A data science project that explores a dataset of 6,300 Spotify tracks and builds a
machine learning model to predict a song's popularity score (0–100). Includes a full
EDA notebook, a trained model, and an interactive Streamlit web app.

## 📊 Dataset

The dataset contains 6,300 tracks with the following columns:

| Column        | Description                          |
|---------------|---------------------------------------|
| `id`          | Unique Spotify track ID               |
| `name`        | Track name                            |
| `genre`       | Genre (126 unique genres)             |
| `artists`     | Artist(s) name                        |
| `album`       | Album name                            |
| `popularity`  | Popularity score (0–100) — **target** |
| `duration_ms` | Track duration in milliseconds        |
| `explicit`    | Whether the track has explicit content|

No missing values or duplicate rows were found in the dataset.

## 🔍 Project Workflow

1. **Data Cleaning** — checked for missing values and duplicates
2. **Exploratory Data Analysis (Seaborn/Matplotlib)** — correlation heatmap, popularity
   distribution, top artists by average popularity, genre and explicit-content
   comparisons
3. **Model Training (Scikit-learn)** — a `RandomForestRegressor` trained inside a
   `Pipeline` with a `ColumnTransformer` (`StandardScaler` for numeric features,
   `OneHotEncoder` for categorical features)
4. **Feature Importance** — identified which features influence predictions most
5. **Model Persistence** — trained pipeline saved with `joblib`
6. **Database Storage (SQLAlchemy)** — dataset and predictions stored in a local
   SQLite database
7. **Streamlit App** — interactive UI for exploring the data and getting live
   predictions

## 🤖 Model Details

- **Algorithm:** Random Forest Regressor (`max_depth=15`, `min_samples_leaf=2`)
- **Features used:** `genre`, `duration_ms`, `explicit`
- **Target:** `popularity`
- **Test R² score:** ~0.19–0.27

The moderate accuracy is expected given the limited feature set. Real-world
popularity is driven mainly by factors this dataset doesn't include — artist fame,
stream counts, playlist adds, and audio characteristics like energy or
danceability.

## 🚀 Streamlit App

An interactive app with four sections:

- **Overview** — dataset stats and model summary
- **Explore the Data** — correlation heatmap, popularity distribution, top artists,
  duration vs. popularity, genre/explicit comparisons
- **Predict Popularity** — pick a genre, duration, and explicit flag to get a live
  popularity prediction
- **Database** — run SQL queries against the SQLite-stored dataset

### Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Make sure `app.py`, `requirements.txt`, `spotify_popularity_model.pkl`, and
`spotify_tracks.csv` are all in the same folder.

## 🛠️ Tech Stack

- Python, Pandas, NumPy
- Scikit-learn (Pipeline, ColumnTransformer, RandomForestRegressor)
- Seaborn, Matplotlib
- SQLAlchemy
- Streamlit
- Joblib

## 📈 Future Improvements

- Add audio features (energy, danceability, tempo, valence) to the dataset
- Include artist popularity/follower counts
- Try additional models (Gradient Boosting, XGBoost) for comparison
- Add hyperparameter tuning with `GridSearchCV`

## 👤 Author

Muaaz — [GitHub: muaazcodes](https://github.com/muaazcodes)
