import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Spotify Data Analysis",
    page_icon="🎧",
    layout="wide",
)

sns.set_theme(style="darkgrid")

DATA_PATH = "spotify_tracks.csv"
MODEL_PATH = "spotify_popularity_model.pkl"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates()
    return df


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


df = load_data()
pipeline = load_model()

genres = sorted(df["genre"].dropna().unique().tolist())

# ----------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------
st.sidebar.title("🎧 Spotify Popularity")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "📊 Explore the Data", "🔮 Predict Popularity"],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Predicts a track's popularity score using genre, duration, and explicit flag. "
    "Trained with a Random Forest Regressor."
)

# ----------------------------------------------------------------------
# Page: Overview
# ----------------------------------------------------------------------
if page == "🏠 Overview":
    st.title("🎧 Spotify Song Popularity Prediction")
    st.markdown(
        """
        This app explores a dataset of **6,300 Spotify tracks** and predicts a song's
        popularity score (0–100) from three features: **genre**, **duration**, and
        whether the track is **explicit**.
        """
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Tracks", f"{df.shape[0]:,}")
    col2.metric("Unique Genres", f"{df['genre'].nunique()}")
    col3.metric("Unique Artists", f"{df['artists'].nunique():,}")
    col4.metric("Avg. Popularity", f"{df['popularity'].mean():.1f}")

    st.markdown("### Sample of the data")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("### About the model")
    st.markdown(
        """
        - **Model:** Random Forest Regressor (`max_depth=15`, `min_samples_leaf=2`)
        - **Features used:** `genre`, `duration_ms`, `explicit`
        - **Target:** `popularity`
        - **Test R² score:** ~0.19–0.27

        The moderate accuracy is expected — the dataset doesn't include the strongest
        real-world drivers of popularity, such as artist fame, stream counts, or audio
        features like energy and danceability. Head to **Predict Popularity** to try it out,
        or **Explore the Data** for the full EDA.
        """
    )

# ----------------------------------------------------------------------
# Page: Explore the Data (EDA)
# ----------------------------------------------------------------------
elif page == "📊 Explore the Data":
    st.title("📊 Exploratory Data Analysis")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Correlation", "Popularity Distribution", "Top Artists", "Duration vs Popularity", "Genre & Explicit"]
    )

    with tab1:
        st.subheader("Feature Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
        st.caption(
            "Numerical features show weak correlation with popularity — duration alone "
            "isn't enough to explain it."
        )

    with tab2:
        st.subheader("Popularity Distribution")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(df["popularity"], bins=20, color="#1DB954", edgecolor="white")
        ax.set_xlabel("Popularity")
        ax.set_ylabel("Number of Songs")
        st.pyplot(fig)

    with tab3:
        st.subheader("Top 10 Artists by Average Popularity")
        top_artists = (
            df.groupby("artists")["popularity"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.barplot(data=top_artists, x="popularity", y="artists", ax=ax, color="#1DB954")
        ax.set_xlabel("Average Popularity")
        ax.set_ylabel("Artist(s)")
        st.pyplot(fig)

    with tab4:
        st.subheader("Duration vs Popularity")
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.scatterplot(data=df, x="duration_ms", y="popularity", alpha=0.4, ax=ax, color="#1DB954")
        st.pyplot(fig)

    with tab5:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Popularity by Genre (Top 5)")
            top_genres = df["genre"].value_counts().head(5).index
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.boxplot(
                data=df[df["genre"].isin(top_genres)],
                x="popularity", y="genre", ax=ax,
            )
            st.pyplot(fig)
        with col2:
            st.subheader("Explicit vs Non-Explicit")
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.boxplot(data=df, x="explicit", y="popularity", ax=ax)
            ax.set_xlabel("Explicit")
            ax.set_ylabel("Popularity")
            st.pyplot(fig)

# ----------------------------------------------------------------------
# Page: Predict
# ----------------------------------------------------------------------
elif page == "🔮 Predict Popularity":
    st.title("🔮 Predict Song Popularity")
    st.markdown("Enter track details below to get a predicted popularity score (0–100).")

    col1, col2 = st.columns(2)

    with col1:
        genre = st.selectbox("Genre", genres, index=genres.index("pop") if "pop" in genres else 0)
        explicit = st.toggle("Explicit content", value=False)

    with col2:
        duration_min = st.slider("Duration (minutes)", 1.0, 8.0, 3.5, 0.1)
        duration_ms = int(duration_min * 60 * 1000)
        st.caption(f"= {duration_ms:,} ms")

    if st.button("Predict Popularity", type="primary", use_container_width=True):
        input_df = pd.DataFrame(
            {"duration_ms": [duration_ms], "genre": [genre], "explicit": [explicit]}
        )
        prediction = pipeline.predict(input_df)[0]
        prediction = float(np.clip(prediction, 0, 100))

        st.markdown("### Predicted Popularity")
        st.progress(int(prediction))
        st.metric("Score (0–100)", f"{prediction:.1f}")

        if prediction >= 60:
            st.success("This track profile looks quite popular! 🔥")
        elif prediction >= 35:
            st.info("Moderate popularity expected.")
        else:
            st.warning("Predicted to be a lower-popularity track.")

    
  


