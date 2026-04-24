# Spotify Product Analytics & Recommendation System

This project is a data analytics and machine learning dashboard designed to analyze how different audio features influence song performance and user engagement. It also includes a recommendation system that suggests songs based on similarity and user preferences.

The goal of this project is to apply core data science concepts such as feature engineering, clustering, and recommendation systems on a real-world dataset, while presenting the results through an interactive dashboard.


## Live Application

You can explore the deployed application here:
https://spotify-interactive-analytic.streamlit.app/


## What this project does

* Analyzes music data to understand what drives song engagement
* Segments songs into meaningful groups using clustering
* Recommends similar songs using a content-based approach
* Provides simple user-based personalization
* Displays insights through an interactive dashboard


## Key Features

### Analytics Dashboard

* Custom engagement score built using popularity, energy, danceability, and duration
* Key metrics such as average popularity, engagement, and duration
* Interactive filtering for better data exploration


### Clustering (Song Segmentation)

Songs are grouped into three clusters using KMeans:

* Chill
* Party
* Focus

This helps identify patterns in different types of music.


### Recommendation System

#### Similar Song Recommendation

* Uses cosine similarity on audio features
* Suggests songs similar to a selected track

#### Fuzzy Search

* Handles typos in user input

#### Personalized Recommendations

* Builds a user preference profile based on selected songs
* Recommends tracks aligned with user taste


### A/B Style Insight

* Compares high-energy vs low-energy songs
* Measures difference in engagement
* Uses statistical testing to check significance


## Key Insights

* High-energy songs tend to achieve higher engagement
* Danceable tracks generally perform better
* Shorter songs show slightly better performance trends
* A large portion of songs fall into the medium engagement range


## Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn (clustering, scaling, similarity)
* Streamlit (dashboard)
* Plotly and Seaborn (visualization)
* SciPy (statistical testing)


## How to Run

```id="r4m3zx"
git clone https://github.com/adityamsr2606/spotify-product-analytics.git
cd spotify-product-analytics
pip install -r requirements.txt
streamlit run dashboard/app.py
```


## Motivation

This project was built to move beyond theoretical learning and apply data analytics and machine learning techniques to a practical use case. It focuses on understanding user engagement and building a working recommendation system within an interactive application.


## Future Improvements

* Enhance recommendation system using hybrid or deep learning approaches
* Add playlist generation functionality
* Incorporate real-time user interaction data
* Integrate with Spotify API


## Author

Aditya Mohan Srivastava



