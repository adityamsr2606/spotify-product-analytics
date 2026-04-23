import os
import streamlit as st
import pandas as pd
import plotly.express as px

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Spotify Analytics",
    page_icon="🎧",
    layout="wide"
)

# -------------------------------
# BACKGROUND + STYLING (Premium UI)
# -------------------------------
st.markdown("""
<style>
.stApp {
    background: url("https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4") no-repeat center center fixed;
    background-size: cover;
}

/* Dark overlay */
.stApp::before {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.75);
    z-index: -1;
}

/* Headings */
h1, h2, h3 {
    color: #1DB954;
}

/* KPI Cards */
[data-testid="metric-container"] {
    background-color: rgba(255,255,255,0.1);
    padding: 12px;
    border-radius: 12px;
    backdrop-filter: blur(5px);
}
</style>
""", unsafe_allow_html=True)

st.title("🎧 Spotify Product Analytics Dashboard")
st.markdown("Analyze music engagement and performance patterns")

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    current_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    data_path = os.path.join(project_root, "data", "spotify", "tracks.csv")
    return pd.read_csv(data_path)

df = load_data()

# -------------------------------
# FEATURE ENGINEERING
# -------------------------------
df['duration_min'] = df['duration_ms'] / 60000

df['engagement_score'] = (
    0.4 * df['popularity'] +
    20 * df['danceability'] +
    20 * df['energy'] +
    2 * df['duration_min']
)

# Normalize
df['engagement_score'] = (
    (df['engagement_score'] - df['engagement_score'].min()) /
    (df['engagement_score'].max() - df['engagement_score'].min())
)

# Categorize
df['engagement_level'] = pd.cut(
    df['engagement_score'],
    bins=[0, 0.3, 0.7, 1],
    labels=['Low', 'Medium', 'High']
)

# -------------------------------
# SIDEBAR FILTERS
# -------------------------------
st.sidebar.header("🔍 Filters")

levels = st.sidebar.multiselect(
    "Engagement Level",
    df['engagement_level'].dropna().unique(),
    default=df['engagement_level'].dropna().unique()
)

pop_range = st.sidebar.slider(
    "Popularity Range",
    int(df['popularity'].min()),
    int(df['popularity'].max()),
    (20, 80)
)

search_song = st.sidebar.text_input("🔎 Search Song")

filtered_df = df[
    (df['engagement_level'].isin(levels)) &
    (df['popularity'].between(pop_range[0], pop_range[1]))
]

if search_song:
    filtered_df = filtered_df[
        filtered_df["name"].str.contains(search_song, case=False)
    ]

# -------------------------------
# KPI SECTION
# -------------------------------
st.subheader("📊 Key Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Tracks", len(filtered_df))
col2.metric("Avg Popularity", round(filtered_df['popularity'].mean(), 2))
col3.metric("Avg Engagement", round(filtered_df['engagement_score'].mean(), 2))
col4.metric("Avg Duration (min)", round(filtered_df['duration_min'].mean(), 2))

# -------------------------------
# HIGHLIGHT INSIGHT
# -------------------------------
st.subheader("🔥 Highlight")

top_song = filtered_df.sort_values(
    by="engagement_score", ascending=False
).iloc[0]

st.success(f"Top Performing Track: {top_song['name']}")

# -------------------------------
# VISUALS
# -------------------------------
st.subheader("📈 Engagement Insights")

col1, col2 = st.columns(2)

# Histogram
with col1:
    fig1 = px.histogram(
        filtered_df,
        x="engagement_score",
        nbins=40,
        title="Engagement Score Distribution"
    )
    fig1.update_layout(template="plotly_dark")
    st.plotly_chart(fig1, use_container_width=True)

# Scatter
with col2:
    fig2 = px.scatter(
        filtered_df,
        x="energy",
        y="popularity",
        color="engagement_level",
        hover_data=["name", "danceability", "tempo"],
        title="Energy vs Popularity"
    )
    fig2.update_layout(template="plotly_dark")
    st.plotly_chart(fig2, use_container_width=True)

# -------------------------------
# TOP TRACKS
# -------------------------------
st.subheader("🎵 Top Tracks")

top_tracks = filtered_df.sort_values(
    by="engagement_score", ascending=False
).head(10)

fig3 = px.bar(
    top_tracks,
    x="engagement_score",
    y="name",
    orientation="h",
    color="engagement_score",
    title="Top 10 Tracks by Engagement"
)

fig3.update_layout(template="plotly_dark", transition_duration=500)
st.plotly_chart(fig3, use_container_width=True)

# -------------------------------
# DURATION ANALYSIS
# -------------------------------
st.subheader("⏱️ Duration vs Engagement")

fig4 = px.scatter(
    filtered_df,
    x="duration_min",
    y="engagement_score",
    color="engagement_level",
    hover_data=["name"]
)

fig4.update_layout(template="plotly_dark")
st.plotly_chart(fig4, use_container_width=True)

# -------------------------------
# INSIGHTS
# -------------------------------
st.subheader("🧠 Insights")

if filtered_df['duration_min'].corr(filtered_df['engagement_score']) < 0:
    st.write("• Shorter songs tend to perform better.")

if filtered_df['danceability'].corr(filtered_df['engagement_score']) > 0:
    st.write("• Danceable songs drive higher engagement.")

if filtered_df['energy'].corr(filtered_df['engagement_score']) > 0:
    st.write("• High-energy tracks are more engaging.")

# -------------------------------
# DATA PREVIEW
# -------------------------------
st.subheader("📄 Data Preview")
st.dataframe(filtered_df.head(50))