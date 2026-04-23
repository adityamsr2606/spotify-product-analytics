import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import ttest_ind

from rapidfuzz import process, fuzz

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Spotify Analytics", page_icon="🎧", layout="wide")

# -------------------------------
# UI STYLE
# -------------------------------
st.markdown("""
<style>
.stApp { background-color: #121212; color: white; }
section[data-testid="stSidebar"] { background-color: #000; }
button { background-color: #1DB954 !important; color: white !important; border-radius: 20px !important; }
</style>
""", unsafe_allow_html=True)

st.title("🎧 Spotify Product Analytics Dashboard")

# -------------------------------
# LOAD DATA (FIXED)
# -------------------------------
@st.cache_data
def load_data():
    data_url = "https://drive.google.com/uc?id=1dssK34qFIjUguEozPywTE1Ta6sgGfSZZ"
    df = pd.read_csv(data_url, low_memory=False)
    df = df.dropna()
    df = df.drop_duplicates()
    return df

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

df['engagement_score'] = (
    (df['engagement_score'] - df['engagement_score'].min()) /
    (df['engagement_score'].max() - df['engagement_score'].min())
)

df['engagement_level'] = pd.cut(
    df['engagement_score'],
    bins=[0, 0.3, 0.7, 1],
    labels=['Low', 'Medium', 'High']
)

# -------------------------------
# CLUSTERING (CACHED)
# -------------------------------
@st.cache_resource
def compute_clusters(data):
    features = data[['danceability','energy','valence','tempo']]
    scaled = StandardScaler().fit_transform(features)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled)
    return clusters, scaled

df['cluster'], scaled_cluster = compute_clusters(df)

cluster_map = {0:"Chill 😌",1:"Party 🎉",2:"Focus 🧘"}
df['cluster_label'] = df['cluster'].map(cluster_map)

# -------------------------------
# RECOMMENDER (OPTIMIZED)
# -------------------------------
@st.cache_resource
def build_recommender(data):
    features = data[['danceability','energy','valence','tempo','acousticness']]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    return pd.DataFrame(scaled, index=data.index)

scaled_df = build_recommender(df)

def recommend_songs(song_name, top_n=5):
    match = df[df['name'].str.lower() == song_name.lower()]
    if match.empty:
        return None

    idx = match.index[0]
    vec = scaled_df.loc[idx].values.reshape(1,-1)
    sim = cosine_similarity(vec, scaled_df)[0]

    indices = sim.argsort()[::-1][1:top_n+1]
    return df[['name','artists','engagement_level']].iloc[indices]

# -------------------------------
# FUZZY SEARCH
# -------------------------------
def fuzzy_search(query, choices, limit=5):
    results = process.extract(query, choices, scorer=fuzz.token_sort_ratio, limit=limit)
    return [r[0] for r in results if r[1] > 60]

# -------------------------------
# USER PERSONALIZATION
# -------------------------------
def build_user_profile(selected_songs):
    user_data = df[df['name'].isin(selected_songs)]
    if user_data.empty:
        return None
    return scaled_df.loc[user_data.index].mean().values.reshape(1,-1)

def recommend_for_user(user_vector, top_n=5):
    sim = cosine_similarity(user_vector, scaled_df)[0]
    indices = sim.argsort()[::-1][:top_n]
    return df[['name','artists','engagement_level']].iloc[indices]

# -------------------------------
# SIDEBAR FILTERS
# -------------------------------
st.sidebar.header("Filters")

levels = st.sidebar.multiselect(
    "Engagement Level",
    df['engagement_level'].dropna().unique(),
    default=df['engagement_level'].dropna().unique()
)

cluster_filter = st.sidebar.multiselect(
    "Cluster",
    df['cluster_label'].dropna().unique(),
    default=df['cluster_label'].dropna().unique()
)

filtered_df = df[
    df['engagement_level'].isin(levels) &
    df['cluster_label'].isin(cluster_filter)
]

if filtered_df.empty:
    st.warning("No data after filters")
    st.stop()

# -------------------------------
# KPIs
# -------------------------------
st.subheader("📊 KPIs")

c1,c2,c3 = st.columns(3)
c1.metric("Tracks", f"{len(filtered_df):,}")
c2.metric("Avg Engagement", round(filtered_df['engagement_score'].mean(),2))
c3.metric("Avg Popularity", round(filtered_df['popularity'].mean(),2))

# -------------------------------
# VISUALS (COLORBLIND SAFE)
# -------------------------------
st.subheader("📈 Insights")

col1,col2 = st.columns(2)

with col1:
    fig1 = px.histogram(filtered_df, x="engagement_score", color="engagement_level",
                        color_discrete_sequence=px.colors.qualitative.Safe)
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.scatter(filtered_df, x="energy", y="popularity", color="engagement_level",
                      color_discrete_sequence=px.colors.qualitative.Safe)
    st.plotly_chart(fig2, use_container_width=True)

# -------------------------------
# CLUSTER VIEW
# -------------------------------
st.subheader("🎯 Cluster Analysis")

fig_cluster = px.scatter(
    filtered_df,
    x="danceability",
    y="energy",
    color="cluster_label",
    color_discrete_sequence=px.colors.qualitative.Safe
)
st.plotly_chart(fig_cluster, use_container_width=True)

# -------------------------------
# A/B TEST
# -------------------------------
st.subheader("🧪 A/B Insight")

high = df[df['energy'] > df['energy'].median()]
low = df[df['energy'] <= df['energy'].median()]

lift = ((high['engagement_score'].mean() - low['engagement_score'].mean()) / low['engagement_score'].mean())*100
_, p = ttest_ind(high['engagement_score'], low['engagement_score'])

st.write(f"Lift: {lift:.2f}%")
st.write(f"P-value: {p:.5f}")

# -------------------------------
# SMART SEARCH
# -------------------------------
st.subheader("🎧 Smart Song Search")

search_query = st.text_input("Type song name")

if search_query:
    matches = fuzzy_search(search_query, df['name'].dropna().unique())

    if matches:
        selected_song = st.selectbox("Did you mean:", matches)
    else:
        st.warning("No similar songs found")
        selected_song = None
else:
    selected_song = None

# -------------------------------
# RECOMMENDATIONS
# -------------------------------
if selected_song:
    st.markdown(f"### 🎵 {selected_song}")

    results = recommend_songs(selected_song)

    if results is not None:
        for _, row in results.iterrows():
            st.markdown(f"**{row['name']}** — {row['artists']} ({row['engagement_level']})")

# -------------------------------
# PERSONALIZATION
# -------------------------------
st.subheader("👤 Personalized Recommendations")

user_songs = st.multiselect("Select songs you like", df['name'].dropna().sample(500))

if st.button("Get Personalized Recommendations"):
    vec = build_user_profile(user_songs)
    if vec is not None:
        st.dataframe(recommend_for_user(vec))
    else:
        st.warning("Select songs")

# -------------------------------
# DATA
# -------------------------------
st.subheader("📄 Data Preview")
st.dataframe(filtered_df.head(50))