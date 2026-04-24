import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import ttest_ind

from rapidfuzz import process, fuzz

# page configurations
st.set_page_config(page_title="Spotify Analytics", page_icon="🎧", layout="wide")

# for device detection
st.markdown("""
<script>
function sendSize() {
    const width = window.innerWidth;
    window.parent.postMessage({type: "streamlit:setComponentValue", value: width}, "*");
}
window.addEventListener("load", sendSize);
window.addEventListener("resize", sendSize);
</script>
""", unsafe_allow_html=True)

screen_width = st.session_state.get("screen_width", 1200)
is_mobile = screen_width < 768

# UI
st.markdown("""
<style>
.stApp { background-color: #121212; color: white; }

section[data-testid="stSidebar"] {
    background-color: #000;
}

.card {
    background-color: rgba(255,255,255,0.08);
    padding: 12px;
    border-radius: 10px;
    margin-bottom: 10px;
}

button {
    background-color: #1DB954 !important;
    color: white !important;
    border-radius: 20px !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🎧 Spotify Product Analytics Dashboard")

# loading data 
@st.cache_data
def load_data():
    url = "https://drive.google.com/uc?id=1dssK34qFIjUguEozPywTE1Ta6sgGfSZZ"
    return pd.read_csv(url)

df = load_data()

# features
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

# clustering
features_cluster = df[['danceability','energy','valence','tempo']]
scaled_cluster = StandardScaler().fit_transform(features_cluster)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(scaled_cluster)

cluster_map = {0:"Chill 😌",1:"Party 🎉",2:"Focus 🧘"}
df['cluster_label'] = df['cluster'].map(cluster_map)

# recommanding system

features_rec = df[['danceability','energy','valence','tempo','acousticness']]
scaled_rec = StandardScaler().fit_transform(features_rec)
scaled_df = pd.DataFrame(scaled_rec, index=df.index)

def recommend_songs(song_name, top_n=5):
    match = df[df['name'].str.lower() == song_name.lower()]
    if match.empty:
        return None

    idx = match.index[0]
    vec = scaled_df.loc[idx].values.reshape(1,-1)
    sim = cosine_similarity(vec, scaled_df)[0]

    indices = sim.argsort()[::-1][1:top_n+1]
    return df[['name','artists','engagement_level']].iloc[indices]

def fuzzy_search(query, choices, limit=5):
    results = process.extract(
        query,
        choices,
        scorer=fuzz.token_sort_ratio,
        limit=limit
    )
    return [r[0] for r in results if r[1] > 60]

# personalization

def build_user_profile(selected_songs):
    user_data = df[df['name'].isin(selected_songs)]
    if user_data.empty:
        return None
    return scaled_df.loc[user_data.index].mean().values.reshape(1,-1)

def recommend_for_user(user_vector, top_n=5):
    sim = cosine_similarity(user_vector, scaled_df)[0]
    indices = sim.argsort()[::-1][:top_n]
    return df[['name','artists','engagement_level']].iloc[indices]

# sidebar filters

st.sidebar.header("Filters")

levels = st.sidebar.multiselect(
    "Engagement Level",
    df['engagement_level'].dropna().unique(),
    default=df['engagement_level'].dropna().unique()
)

cluster_filter = st.sidebar.multiselect(
    "Cluster",
    df['cluster_label'].unique(),
    default=df['cluster_label'].unique()
)

filtered_df = df[
    df['engagement_level'].isin(levels) &
    df['cluster_label'].isin(cluster_filter)
]

# KPIs

st.subheader("📊 KPIs")

if is_mobile:
    c1 = st.container()
    c2 = st.container()
    c3 = st.container()
else:
    c1, c2, c3 = st.columns(3)

c1.metric("Tracks", len(filtered_df))
c2.metric("Avg Engagement", round(filtered_df['engagement_score'].mean(),2))
c3.metric("Avg Popularity", round(filtered_df['popularity'].mean(),2))

# insights

st.subheader("📈 Insights")

if is_mobile:
    col1 = st.container()
    col2 = st.container()
else:
    col1, col2 = st.columns(2)

with col1:
    fig1 = px.histogram(filtered_df, x="engagement_score", color="engagement_level")
    fig1.update_layout(height=300 if is_mobile else 400)
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.scatter(filtered_df, x="energy", y="popularity", color="engagement_level")
    fig2.update_layout(height=300 if is_mobile else 400)
    st.plotly_chart(fig2, use_container_width=True)

# cluster analysis

st.subheader("🎯 Cluster Analysis")

fig_cluster = px.scatter(
    filtered_df,
    x="danceability",
    y="energy",
    color="cluster_label"
)
fig_cluster.update_layout(height=300 if is_mobile else 400)

st.plotly_chart(fig_cluster, use_container_width=True)

# A/B testing

st.subheader("🧪 A/B Insight")

high = df[df['energy'] > df['energy'].median()]
low = df[df['energy'] <= df['energy'].median()]

lift = ((high['engagement_score'].mean() - low['engagement_score'].mean()) / low['engagement_score'].mean())*100
_, p = ttest_ind(high['engagement_score'], low['engagement_score'])

st.write(f"Lift: {lift:.2f}%")
st.write(f"P-value: {p:.5f}")

# recommander UI

st.subheader("🎧 Smart Search")

query = st.text_input("Search song (typo supported)")

selected_song = None

if query:
    matches = fuzzy_search(query, df['name'].dropna().unique())
    if matches:
        selected_song = st.selectbox("Did you mean:", matches)

# recommandation results

if selected_song:
    st.markdown(f"### 🎵 {selected_song}")

    results = recommend_songs(selected_song)

    if results is not None:
        if is_mobile:
            for _, row in results.iterrows():
                st.markdown(f"""
                <div class="card">
                <b>{row['name']}</b><br>
                👤 {row['artists']}<br>
                🎯 {row['engagement_level']}
                </div>
                """, unsafe_allow_html=True)
        else:
            cols = st.columns(2)
            for i, (_, row) in enumerate(results.iterrows()):
                with cols[i % 2]:
                    st.markdown(f"""
                    <div class="card">
                    <b>{row['name']}</b><br>
                    👤 {row['artists']}<br>
                    🎯 {row['engagement_level']}
                    </div>
                    """, unsafe_allow_html=True)

# personalizations

st.subheader("👤 Personalized Recommendations")

user_songs = st.multiselect(
    "Select songs you like",
    df['name'].dropna().sample(500)
)

if st.button("Get Recommendations"):
    vec = build_user_profile(user_songs)
    if vec is not None:
        st.dataframe(recommend_for_user(vec))
    else:
        st.warning("Select songs")

# data preview

st.subheader("📄 Data Preview")
st.dataframe(filtered_df.head(50))