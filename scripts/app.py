import os
# os.environ["JAVA_HOME"] = r"C:\Users\User\AppData\Local\Programs\ECLIPS~1\JDK-17~1.10-"

import streamlit as st
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import collect_set, first, concat_ws


# CONFIGURATION

st.set_page_config(
    page_title="TV Series Recommender",
    page_icon="🎬",
    layout="wide"
)

# SPARK
@st.cache_resource
def get_spark():
    return SparkSession.builder \
        .appName("TV Series - Streamlit") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

spark = get_spark()
spark.sparkContext.setLogLevel("ERROR")

HDFS_PATH = "hdfs://localhost:9000/user/user/data"

def get_recommendations(df_clean, df_clustered, selected_series, n=5):
    serie_info = df_clustered[df_clustered["name"] == selected_series]

    if serie_info.empty:
        return pd.DataFrame()

    cluster_id = serie_info.iloc[0]["cluster_id"]

    same_cluster = df_clustered[
        (df_clustered["cluster_id"] == cluster_id)
        & (df_clustered["name"] != selected_series)
        & (df_clustered["vote_count"] >= 10)
    ].copy()

    clean_columns = [
        "show_id",
        "genres",
        "network",
        "number_of_seasons",
        "vote_average",
        "decade"
    ]

    results = same_cluster.merge(
        df_clean[clean_columns],
        on="show_id",
        how="left"
    )

    results = results.sort_values(
        "vote_quality",
        ascending=False
    )

    results = results.drop_duplicates("name")

    return results.head(n)


def get_top_by_genre(df_clean, genre, n=5):
    results = df_clean[
        df_clean["genres"].str.contains(genre, na=False)
        & (df_clean["vote_count"] >= 10)
    ].copy()

    results = results.sort_values(
        "vote_quality",
        ascending=False
    )

    results = results.drop_duplicates("name")

    return results.head(n)


def search_multifiltres(
    df_clean,
    genre=None,
    decade=None,
    note_min=0,
    votes_min=10,
    length_range=None,
    n=5
):
    results = df_clean[
        df_clean["vote_count"] >= votes_min
    ].copy()

    if genre and genre != "Tous":
        results = results[
            results["genres"].str.contains(genre, na=False)
        ]

    if decade and decade != "Toutes":
        results = results[
            results["decade"] == float(decade)
        ]

    if note_min:
        results = results[
            results["vote_average"] >= note_min
        ]

    if length_range:
        minimum = length_range[0]
        maximum = length_range[1]

        results = results[
            (results["number_of_seasons"] >= minimum)
            & (results["number_of_seasons"] <= maximum)
        ]

    results = results.sort_values(
        "vote_quality",
        ascending=False
    )

    results = results.drop_duplicates("name")

    return results.head(n)

# CHARGER LES DONNEES
@st.cache_data
def load_data():
    # shows_clean — pour les filtres et infos d'affichage
    clean = spark.read.parquet(f"{HDFS_PATH}/shows_clean") \
        .groupBy("show_id", "name", "popularity", "vote_average",
                 "vote_count", "decade", "number_of_seasons",
                 "number_of_episodes", "episode_run_time", "status_name") \
        .agg(
            concat_ws(", ", collect_set("genre_name")).alias("genres"),
            first("network_name").alias("network")
        ).toPandas()

    clean["genres"] = clean["genres"].replace("", "Non renseigné")
    clean["genres"] = clean["genres"].fillna("Non renseigné")
    clean["vote_quality"] = clean["vote_average"] * (
        clean["vote_count"] / (clean["vote_count"] + 100)
    )

    # shows_clustered — résultat du modèle KMeans
    clustered = spark.read.parquet(f"{HDFS_PATH}/shows_clustered").toPandas()

    return clean, clustered

with st.spinner("⏳ Chargement des données depuis HDFS..."):
    df_clean, df_clustered = load_data()

# GENRES
all_genres = set()
for g in df_clean["genres"].dropna():
    for genre in g.split(", "):
        if genre.strip() and genre.strip() != "Non renseigné":
            all_genres.add(genre.strip())
all_genres = sorted(list(all_genres))

# CARTE SERIE
def show_card(row):
    note = round(float(row.get("vote_average", 0)), 1)
    votes = int(row.get("vote_count", 0))
    decade = f"{int(row['decade'])}s" if pd.notna(row.get("decade")) else "N/A"
    genres = row.get("genres", "Non renseigné")
    network = row.get("network", "")
    seasons = int(row["number_of_seasons"]) if pd.notna(row.get("number_of_seasons")) else "?"

    if str(network) == "nan" or not network or pd.isna(network):
        network = "Chaîne inconnue"

    note_color = "#22c55e" if note >= 7 else "#f59e0b" if note >= 5 else "#ef4444"

    st.markdown(f"""
    <div style="
        background: #1e1e2e;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 8px 0;
        border-left: 4px solid #7c3aed;
    ">
        <h4 style="color: #e2e8f0; margin: 0 0 10px 0;">🎬 {row['name']}</h4>
        <p style="color: #94a3b8; margin: 4px 0; font-size: 14px;">
            ⭐ <b style="color:{note_color}">{note}/10</b>
            <span style="color:#64748b; font-size:11px"> ({votes} votes)</span>
            &nbsp;&nbsp;|&nbsp;&nbsp;
            📅 <b style="color:#60a5fa">{decade}</b>
            &nbsp;&nbsp;|&nbsp;&nbsp;
            🎭 <b style="color:#34d399">{genres}</b>
        </p>
        <p style="color: #64748b; margin: 6px 0 0 0; font-size: 12px;">
            📺 {network} &nbsp;&nbsp;|&nbsp;&nbsp; 🗂️ {seasons} saison(s)
        </p>
    </div>
    """, unsafe_allow_html=True)

# TITRE
st.title("🎬 TV Series Recommender")
st.markdown("*Trouve ta prochaine série préférée*")
st.divider()

# ONGLETS
tab1, tab2, tab3 = st.tabs([
    "🔍 Par série",
    "🎭 Par genre",
    "🎛️ Multi-filtres"
])

# ONGLET 1 — PAR SERIE (utilise le modèle KMeans)
with tab1:
    st.header("Recommandation à partir d'une série")
    st.markdown("Choisis une série que tu as aimée et on te recommande des séries similaires.")

    series_names = sorted(df_clustered["name"].dropna().unique().tolist())
    selected_series = st.selectbox("Choisis une série", series_names, key="tab1_series")

    if selected_series:
        serie_info = df_clean[df_clean["name"] == selected_series]
        if not serie_info.empty:
            st.markdown("**Série sélectionnée :**")
            show_card(serie_info.iloc[0])
            st.markdown("---")

        recommendations = get_recommendations(df_clean, df_clustered, selected_series)

        st.subheader(f"Top 5 séries similaires à *{selected_series}*")
        if recommendations.empty:
            st.warning("Aucune recommandation trouvée.")
        else:
            for _, row in recommendations.iterrows():
                show_card(row)

# ONGLET 2 — PAR GENRE
with tab2:
    st.header("Top séries par genre")
    st.markdown("Choisis un genre et découvre les meilleures séries.")

    selected_genre = st.selectbox("Choisis un genre", all_genres, key="tab2_genre")

    if selected_genre:
        filtered = get_top_by_genre(df_clean, selected_genre)
        st.subheader(f"Top 5 séries — {selected_genre}")
        if filtered.empty:
            st.warning("Aucune série trouvée pour ce genre.")
        else:
            for _, row in filtered.iterrows():
                show_card(row)

# ONGLET 3 — MULTI-FILTRES
with tab3:
    st.header("Recherche personnalisée")
    st.markdown("Combine plusieurs critères pour trouver ta série idéale.")

    col1, col2 = st.columns(2)

    with col1:
        genre_filter = st.selectbox("Genre", ["Tous"] + all_genres, key="tab3_genre")
        decade_options = sorted([int(d) for d in df_clean["decade"].dropna().unique()])
        decade_filter = st.selectbox(
            "Décennie", ["Toutes"] + [str(d) for d in decade_options], key="tab3_decade"
        )

    with col2:
        note_min = st.slider("Note minimum", 0.0, 10.0, 6.0, 0.5, key="tab3_note")
        length_options = {
            "Toutes": None,
            "Mini-série (1 saison)": (1, 1),
            "Courte (2-3 saisons)": (2, 3),
            "Moyenne (4-7 saisons)": (4, 7),
            "Longue (8+ saisons)": (8, 999)
        }
        length_filter = st.selectbox(
            "Longueur de série", list(length_options.keys()), key="tab3_length"
        )

    votes_min = st.slider("Nombre minimum de votes", 0, 500, 10, 10, key="tab3_votes")

    if st.button("🔍 Rechercher", type="primary"):
        results = search_multifiltres(
            df_clean,
            genre=genre_filter,
            decade=decade_filter,
            note_min=note_min,
            votes_min=votes_min,
            length_range=length_options[length_filter]
        )

        st.subheader("Top 5 résultats")
        if results.empty:
            st.warning("Aucune série trouvée. Essaie d'élargir tes critères.")
        else:
            for _, row in results.iterrows():
                show_card(row)
