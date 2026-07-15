import pandas as pd
import streamlit as st

from pyspark.sql import SparkSession
from pyspark.sql.functions import collect_set, concat_ws, first


# ============================================================
# 1. CONFIGURATION
# ============================================================

HDFS_PATH = "hdfs://localhost:9000/user/user/data"

st.set_page_config(
    page_title="Recommandation de séries",
    page_icon="🎬",
    layout="wide"
)


# ============================================================
# 2. DÉMARRAGE DE SPARK
# ============================================================

@st.cache_resource
def create_spark_session():

    spark_session = (
        SparkSession.builder
        .appName("TV Series - Streamlit")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )

    spark_session.sparkContext.setLogLevel("ERROR")

    return spark_session


spark = create_spark_session()


# ============================================================
# 3. CHARGEMENT DES DONNÉES
# ============================================================

@st.cache_data
def load_data():

    clean_path = f"{HDFS_PATH}/shows_clean"
    clustered_path = f"{HDFS_PATH}/shows_clustered"

    clean_spark = spark.read.parquet(
        clean_path
    )

    # Une seule ligne par série
    clean_spark = clean_spark.groupBy(
        "show_id",
        "name",
        "popularity",
        "vote_average",
        "vote_count",
        "decade",
        "number_of_seasons",
        "number_of_episodes",
        "episode_run_time",
        "status_name"
    ).agg(
        concat_ws(
            ", ",
            collect_set("genre_name")
        ).alias("genres"),

        first(
            "network_name"
        ).alias("network")
    )

    df_clean = clean_spark.toPandas()

    # Remplacement des genres manquants
    df_clean["genres"] = df_clean["genres"].replace(
        "",
        "Non renseigné"
    )

    df_clean["genres"] = df_clean["genres"].fillna(
        "Non renseigné"
    )

    # Même formule que pendant l'entraînement
    df_clean["vote_quality"] = (
        df_clean["vote_average"]
        * (
            df_clean["vote_count"]
            / (df_clean["vote_count"] + 100)
        )
    )

    clustered_spark = spark.read.parquet(
        clustered_path
    )

    df_clustered = clustered_spark.toPandas()

    return df_clean, df_clustered


with st.spinner("Chargement des données depuis HDFS..."):

    df_clean, df_clustered = load_data()


# ============================================================
# 4. LISTE DES GENRES
# ============================================================

all_genres = set()

for genres_text in df_clean["genres"]:

    if pd.isna(genres_text):
        continue

    genres = genres_text.split(", ")

    for genre in genres:

        genre = genre.strip()

        if genre == "":
            continue

        if genre == "Non renseigné":
            continue

        all_genres.add(genre)


all_genres = sorted(
    list(all_genres)
)


# ============================================================
# 5. RECOMMANDATION PAR CLUSTER
# ============================================================

def get_recommendations(
    clean_data,
    clustered_data,
    selected_series,
    number_of_results=5
):

    selected_rows = clustered_data[
        clustered_data["name"] == selected_series
    ]

    if selected_rows.empty:
        return pd.DataFrame()

    selected_cluster = selected_rows.iloc[0][
        "cluster_id"
    ]

    candidates = clustered_data[
        clustered_data["cluster_id"] == selected_cluster
    ].copy()

    candidates = candidates[
        candidates["name"] != selected_series
    ]

    candidates = candidates[
        candidates["vote_count"] >= 10
    ]

    display_columns = [
        "show_id",
        "genres",
        "network",
        "number_of_seasons",
        "vote_average"
    ]

    results = candidates.merge(
        clean_data[display_columns],
        on="show_id",
        how="left"
    )

    results = results.sort_values(
        "vote_quality",
        ascending=False
    )

    results = results.drop_duplicates(
        "name"
    )

    return results.head(
        number_of_results
    )


# ============================================================
# 6. CLASSEMENT PAR GENRE
# ============================================================

def get_top_by_genre(
    clean_data,
    selected_genre,
    number_of_results=5
):

    contains_genre = clean_data[
        "genres"
    ].str.contains(
        selected_genre,
        na=False
    )

    enough_votes = (
        clean_data["vote_count"] >= 10
    )

    results = clean_data[
        contains_genre & enough_votes
    ].copy()

    results = results.sort_values(
        "vote_quality",
        ascending=False
    )

    results = results.drop_duplicates(
        "name"
    )

    return results.head(
        number_of_results
    )


# ============================================================
# 7. RECHERCHE MULTI-CRITÈRES
# ============================================================

def search_with_filters(
    clean_data,
    selected_genre,
    selected_decade,
    minimum_rating,
    minimum_votes,
    season_range,
    number_of_results=5
):

    results = clean_data[
        clean_data["vote_count"] >= minimum_votes
    ].copy()

    if selected_genre != "Tous":

        results = results[
            results["genres"].str.contains(
                selected_genre,
                na=False
            )
        ]

    if selected_decade != "Toutes":

        selected_decade = float(
            selected_decade
        )

        results = results[
            results["decade"] == selected_decade
        ]

    results = results[
        results["vote_average"] >= minimum_rating
    ]

    if season_range is not None:

        minimum_seasons = season_range[0]
        maximum_seasons = season_range[1]

        results = results[
            (
                results["number_of_seasons"]
                >= minimum_seasons
            )
            & (
                results["number_of_seasons"]
                <= maximum_seasons
            )
        ]

    results = results.sort_values(
        "vote_quality",
        ascending=False
    )

    results = results.drop_duplicates(
        "name"
    )

    return results.head(
        number_of_results
    )


# ============================================================
# 8. AFFICHAGE D'UNE SÉRIE
# ============================================================

def display_series(series):

    name = series.get(
        "name",
        "Titre inconnu"
    )

    rating = series.get(
        "vote_average",
        0
    )

    vote_count = series.get(
        "vote_count",
        0
    )

    decade = series.get(
        "decade"
    )

    genres = series.get(
        "genres",
        "Non renseigné"
    )

    network = series.get(
        "network",
        "Chaîne inconnue"
    )

    seasons = series.get(
        "number_of_seasons"
    )

    if pd.isna(rating):
        rating = 0

    if pd.isna(vote_count):
        vote_count = 0

    if pd.isna(decade):
        decade_text = "Décennie inconnue"
    else:
        decade_text = str(
            int(decade)
        )

    if pd.isna(network) or network == "":
        network = "Chaîne inconnue"

    if pd.isna(seasons):
        seasons_text = "Nombre de saisons inconnu"
    else:
        seasons_text = (
            str(int(seasons))
            + " saison(s)"
        )

    st.subheader(name)

    st.write(
        "Note :",
        round(float(rating), 1),
        "/ 10"
    )

    st.write(
        "Nombre de votes :",
        int(vote_count)
    )

    st.write(
        "Décennie :",
        decade_text
    )

    st.write(
        "Genres :",
        genres
    )

    st.write(
        "Chaîne :",
        network
    )

    st.write(
        "Longueur :",
        seasons_text
    )

    st.divider()


# ============================================================
# 9. INTERFACE
# ============================================================

st.title("Recommandation de séries TV")

st.write(
    "Application utilisant les résultats "
    "du modèle K-Means."
)

recommendation_tab, genre_tab, filters_tab = st.tabs(
    [
        "Par série",
        "Par genre",
        "Recherche avancée"
    ]
)


# ============================================================
# 10. ONGLET RECOMMANDATION
# ============================================================

with recommendation_tab:

    st.header(
        "Recommandation à partir d'une série"
    )

    series_names = (
        df_clustered["name"]
        .dropna()
        .unique()
        .tolist()
    )

    series_names = sorted(
        series_names
    )

    selected_series = st.selectbox(
        "Choisissez une série",
        series_names
    )

    if selected_series:

        selected_information = df_clean[
            df_clean["name"] == selected_series
        ]

        if not selected_information.empty:

            st.write("Série sélectionnée :")

            display_series(
                selected_information.iloc[0]
            )

        recommendations = get_recommendations(
            df_clean,
            df_clustered,
            selected_series
        )

        st.header(
            "Séries du même cluster"
        )

        if recommendations.empty:

            st.warning(
                "Aucune recommandation trouvée."
            )

        else:

            for index, series in recommendations.iterrows():

                display_series(
                    series
                )


# ============================================================
# 11. ONGLET GENRE
# ============================================================

with genre_tab:

    st.header(
        "Meilleures séries par genre"
    )

    selected_genre = st.selectbox(
        "Choisissez un genre",
        all_genres
    )

    if selected_genre:

        genre_results = get_top_by_genre(
            df_clean,
            selected_genre
        )

        if genre_results.empty:

            st.warning(
                "Aucune série trouvée."
            )

        else:

            for index, series in genre_results.iterrows():

                display_series(
                    series
                )


# ============================================================
# 12. ONGLET RECHERCHE AVANCÉE
# ============================================================

with filters_tab:

    st.header(
        "Recherche avec plusieurs critères"
    )

    selected_genre_filter = st.selectbox(
        "Genre",
        ["Tous"] + all_genres
    )

    decade_values = (
        df_clean["decade"]
        .dropna()
        .unique()
        .tolist()
    )

    decade_values = [
        str(int(decade))
        for decade in decade_values
    ]

    decade_values = sorted(
        decade_values
    )

    selected_decade = st.selectbox(
        "Décennie",
        ["Toutes"] + decade_values
    )

    minimum_rating = st.slider(
        "Note minimum",
        min_value=0.0,
        max_value=10.0,
        value=6.0,
        step=0.5
    )

    minimum_votes = st.slider(
        "Nombre minimum de votes",
        min_value=0,
        max_value=500,
        value=10,
        step=10
    )

    length_options = {
        "Toutes": None,
        "Une saison": (1, 1),
        "Deux ou trois saisons": (2, 3),
        "Quatre à sept saisons": (4, 7),
        "Plus de sept saisons": (8, 999)
    }

    selected_length = st.selectbox(
        "Longueur de la série",
        list(length_options.keys())
    )

    if st.button("Rechercher"):

        search_results = search_with_filters(
            df_clean,
            selected_genre_filter,
            selected_decade,
            minimum_rating,
            minimum_votes,
            length_options[selected_length]
        )

        if search_results.empty:

            st.warning(
                "Aucune série ne correspond aux critères."
            )

        else:

            for index, series in search_results.iterrows():

                display_series(
                    series
                )