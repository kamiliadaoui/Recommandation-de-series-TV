import pandas as pd
import numpy as np
from numpy.linalg import norm


def cosine_distance(a, b):
    """Calcule la distance cosinus entre deux vecteurs"""
    if norm(a) == 0 or norm(b) == 0:
        return 1.0
    return 1 - np.dot(a, b) / (norm(a) * norm(b))


def get_recommendations(df_clean, df_clustered, selected_series, n=5):
    """Recommande n séries du même cluster triées par distance cosinus"""
    serie_info = df_clustered[df_clustered["name"] == selected_series]
    if serie_info.empty:
        return pd.DataFrame()

    cluster_id = serie_info.iloc[0]["cluster_id"]

    # Features numériques + genres disponibles dans shows_clustered
    excluded = ["show_id", "name", "cluster_id"]
    feature_cols = [c for c in df_clustered.columns if c not in excluded]

    # Vecteur de la série choisie
    serie_vector = serie_info[feature_cols].fillna(0).values[0].astype(float)

    # Séries du même cluster
    same_cluster = df_clustered[
        (df_clustered["cluster_id"] == cluster_id) &
        (df_clustered["name"] != selected_series) &
        (df_clustered["vote_count"] >= 10)
    ].copy()

    # Calcul de la distance cosinus
    same_cluster["distance"] = same_cluster[feature_cols].fillna(0).apply(
        lambda row: cosine_distance(
            row.values.astype(float), serie_vector
        ),
        axis=1
    )

    # Joindre avec df_clean pour avoir genres, network, etc.
    results = same_cluster.merge(
        df_clean[["show_id", "genres", "network",
                  "number_of_seasons", "vote_average", "decade"]],
        on="show_id", how="left",
        suffixes=("", "_clean")
    )

    # Trier par distance cosinus (plus petit = plus similaire)
    return results.sort_values("distance", ascending=True) \
                  .drop_duplicates("name") \
                  .head(n)


def get_top_by_genre(df_clean, genre, n=5):
    """Top 5 séries d'un genre donné"""
    return df_clean[
        df_clean["genres"].str.contains(genre, na=False) &
        (df_clean["vote_count"] >= 10)
    ].sort_values("vote_quality", ascending=False) \
     .drop_duplicates("name") \
     .head(n)


def search_multifiltres(df_clean, genre=None, decade=None,
                        note_min=0, votes_min=10,
                        length_range=None, n=5):
    """Recherche multi-critères combinés"""
    results = df_clean[df_clean["vote_count"] >= votes_min].copy()

    if genre and genre != "Tous":
        results = results[results["genres"].str.contains(genre, na=False)]
    if decade and decade != "Toutes":
        results = results[results["decade"] == float(decade)]
    if note_min:
        results = results[results["vote_average"] >= note_min]
    if length_range:
        results = results[
            (results["number_of_seasons"] >= length_range[0]) &
            (results["number_of_seasons"] <= length_range[1])
        ]

    return results.sort_values("vote_quality", ascending=False) \
                  .drop_duplicates("name") \
                  .head(n)