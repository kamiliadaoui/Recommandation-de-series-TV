import pandas as pd


def get_recommendations(df_clean, df_clustered, selected_series, n=5):
    """Recommande n séries du même cluster KMeans"""
    serie_info = df_clustered[df_clustered["name"] == selected_series]
    if serie_info.empty:
        return pd.DataFrame()

    cluster_id = serie_info.iloc[0]["cluster_id"]

    # Séries du même cluster KMeans
    same_cluster = df_clustered[
        (df_clustered["cluster_id"] == cluster_id) &
        (df_clustered["name"] != selected_series) &
        (df_clustered["vote_count"] >= 10)
    ].copy()

    # Joindre avec df_clean pour avoir genres, network, number_of_seasons
    results = same_cluster.merge(
        df_clean[["show_id", "genres", "network",
                  "number_of_seasons", "vote_average", "decade"]],
        on="show_id", how="left",
        suffixes=("", "_clean")
    )

    return results.sort_values("vote_quality", ascending=False) \
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