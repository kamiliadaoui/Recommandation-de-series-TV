import pandas as pd


def get_recommendations(df_clean, selected_series, n=5):
    """Recommande n séries similaires à la série sélectionnée"""
    serie_info = df_clean[df_clean["name"] == selected_series]
    if serie_info.empty:
        return pd.DataFrame()

    serie_row = serie_info.iloc[0]
    genres_serie = [g.strip() for g in str(serie_row["genres"]).split(",")
                   if g.strip() and g.strip() != "Non renseigné"]
    decade_serie = serie_row["decade"]

    results = df_clean[
        (df_clean["name"] != selected_series) &
        (df_clean["vote_count"] >= 10)
    ].copy()

    def compute_similarity(row):
        score = 0
        row_genres = [g.strip() for g in str(row["genres"]).split(",")]
        common_genres = len(set(genres_serie) & set(row_genres))
        score += common_genres * 3
        if pd.notna(row["decade"]) and pd.notna(decade_serie):
            diff_decade = abs(row["decade"] - decade_serie)
            if diff_decade == 0:
                score += 2
            elif diff_decade <= 10:
                score += 1
        score += row["vote_quality"] * 0.5
        return score

    results["similarity_score"] = results.apply(compute_similarity, axis=1)

    return results.sort_values("similarity_score", ascending=False) \
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