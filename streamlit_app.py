import os
os.environ["JAVA_HOME"] = r"C:\Program Files\Eclipse Adoptium\jdk-17.0.19.10-hotspot"

import streamlit as st
import plotly.express as px
from pyspark.sql import SparkSession

st.set_page_config(page_title="Recommandation TV Shows", layout="wide")
st.title("Recommandation de séries TV")
st.caption("Dataset Kaggle — 160 000+ séries | PySpark + HDFS")

@st.cache_resource
def get_spark():
    spark = SparkSession.builder \
        .appName("TV Series - Reco") \
        .master("local[*]") \
        .config("spark.sql.shuffle.partitions", "4") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark

@st.cache_data
def load_data():
    spark = get_spark()
    df = spark.read.parquet("hdfs://localhost:9000/tvshows/input/shows_clean").toPandas()
    return df

with st.spinner("Chargement depuis HDFS..."):
    df = load_data()

df_unique = df.drop_duplicates(subset=["show_id"])
c1, c2, c3 = st.columns(3)
c1.metric("Séries", f"{len(df_unique):,}")
c2.metric("Note moyenne", f"{df_unique['vote_average'].mean():.1f} / 10")
c3.metric("Genres", df["genre_name"].nunique())

st.divider()

st.header("Trouve ta prochaine série")

col_genre, col_note, col_saisons = st.columns(3)

with col_genre:
    genres = sorted(df["genre_name"].dropna().unique().tolist())
    genre = st.selectbox("Genre", ["Tous"] + genres)

with col_note:
    note_min = st.slider("Note minimum", 0.0, 10.0, 6.0, 0.5)

with col_saisons:
    saisons_min = st.number_input("Saisons minimum", min_value=0, max_value=50, value=1)
result = df_unique.copy()

if genre != "Tous":
    ids_genre = df[df["genre_name"] == genre]["show_id"].unique()
    result = result[result["show_id"].isin(ids_genre)]

result = result[
    (result["vote_average"] >= note_min) &
    (result["number_of_seasons"] >= saisons_min) &
    (result["vote_count"] > 10)
]

result = result.sort_values("popularity", ascending=False).head(20)

st.divider()

if len(result) > 0:
    st.subheader(f"🎬 {len(result)} séries trouvées")

    st.dataframe(
        result[["name", "popularity", "vote_average", "vote_count", "number_of_seasons"]].rename(columns={
            "name": "Série",
            "popularity": "Popularité",
            "vote_average": "Note",
            "vote_count": "Votes",
            "number_of_seasons": "Saisons"
        }),
        use_container_width=True,
        hide_index=True
    )

    fig = px.bar(
        result.head(10).sort_values("popularity"),
        x="popularity",
        y="name",
        orientation="h",
        color="vote_average",
        color_continuous_scale="RdYlGn",
        text="vote_average"
    )
    fig.update_traces(texttemplate="%{text:.1f} ⭐", textposition="outside")
    fig.update_layout(
        height=400,
        yaxis_title="",
        xaxis_title="Popularité",
        coloraxis_colorbar_title="Note"
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Aucune série trouvée, ajuste les filtres.")

st.divider()
st.caption("Projet Hadoop ")
