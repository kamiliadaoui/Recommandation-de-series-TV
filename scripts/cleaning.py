import os
os.environ["JAVA_HOME"] = r"C:\Users\kamil\AppData\Local\Programs\ECLIPS~1\JDK-17~1.10-"

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, isnan, isnull

#On demarre spark
spark = SparkSession.builder \
    .appName("TV Series - Cleaning") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
 
#LIRE LES FICHIERS DEPUIS HDFS
HDFS_PATH = "hdfs://localhost:9000/user/kamil/data"

shows = spark.read.csv(f"{HDFS_PATH}/shows.csv",header=True,inferSchema=True,quote='"',
    escape='"',multiLine=True)
show_votes = spark.read.csv(f"{HDFS_PATH}/show_votes.csv", header=True, inferSchema=True)
genres = spark.read.csv(f"{HDFS_PATH}/genres.csv", header=True, inferSchema=True)
genre_types = spark.read.csv(f"{HDFS_PATH}/genre_types.csv", header=True, inferSchema=True)
networks = spark.read.csv(f"{HDFS_PATH}/networks.csv", header=True, inferSchema=True)
network_types = spark.read.csv(f"{HDFS_PATH}/network_types.csv", header=True, inferSchema=True)
status = spark.read.csv(f"{HDFS_PATH}/status.csv", header=True, inferSchema=True)
types = spark.read.csv(f"{HDFS_PATH}/types.csv", header=True, inferSchema=True)
air_dates = spark.read.csv(f"{HDFS_PATH}/air_dates.csv", header=True, inferSchema=True)

#on corrige les types 
shows = shows.withColumn("show_id", col("show_id").cast("integer"))

#On verifie les valeurs nulles
print("VALEURS NULLES DANS SHOWS ")
shows.select([
    count(when(isnull(c), c)).alias(c) 
    for c in shows.columns
]).show()

#On supprimes les doublons 
print(f"Avant déduplication : {shows.count()} lignes")
shows = shows.dropDuplicates(["show_id"])
print(f"Après déduplication : {shows.count()} lignes")

#On supprime les lignes avec des valeurs nulles dans les colonnes critiques
shows = shows.dropna(subset=["show_id", "name", "popularity"])
# On garde uniquement la date de première diffusion
first_air_dates = air_dates.filter(col("is_first") == 1) \
    .select("show_id", "date") \
    .withColumnRenamed("date", "first_air_date")

# On déduplique au cas où il y aurait plusieurs lignes is_first=1 par série
first_air_dates = first_air_dates.dropDuplicates(["show_id"])

#Jointures
# shows + votes
df = shows.join(show_votes, on="show_id", how="left")

# shows + genres + genre_types
genres_named = genres.join(genre_types, on="genre_type_id", how="left")
df = df.join(genres_named, on="show_id", how="left")

# shows + networks + network_types
networks_named = networks.join(network_types, on="network_type_id", how="left")
df = df.join(networks_named, on="show_id", how="left")

# shows + status
df = df.join(status, on="status_id", how="left")

# shows + types
df = df.join(types, on="type_id", how="left")

df = df.join(first_air_dates, on="show_id", how="left")
#On garde uniquement les colonnes nécessaires
df_final = df.select(
    col("show_id"),
    col("name"),
    col("popularity"),
    col("number_of_seasons"),
    col("number_of_episodes"),
    col("eposide_run_time"),
    col("vote_average"),
    col("vote_count"),
    col("genre_name"),
    col("network_name"),
    col("status_name"),
    col("type_name"),
    col("first_air_date")
)
df_final = df_final.withColumnRenamed("eposide_run_time", "episode_run_time")

#STATS DESCRIPTIVES

print("\n=== APERCU DU DATAFRAME FINAL ===")
df_final.show(10)

print("\n=== SCHEMA FINAL ===")
df_final.printSchema()

print(f"\nNombre total de lignes : {df_final.count()}")

print("\n=== STATS DESCRIPTIVES (colonnes numériques) ===")
df_final.select(
    "popularity", 
    "number_of_seasons", 
    "number_of_episodes",
    "episode_run_time",
    "vote_average", 
    "vote_count"
).describe().show()

print("\n=== TOP 10 SERIES LES PLUS POPULAIRES ===")
df_final.orderBy(col("popularity").desc()).show(10)

print("\n=== DISTRIBUTION DES GENRES ===")
df_final.groupBy("genre_name") \
    .count() \
    .orderBy(col("count").desc()) \
    .show(20)

print("\n=== DISTRIBUTION DES STATUTS ===")
df_final.groupBy("status_name") \
    .count() \
    .orderBy(col("count").desc()) \
    .show()

df_final.write.mode("overwrite") \
    .parquet(f"{HDFS_PATH}/shows_clean")

print("\n Données nettoyées etsauvegardées sur HDFS !")

spark.stop()
