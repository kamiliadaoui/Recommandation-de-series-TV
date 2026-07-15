from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, floor, when, year


# ============================================================
# 1. DÉMARRAGE DE SPARK
# ============================================================

spark = (
    SparkSession.builder
    .appName("TV Series - Cleaning")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

HDFS_PATH = "hdfs://localhost:9000/user/user/data"


# ============================================================
# 2. LECTURE DES FICHIERS CSV DEPUIS HDFS
# ============================================================

shows = spark.read.csv(
    f"{HDFS_PATH}/shows.csv",
    header=True,
    inferSchema=True,
    quote='"',
    escape='"',
    multiLine=True
)

show_votes = spark.read.csv(
    f"{HDFS_PATH}/show_votes.csv",
    header=True,
    inferSchema=True
)

genres = spark.read.csv(
    f"{HDFS_PATH}/genres.csv",
    header=True,
    inferSchema=True
)

genre_types = spark.read.csv(
    f"{HDFS_PATH}/genre_types.csv",
    header=True,
    inferSchema=True
)

networks = spark.read.csv(
    f"{HDFS_PATH}/networks.csv",
    header=True,
    inferSchema=True
)

network_types = spark.read.csv(
    f"{HDFS_PATH}/network_types.csv",
    header=True,
    inferSchema=True
)

status = spark.read.csv(
    f"{HDFS_PATH}/status.csv",
    header=True,
    inferSchema=True
)

types = spark.read.csv(
    f"{HDFS_PATH}/types.csv",
    header=True,
    inferSchema=True
)

air_dates = spark.read.csv(
    f"{HDFS_PATH}/air_dates.csv",
    header=True,
    inferSchema=True
)


# ============================================================
# 3. NETTOYAGE DE LA TABLE PRINCIPALE
# ============================================================

# Conversion de show_id en entier
shows = shows.withColumn(
    "show_id",
    col("show_id").cast("integer")
)

# Comptage des valeurs nulles
null_counts = []

for column_name in shows.columns:
    null_count = count(
        when(col(column_name).isNull(), 1)
    ).alias(column_name)

    null_counts.append(null_count)

print("\n=== VALEURS NULLES DANS SHOWS ===")
shows.select(null_counts).show()

# Suppression des doublons
number_before = shows.count()

shows = shows.dropDuplicates(["show_id"])

number_after = shows.count()

print("Avant déduplication :", number_before)
print("Après déduplication :", number_after)

# Suppression des lignes inutilisables
shows = shows.dropna(
    subset=["show_id", "name", "popularity"]
)


# ============================================================
# 4. PRÉPARATION DES TABLES SECONDAIRES
# ============================================================

# Conservation de la première date de diffusion
first_air_dates = air_dates.filter(
    col("is_first") == 1
)

first_air_dates = first_air_dates.select(
    "show_id",
    "date"
)

first_air_dates = first_air_dates.withColumnRenamed(
    "date",
    "first_air_date"
)

first_air_dates = first_air_dates.dropDuplicates(
    ["show_id"]
)

# Ajout du nom des genres
genres_named = genres.join(
    genre_types,
    on="genre_type_id",
    how="left"
)

# Ajout du nom des chaînes
networks_named = networks.join(
    network_types,
    on="network_type_id",
    how="left"
)


# ============================================================
# 5. JOINTURES
# ============================================================

df = shows.join(
    show_votes,
    on="show_id",
    how="left"
)

df = df.join(
    genres_named,
    on="show_id",
    how="left"
)

df = df.join(
    networks_named,
    on="show_id",
    how="left"
)

df = df.join(
    status,
    on="status_id",
    how="left"
)

df = df.join(
    types,
    on="type_id",
    how="left"
)

df = df.join(
    first_air_dates,
    on="show_id",
    how="left"
)


# ============================================================
# 6. SÉLECTION DES COLONNES UTILES
# ============================================================

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
    col("first_air_date"),
    col("adult")
)

# Correction du nom mal orthographié dans le dataset
df_final = df_final.withColumnRenamed(
    "eposide_run_time",
    "episode_run_time"
)


# ============================================================
# 7. CRÉATION DE LA DÉCENNIE
# ============================================================

release_year = year(
    col("first_air_date")
)

decade = floor(
    release_year / 10
) * 10

df_final = df_final.withColumn(
    "decade",
    decade.cast("integer")
)

# Conservation des décennies plausibles
valid_decade = (
    col("decade").isNull()
    | (
        (col("decade") >= 1900)
        & (col("decade") <= 2030)
    )
)

df_final = df_final.filter(valid_decade)


# ============================================================
# 8. STATISTIQUES DESCRIPTIVES
# ============================================================

print("\n=== APERÇU DU DATAFRAME FINAL ===")
df_final.show(10)

print("\n=== SCHÉMA FINAL ===")
df_final.printSchema()

print("\nNombre total de lignes :", df_final.count())

print("\n=== STATISTIQUES NUMÉRIQUES ===")

numeric_columns = [
    "popularity",
    "number_of_seasons",
    "number_of_episodes",
    "episode_run_time",
    "vote_average",
    "vote_count"
]

df_final.select(numeric_columns).describe().show()

# Une série peut apparaître plusieurs fois à cause des genres et chaînes
unique_shows = df_final.dropDuplicates(["show_id"])

print("\n=== TOP 10 DES SÉRIES LES PLUS POPULAIRES ===")

top_popular = unique_shows.orderBy(
    col("popularity").desc()
)

top_popular.select(
    "name",
    "popularity",
    "vote_average",
    "vote_count"
).show(10, truncate=False)

print("\n=== DISTRIBUTION DES GENRES ===")

genre_distribution = df_final.select(
    "show_id",
    "genre_name"
)

genre_distribution = genre_distribution.dropDuplicates()

genre_distribution = genre_distribution.groupBy(
    "genre_name"
).count()

genre_distribution = genre_distribution.orderBy(
    col("count").desc()
)

genre_distribution.show(20)

print("\n=== DISTRIBUTION DES STATUTS ===")

status_distribution = df_final.select(
    "show_id",
    "status_name"
)

status_distribution = status_distribution.dropDuplicates()

status_distribution = status_distribution.groupBy(
    "status_name"
).count()

status_distribution = status_distribution.orderBy(
    col("count").desc()
)

status_distribution.show()

print("\n=== DISTRIBUTION DES DÉCENNIES ===")

decade_distribution = df_final.select(
    "show_id",
    "decade"
)

decade_distribution = decade_distribution.dropDuplicates()

decade_distribution = decade_distribution.groupBy(
    "decade"
).count()

decade_distribution = decade_distribution.orderBy(
    col("decade").asc()
)

decade_distribution.show()


# ============================================================
# 9. SAUVEGARDE SUR HDFS
# ============================================================

output_path = f"{HDFS_PATH}/shows_clean"

df_final.write.mode("overwrite").parquet(
    output_path
)

print("\nDonnées nettoyées et sauvegardées dans :")
print(output_path)

spark.stop()