import os
os.environ["JAVA_HOME"] = r"C:\Users\kamil\AppData\Local\Programs\ECLIPS~1\JDK-17~1.10-"

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator


spark = SparkSession.builder \
    .appName("TV Series - Clustering") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

HDFS_PATH = "hdfs://localhost:9000/user/kamil/data"

df = spark.read.parquet(f"{HDFS_PATH}/shows_clean")

print(f"Nombre de lignes de départ : {df.count()}")


shows_genres = df.groupBy(
    "show_id", "name", "popularity", "vote_average",
    "vote_count", "number_of_seasons",
    "number_of_episodes", "eposide_run_time"
).pivot("genre_name").agg(lit(1)).na.fill(0)

print(f"Nombre de séries après agrégation : {shows_genres.count()}")
print("\n=== APERCU APRES ONE-HOT ENCODING ===")
shows_genres.show(5)

print("\n=== COLONNES DISPONIBLES ===")
print(shows_genres.columns)

# 4. NETTOYAGE DES VALEURS ABERRANTES (outliers)
# On filtre les durées d'épisode aberrantes (ex: 6032 minutes)
shows_genres = shows_genres.filter(
    (col("episode_run_time") < 180) & (col("episode_run_time") >= 0)
)

# On filtre les séries sans aucun vote (vote_count = 0 fausse le clustering)
shows_genres = shows_genres.filter(col("vote_count") > 0)

print(f"\nNombre de séries après filtrage des outliers : {shows_genres.count()}")

# 5. PREPARER LES FEATURES POUR KMEANS
# Colonnes numériques de base
numeric_cols = [
    "popularity", "vote_average", "vote_count",
    "number_of_seasons", "number_of_episodes", "eposide_run_time"
]

# Colonnes de genres (tout sauf les colonnes connues)
excluded_cols = ["show_id", "name"] + numeric_cols
genre_cols = [c for c in shows_genres.columns if c not in excluded_cols and c != "null"]

print(f"\nColonnes de genres détectées : {genre_cols}")

feature_cols = numeric_cols + genre_cols

# 6. VECTOR ASSEMBLER - combine toutes les features en un vecteur
assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features_raw",
    handleInvalid="skip"
)

data_assembled = assembler.transform(shows_genres)

# ============================================================
# 7. NORMALISATION avec StandardScaler
# ============================================================
scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features",
    withStd=True,
    withMean=True
)

scaler_model = scaler.fit(data_assembled)
data_scaled = scaler_model.transform(data_assembled)

print("\n=== DONNEES PRETES POUR KMEANS ===")
data_scaled.select("show_id", "name", "features").show(5, truncate=False)

# 8. METHODE DU COUDE - trouver le meilleur K
print("\n=== METHODE DU COUDE (test de plusieurs K) ===")
costs = []
K_range = [3, 5, 8, 10, 12, 15, 20]

for k in K_range:
    kmeans_test = KMeans(featuresCol="features", k=k, seed=42)
    model_test = kmeans_test.fit(data_scaled)
    cost = model_test.summary.trainingCost
    costs.append(cost)
    print(f"K={k} -> Coût (inertie) = {cost:.2f}")

