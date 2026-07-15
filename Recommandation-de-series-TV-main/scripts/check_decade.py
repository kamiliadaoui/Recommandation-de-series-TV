from pyspark.sql import SparkSession
from pyspark.sql.functions import min, max

spark = SparkSession.builder.appName("Check decade").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

df = spark.read.parquet(
    "hdfs://localhost:9000/user/user/data/shows_clean"
)

df.select("decade").distinct().orderBy("decade").show(100)
df.agg(
    min("decade").alias("minimum"),
    max("decade").alias("maximum")
).show()

spark.stop()
