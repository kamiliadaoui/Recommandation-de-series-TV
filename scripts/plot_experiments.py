import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

Path("outputs").mkdir(exist_ok=True)

df = pd.read_csv("experiments/results.csv")
summary = df.groupby("k", as_index=False).agg(
    silhouette=("silhouette", "mean"),
    cout=("cout", "mean")
)

plt.figure()
plt.plot(summary["k"], summary["silhouette"], marker="o")
plt.xlabel("Nombre de clusters K")
plt.ylabel("Silhouette moyenne")
plt.title("Score de silhouette selon K")
plt.savefig("outputs/silhouette_par_k.png", bbox_inches="tight")
plt.close()

plt.figure()
plt.plot(summary["k"], summary["cout"], marker="o")
plt.xlabel("Nombre de clusters K")
plt.ylabel("Coût moyen")
plt.title("Coût K-Means selon K")
plt.savefig("outputs/cout_par_k.png", bbox_inches="tight")
plt.close()

print(summary)
print("Graphes sauvegardés dans outputs/")
