```python
"""
Data and Model for Soccer Analytics
Bachelor's Thesis Project

Reconstructed Python implementation of the FIFA 19 exploratory
data analysis, preprocessing, PCA, and K-Means clustering.

Dataset:
FIFA 15-21.xlsx, sheet: FIFA 19

Selected clustering features:
Pace, Shooting, Passing, Dribbling, Defending, Physicality
"""

# ============================================================
# A.1 IMPORTING PYTHON LIBRARIES
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# ============================================================
# A.2 LOADING THE FIFA 19 DATASET
# ============================================================

DATA_PATH = "FIFA 15-21.xlsx"
SHEET_NAME = "FIFA 19"

OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)

print("\nFirst five records:")
print(df.head())


# ============================================================
# A.3 DATA PREPROCESSING
# ============================================================

# Selected performance attributes
features = [
    "pace",
    "shooting",
    "passing",
    "dribbling",
    "defending",
    "physic"
]

# Check that all required columns are available
missing_columns = [col for col in features if col not in df.columns]

if missing_columns:
    raise ValueError(
        f"Missing required dataset columns: {missing_columns}"
    )

# Exclude goalkeepers using their player-position information
outfield_players = df[
    ~df["player_positions"].fillna("").str.contains(
        r"(^|,\s*)GK(\s*,|$)",
        case=False,
        regex=True
    )
].copy()

# Retain records with complete selected attributes
analysis_data = outfield_players.dropna(
    subset=features
).copy()

print("\nPreprocessing completed.")
print("Original dataset:", len(df))
print("Outfield players:", len(outfield_players))
print("Complete records used for analysis:", len(analysis_data))

# Create the feature matrix
X = analysis_data[features].astype(float)


# ============================================================
# A.4 EXPLORATORY DATA ANALYSIS
# ============================================================

sns.set_theme(style="whitegrid")

# Age distribution
plt.figure(figsize=(9, 5))
sns.histplot(df["age"].dropna(), bins=25, kde=True)
plt.title("Age Distribution of FIFA 19 Players")
plt.xlabel("Age")
plt.ylabel("Number of Players")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "Figure_4_2_Age_Distribution.png"),
    dpi=300
)
plt.close()

# Overall rating distribution
plt.figure(figsize=(9, 5))
sns.histplot(df["overall"].dropna(), bins=25, kde=True)
plt.title("Overall Rating Distribution")
plt.xlabel("Overall Rating")
plt.ylabel("Number of Players")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "Figure_4_3_Overall_Rating_Distribution.png"),
    dpi=300
)
plt.close()

# Player position distribution
positions = df["team_position"].dropna().value_counts()

plt.figure(figsize=(11, 6))
sns.barplot(x=positions.index, y=positions.values)
plt.title("Distribution of Player Positions")
plt.xlabel("Player Position")
plt.ylabel("Number of Players")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "Figure_4_4_Player_Position_Distribution.png"),
    dpi=300
)
plt.close()

# Top 10 nationalities
nationalities = df["nationality"].value_counts().head(10)

plt.figure(figsize=(10, 6))
sns.barplot(
    x=nationalities.values,
    y=nationalities.index
)
plt.title("Top 10 Nationalities in FIFA 19")
plt.xlabel("Number of Players")
plt.ylabel("Nationality")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "Figure_4_5_Top_10_Nationalities.png"),
    dpi=300
)
plt.close()

# Correlation heatmap
plt.figure(figsize=(9, 7))
correlation = analysis_data[features].corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap of Selected Attributes")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "Figure_4_6_Correlation_Heatmap.png"),
    dpi=300
)
plt.close()

# Boxplots of selected attributes
plt.figure(figsize=(11, 6))

sns.boxplot(data=analysis_data[features])

plt.title("Boxplots of Selected Player Attributes")
plt.xlabel("Attributes")
plt.ylabel("Attribute Rating")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "Figure_4_7_Boxplots_Selected_Attributes.png"),
    dpi=300
)
plt.close()

print("\nExploratory Data Analysis completed.")


# ============================================================
# A.5 FEATURE STANDARDIZATION
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nFeature standardization completed.")


# ============================================================
# A.6 ELBOW METHOD
# ============================================================

inertia_values = []

for k in range(1, 11):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    inertia_values.append(model.inertia_)

plt.figure(figsize=(9, 5))

plt.plot(
    range(1, 11),
    inertia_values,
    marker="o"
)

plt.title("Elbow Method for Optimal Number of Clusters")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Within-Cluster Sum of Squares (Inertia)")
plt.xticks(range(1, 11))
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "Figure_5_1_Elbow_Method.png"),
    dpi=300
)

plt.close()

print("\nElbow Method completed.")


# ============================================================
# A.7 K-MEANS CLUSTERING
# ============================================================

# Three clusters are used, following the thesis analysis.
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

cluster_labels = kmeans.fit_predict(X_scaled)

analysis_data["Cluster"] = cluster_labels

print("\nK-Means clustering completed.")

print("\nCluster sizes:")
print(analysis_data["Cluster"].value_counts().sort_index())


# ============================================================
# A.8 CLUSTER CHARACTERISTICS
# ============================================================

cluster_summary = (
    analysis_data.groupby("Cluster")[features]
    .mean()
    .round(2)
)

print("\nCluster characteristics:")
print(cluster_summary)

cluster_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "Table_8_1_KMeans_Cluster_Characteristics.csv"
    )
)


# ============================================================
# A.9 PRINCIPAL COMPONENT ANALYSIS (PCA)
# ============================================================

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

print("\nExplained variance by first two components:")
print(pca.explained_variance_ratio_)

print(
    "Total explained variance:",
    round(sum(pca.explained_variance_ratio_) * 100, 2),
    "%"
)

# PCA projection
plt.figure(figsize=(10, 7))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=cluster_labels,
    cmap="viridis",
    s=12,
    alpha=0.6
)

plt.title("PCA Projection of FIFA 19 Player Clusters")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "Figure_8_7_PCA_Projection.png"),
    dpi=300
)

plt.close()


# ============================================================
# A.10 K-MEANS CLUSTER VISUALIZATION
# ============================================================

plt.figure(figsize=(10, 7))

plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=cluster_labels,
    cmap="viridis",
    s=12,
    alpha=0.6
)

plt.title("K-Means Clustering of FIFA 19 Players")
plt.xlabel("Standardized Pace")
plt.ylabel("Standardized Shooting")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "Figure_8_8_KMeans_Clustering.png"),
    dpi=300
)

plt.close()


# ============================================================
# A.11 FINAL OUTPUT
# ============================================================

print("\n======================================")
print("SOCCER ANALYTICS ANALYSIS COMPLETED")
print("======================================")

print("\nNumber of players analysed:", len(analysis_data))

print("\nFinal cluster sizes:")
print(analysis_data["Cluster"].value_counts().sort_index())

print("\nGenerated files are saved in:", OUTPUT_DIR)
```
