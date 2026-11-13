# TP4 – Workflow Machine Learning complet : de scikit-learn à Spark

> **Consigne** : (1) choisir un exemple Kaggle quelconque, sans tenir compte de la taille des données ;
> (2) appliquer le workflow `Data → EDA → Prétraitement → Transformation → ML → Évaluation → Déploiement` ;
> (3) refaire l'exercice en Big Data (Hadoop/Spark) ; (4) publier sur GitHub.

## Jeux de données

| Jeu de données | Taille | Rôle | Cible à prédire |
|---|---|---|---|
| [World Data Centers](https://www.kaggle.com/datasets/rockyt0/world-data-centers) (OpenStreetMap) | 4 802 sites, 1 CSV | Exemple « classique » (partie 1) puis refait en Spark (partie 3) | Type de site : `colocation`, `hyperscale`, `telecom_carrier` |
| [Full Body Checkup Lab](https://www.kaggle.com/datasets/opendataforresearch/full-body-checkup-lab-dataset) | 250 164 bilans × 109 analyses, 12 CSV (~430 Mo) | Cas Big Data (partie 2) | Pré-diabète/diabète : `HbA1c ≥ 5,7 %` |

## Structure du notebook

| Partie | Outils | Contenu |
|---|---|---|
| **1. World Data Centers** | pandas, scikit-learn | EDA, indicateurs de présence, `Pipeline` + `ColumnTransformer`, régression logistique vs forêt aléatoire (CV 5 plis), sauvegarde `joblib` |
| **2. Full Body Checkup** | PySpark (DataFrame, SQL, MLlib), Parquet | Lecture parallèle des 12 CSV, jointure avec le profil patient, EDA en Spark SQL, écriture Parquet, `Pipeline` MLlib (`Imputer` → `StringIndexer` → `OneHotEncoder` → `VectorAssembler` → `StandardScaler`), 3 modèles, sauvegarde du `PipelineModel` |
| **3. World Data Centers en Spark** | PySpark MLlib | Même workflow que la partie 1 + agrégation SQL par pays jointe aux totaux officiels (`reported_totals.csv`) |

## Résultats

| Partie | Meilleur modèle | Score |
|---|---|---|
| 1 – Data centers (sklearn) | Forêt aléatoire | F1 macro CV **0,755** ; précision test **0,81** (classe `telecom_carrier` plus faible : F1 0,45) |
| 2 – Checkup (Spark) | Régression logistique | AUC test **0,687** (GBT 0,685, forêt 0,655) ; accuracy 0,737 |

Le signal de la partie 2 est **modeste** : sans les mesures de glycémie, les autres analyses expliquent mal l'HbA1c
(prévalence positive ≈ 27 %, un classifieur qui répond toujours « non » atteint déjà ≈ 73 % d'accuracy). Ces scores sont rapportés tels quels ;
les données semblent par ailleurs (semi-)synthétiques et n'ont aucune valeur clinique.

## Choix méthodologiques

- **Fuite de données évitée** : `fasting_glucose`, `fbs`, `plbs`, `pp_glucose`, `eag` et `hba1c` sont exclus des variables explicatives ;
  les noms de sites (« AWS », « Equinix »…) ne sont pas utilisés pour prédire le type de data center.
- **Split par patient** (hash de `MedID`) : un patient apparaît dans les 12 mois, un split par ligne le placerait des deux côtés.
- **Déséquilibre de classes** : `class_weight="balanced"` et F1 macro (partie 1), AUC (partie 2).
- **Imputation dans le pipeline** (après le split) pour ne rien apprendre du jeu de test.
- **Parquet** (format colonnaire de l'écosystème Hadoop) pour le jeu nettoyé ; en cluster, remplacer `local[*]` par YARN et les chemins locaux par `hdfs://…`.

## Déploiement

| Script | Rôle | Exemple |
|---|---|---|
| `predict_datacenter.py` | Prédit le type d'un data center avec le pipeline sklearn (`models/datacenter_rf.joblib`) | `python predict_datacenter.py 60000 45.5 -122.6 "North America"` |
| `score_checkup.py` | Score en batch un fichier Parquet avec le `PipelineModel` Spark (`models/checkup_spark_model`) | `spark-submit score_checkup.py checkup_clean.parquet scores.parquet` |

Les modèles sont générés en exécutant le notebook (dossier `models/`, non versionné).

## Reproduire

1. Télécharger les deux jeux de données Kaggle et extraire les CSV dans `world_data/` et `checkup_data/` (non versionnés, cf. `.gitignore`).
2. Installer les dépendances : `pip install pyspark==3.5.3 pandas scikit-learn matplotlib joblib jupyter` (Java 17 requis par Spark).
3. **Sous Windows**, Spark a besoin de `winutils.exe` et `hadoop.dll` (Hadoop 3.3.x) dans `C:\hadoop\bin`, et des variables
   `HADOOP_HOME=C:\hadoop`, `PATH` incluant `C:\hadoop\bin`, `PYSPARK_PYTHON` pointant vers l'interpréteur Python utilisé.
4. Lancer `jupyter nbconvert --to notebook --execute --inplace TP_kaggle_ml_spark.ipynb` (≈ 5 min, 6 Go de RAM pour le driver Spark) ou ouvrir le notebook.

## Fichiers

```
TP_kaggle_ml_spark.ipynb   notebook complet, exécuté (sorties incluses)
predict_datacenter.py      déploiement sklearn (prédiction unitaire)
score_checkup.py           déploiement Spark (scoring batch)
world_data/  checkup_data/ données Kaggle (non versionnées)
models/  checkup_clean.parquet   générés par le notebook (non versionnés)
```
