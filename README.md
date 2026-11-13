# TP4 – Workflow ML à grande échelle : Full Body Checkup Lab avec Spark

Workflow `Data → EDA → Prétraitement → Transformation → ML → Évaluation → Déploiement` sur
[Full Body Checkup Lab](https://www.kaggle.com/datasets/opendataforresearch/full-body-checkup-lab-dataset)
(250 164 bilans × 109 analyses, 12 CSV, ~430 Mo) : prédire le pré-diabète/diabète (`HbA1c ≥ 5,7 %`).

Outils : PySpark (DataFrame, SQL, MLlib), Parquet. Pipeline MLlib `Imputer → StringIndexer → OneHotEncoder → VectorAssembler → StandardScaler`, 3 modèles.

## Résultats

Meilleur modèle : régression logistique, AUC test **0,687** (GBT 0,685, forêt 0,655), accuracy 0,737.
Le signal est **modeste** : sans les mesures de glycémie, les autres analyses expliquent mal l'HbA1c (prévalence positive ≈ 27 %,
un classifieur qui répond toujours « non » atteint déjà ≈ 73 % d'accuracy). Données (semi-)synthétiques, sans valeur clinique.

## Choix méthodologiques

- Fuite de données évitée : `fasting_glucose`, `fbs`, `plbs`, `pp_glucose`, `eag` et `hba1c` exclus des variables explicatives.
- Split par patient (hash de `MedID`) : un patient apparaît dans les 12 mois.
- Imputation dans le pipeline, après le split.
- Parquet pour le jeu nettoyé ; en cluster, remplacer `local[*]` par YARN et les chemins locaux par `hdfs://…`.

## Fichiers

```
TP_full_body_checkup.ipynb   notebook complet, exécuté (sorties incluses)
score_checkup.py             déploiement : scoring batch avec le PipelineModel Spark
checkup_data/                données Kaggle (non versionnées)
```

## Reproduire

`pip install pyspark==3.5.3 pandas matplotlib jupyter` (Java 17 requis), extraire les CSV dans `checkup_data/`
(sous Windows : `winutils.exe`/`hadoop.dll` dans `C:\hadoop\bin`, `HADOOP_HOME=C:\hadoop`),
puis `jupyter nbconvert --to notebook --execute --inplace TP_full_body_checkup.ipynb` (≈ 5 min, 6 Go de RAM).
Scoring : `spark-submit score_checkup.py checkup_clean.parquet scores.parquet`.
