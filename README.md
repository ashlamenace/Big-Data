# TP3 – Workflow ML complet : World Data Centers (scikit-learn, puis Spark)

Workflow `Data → EDA → Prétraitement → Transformation → ML → Évaluation → Déploiement` sur
[World Data Centers](https://www.kaggle.com/datasets/rockyt0/world-data-centers) (OpenStreetMap, 4 802 sites) : prédire le type de site
(`colocation`, `hyperscale`, `telecom_carrier`).

- Partie 1 : pandas + scikit-learn (`Pipeline` + `ColumnTransformer`, régression logistique vs forêt aléatoire, validation croisée 5 plis, sauvegarde `joblib`).
  Meilleur modèle : forêt aléatoire, F1 macro CV **0,755**, précision test **0,81** (classe `telecom_carrier` plus faible : F1 0,45).
- Partie 2 : même workflow refait avec PySpark MLlib, plus agrégation SQL par pays.

Les noms de sites ne sont pas utilisés pour prédire le type (fuite de données). `class_weight="balanced"` et F1 macro pour le déséquilibre de classes.

## Fichiers

```
TP_world_data_centers.ipynb   notebook complet, exécuté (sorties incluses)
predict_datacenter.py         déploiement : prédiction unitaire avec models/datacenter_rf.joblib
world_data/                   données Kaggle (non versionnées)
```

## Reproduire

`pip install pyspark==3.5.3 pandas scikit-learn matplotlib joblib jupyter` (Java 17 requis par Spark), extraire les CSV dans `world_data/`,
puis `jupyter nbconvert --to notebook --execute --inplace TP_world_data_centers.ipynb`.
Exemple : `python predict_datacenter.py 60000 45.5 -122.6 "North America"`.
