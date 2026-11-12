"""Déploiement (partie 1) : prédit le type d'un data center avec le pipeline sklearn sauvegardé.
Usage : python predict_datacenter.py <surface_m2> <latitude> <longitude> <continent> [has_operator has_name has_website has_city is_way]"""
import sys, joblib, numpy as np, pandas as pd

def predict(area_m2, lat, lon, continent, has_operator=1, has_name=1, has_website=1, has_city=1, is_way=1):
    model = joblib.load("models/datacenter_rf.joblib")
    x = pd.DataFrame([dict(log_area=np.log10(area_m2), latitude=lat, longitude=lon, has_operator=has_operator,
                           has_name=has_name, has_website=has_website, has_city=has_city, is_way=is_way, continent=continent)])
    return dict(zip(model.classes_, model.predict_proba(x)[0].round(3)))

if __name__ == "__main__":
    a = sys.argv[1:]
    print(predict(float(a[0]), float(a[1]), float(a[2]), a[3], *map(int, a[4:])))
