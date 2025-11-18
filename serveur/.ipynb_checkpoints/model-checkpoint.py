import warnings
warnings.filterwarnings('ignore')
import pickle
import pandas as pd

# Chargement du modèle
MODEL_NAME = 'random_forest_air.pkl'
model = pickle.load(open(MODEL_NAME, 'rb'))

def pred_proba(data):
    """
    data : DataFrame (1 ligne) avec les mêmes colonnes que pendant l'entraînement
    Retour : (prediction, probability)
    """
    # convertir en tableau
    X = data.values
    
    # prédiction de la classe
    pred = int(model.predict([X[0]])[0])
    
    # probabilité associée
    try:
        proba = float(model.predict_proba([X[0]])[0][pred])
    except AttributeError:
        proba = None
    
    return pred, proba
