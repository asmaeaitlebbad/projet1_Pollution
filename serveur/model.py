import warnings
warnings.filterwarnings('ignore')
import pickle
import pandas as pd
 
MODEL_NAME = "random_forest_air.pkl"
model = pickle.load(open(MODEL_NAME, 'rb'))
 
def pred_proba(data):
    X = data.values
    pred = int(model.predict([X[0]])[0])
    try:
        proba = float(model.predict_proba([X[0]])[0][pred])
    except AttributeError:
        proba= None
   
    return pred, proba