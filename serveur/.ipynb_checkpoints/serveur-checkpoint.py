# serveur.py
from flask import Flask, request, jsonify
import pandas as pd
from model import pred_proba  # ton model.py pour la qualité de l'air

app = Flask(__name__)

@app.route('/model', methods=['POST'])
def get_info():
    
    data = request.get_json(force=True)
    
    # Convertir en DataFrame 1 ligne (comme le prof)
    new_data = pd.DataFrame(data=data, index=[1])
    
    # Obtenir la prédiction et la probabilité
    pred, proba = pred_proba(new_data)
    print(f'pred={pred}, --> proba={proba}')
    
    # Créer le dictionnaire résultat (comme le prof)
    dict_resultat = {'class': pred, 'proba': proba}
    
    return jsonify(dict_resultat)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
