from flask import Flask, request, jsonify
import pandas as pd
from model import pred_proba
 
app = Flask(__name__)
 
@app.route('/model', methods=['POST'])
def get_info():
    try:
        data = request.get_json(force=True)
        print("Données reçues :", data)
        mesures = data.get('data', {})
        mesures.pop('O3', None)
        for key in mesures:
            mesures[key] = float(mesures[key])
        new_data = pd.DataFrame([mesures])
        pred, proba = pred_proba(new_data)
        print(f'pred={pred}, --> proba={proba}')
        dict_resultat = {'prediction': pred, 'proba': proba}
        return jsonify(dict_resultat)
 
    except Exception as e:
        print("Erreur serveur :", e)
        return jsonify({'error': str(e)}), 500
 
if __name__ == '__main__':
    app.run(debug=True, port=5000)
 