## Description du Projet
 
Application de prédiction de la qualité de l'air en temps réel utilisant l'apprentissage automatique. Le système permet de classifier la qualité de l'air en différentes catégories (Bonne, Modérée, Mauvaise, Dangereuse) à partir de mesures de capteurs environnementaux.
 
### Objectifs
 
- Développer une architecture client-serveur pour la prédiction de la qualité de l'air
- Implémenter des modèles de Machine Learning pour la classification
- Fournir une interface utilisateur intuitive avec Streamlit
- Créer une API REST avec Flask pour les prédictions
 
## Architecture du Projet
 
```
Projet/
├── app.py                 # Application Streamlit (Frontend)
├── serveur.py            # API Flask (Backend)
├── model.py              # Gestion du modèle ML
├── user.py               # Interface utilisateur Streamlit
├── train_model.py        # Script d'entraînement du modèle
├── client.py             # Client de test API
├── projet.ipynb          # Notebook d'analyse exploratoire
├── random_forest_air.pkl # Modèle entraîné sauvegardé
└── README.md             # Documentation
```
 
## Technologies Utilisées
 
### Backend
- **Flask** - Framework web pour l'API REST
- **scikit-learn** - Bibliothèque de Machine Learning
- **pandas** - Manipulation de données
- **pickle** - Sérialisation du modèle
 
### Frontend
- **Streamlit** - Interface utilisateur interactive
- **requests** - Communication avec l'API
 
### Machine Learning
- **Random Forest Classifier** - Algorithme de classification utilisé
 
## Données
 
### Variables d'entrée (Capteurs)
- **PM2.5** - Particules fines (0.0 - 500.0 µg/m³)
- **PM10** - Particules en suspension (0.0 - 600.0 µg/m³)
- **NO2** - Dioxyde d'azote (0.0 - 500.0 µg/m³)
- **SO2** - Dioxyde de soufre (0.0 - 500.0 µg/m³)
- **CO** - Monoxyde de carbone (0.0 - 50.0 ppm)
 
### Variable cible
- **Qualite_air** - Classification de la qualité (0: Bonne, 1: Modérée, 2: Mauvaise, 3: Dangereuse)
 
## 🚀 Installation et Configuration
 
### Installation des dépendances
 
```bash
python -m venv venv
venv\Scripts\activate
 
pip install streamlit flask pandas scikit-learn numpy requests
```
 
##  Utilisation
 
### 1. Entraîner le modèle
 
```bash
python train_model.py
```
Ce script génère le fichier `random_forest_air.pkl` contenant le modèle entraîné.
 
### 2. Lancer le serveur Backend (Flask)
 
```bash
python serveur.py
```
Le serveur démarre sur `http://127.0.0.1:5000`
 
**Sortie attendue:**
```
 * Running on http://127.0.0.1:5000
 * Debug mode: on
```
 
### 3. Lancer l'application Frontend (Streamlit)
 
Dans un nouveau terminal:
```bash
streamlit run app.py
```
L'application s'ouvre automatiquement dans votre navigateur sur `http://localhost:8501`
 
### 4. Utiliser l'application
 
1. **Ajuster les mesures des capteurs** via les sliders dans la barre latérale
2. **Visualiser les valeurs** saisies dans le panneau principal
3. **Cliquer sur "Prédire la qualité de l'air"** pour obtenir la prédiction
4. **Consulter le résultat** avec la catégorie de qualité et la probabilité
 
## 🔌 API REST
 
### `/model`
 
**Méthode:** POST
 
**Format de requête:**
```json
{
  "data": {
    "PM2.5": 15.2,
    "PM10": 30.1,
    "NO2": 6.5,
    "SO2": 1.2,
    "CO": 0.3
  }
}
```
 
**Format de réponse (succès):**
```json
{
  "prediction": 1,
  "proba": 0.87
}
```
 
**Format de réponse (erreur):**
```json
{
  "error": "Description de l'erreur"
}
```
 
### Test de l'API avec curl
 
```bash
curl -X POST http://127.0.0.1:5000/model \
  -H "Content-Type: application/json" \
  -d '{"data": {"PM2.5": 15.2, "PM10": 30.1, "NO2": 6.5, "SO2": 1.2, "CO": 0.3}}'
```
 
### Test avec Python (client.py)
 
```bash
python client.py
```
 
## Modèle de Machine Learning
 
Le modèle a été entraîné sur un dataset de 5000 observations avec les caractéristiques suivantes:
- Features: PM2.5, PM10, NO2, SO2, CO, Temperature, Humidity, Proximité zones industrielles, Densité population
- Classes: 4 catégories de qualité d'air (0-3)
 
## Interface Utilisateur
 
### Page principale
- **Titre**: Prédiction de la qualité de l'air
- **Barre latérale**: Sliders pour ajuster les mesures des capteurs
- **Zone principale**: Affichage des mesures et résultats de prédiction
 
### Fonctionnalités
- Saisie interactive via sliders
- Validation en temps réel
- Affichage de la probabilité de prédiction
- Gestion des erreurs serveur
 
## Structure des Fichiers
 
### `app.py`
Application Streamlit principale - interface utilisateur et coordination avec l'API
 
### `serveur.py`
Serveur Flask - expose l'API REST pour les prédictions
 
### `model.py`
Chargement du modèle et fonction de prédiction
 
### `user.py`
Composant Streamlit pour la saisie des données utilisateur
 
### `train_model.py`
Script d'entraînement du modèle Random Forest
 
### `client.py`
Client de test pour l'API REST
 
### `projet.ipynb`
Notebook Jupyter contenant l'analyse exploratoire des données
 
## Licence
 
Ce projet est développé dans le cadre du cours **420-IAA-TT** à l'**Institut Teccart** - Automne 2025
 de développement
 
##  Remerciements
 
- Institut Teccart pour le support académique
- Professeur Benfriha Hichem pour les directives du projet