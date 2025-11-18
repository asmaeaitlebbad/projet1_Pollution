# client.py
import json, requests

URL = 'http://127.0.0.1:5000/predict'

# exemple de payload (à adapter)
payload = {
    "model": "random_forest",
    "features": {
        "PM2.5": 15.2,
        "PM10": 30.1,
        "NO2": 6.5,
        "O3": 12.0,
        "SO2": 1.2,
        "CO": 0.4
    }
}

resp = requests.post(URL, json=payload)
print("Status:", resp.status_code)
try:
    print("JSON:", resp.json())
except:
    print("Texte:", resp.text)
