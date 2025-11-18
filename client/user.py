import warnings
warnings.filterwarnings('ignore')
 
import streamlit as st
 
def descripteurs_input():
    st.sidebar.header("Entrer les mesures des capteurs")
 
    pm25 = st.sidebar.slider('PM2.5', 0.0, 500.0, 12.0, step=0.1)
    pm10 = st.sidebar.slider('PM10', 0.0, 600.0, 25.0, step=0.1)
    no2 = st.sidebar.slider('NO2', 0.0, 500.0, 10.0, step=0.1)
    so2 = st.sidebar.slider('SO2', 0.0, 500.0, 5.0, step=0.1)
    co = st.sidebar.slider('CO', 0.0, 50.0, 0.5, step=0.01)
   
 
    data = {
        "PM2.5": pm25,
        "PM10": pm10,
        "NO2": no2,
        "SO2": so2,
        "CO": co
    }
 
    return data