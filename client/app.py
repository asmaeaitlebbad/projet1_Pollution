import warnings
warnings.filterwarnings('ignore')
 
import streamlit as st
from user import descripteurs_input
import requests
 
url = 'http://127.0.0.1:5000/model'  
 
def main():
    st.set_page_config(page_title='Qualité de l’air', page_icon=":cloud:")
    st.title(":blue[Prédiction de la qualité de l'air]")
    st.markdown(
        """
        # :orange[Application de prédiction de la qualité de l'air en ligne]
        """
    )
    st.sidebar.header('Entrer les mesures des capteurs :')
 
    mesures = descripteurs_input()
    st.write("Mesures saisies :")
    st.write(mesures)
 
    if st.button("Prédire la qualité de l'air"):
        try:
            rep = requests.post(url, json={'data': mesures})
            dict_rep = rep.json()
            if 'error' in dict_rep:
                st.error(f"Erreur serveur : {dict_rep['error']}")
            else:
                pred, proba = dict_rep['prediction'], dict_rep['proba']
                st.write(f"### Qualité de l'air : {pred}")
                st.write(f"### Probabilité : {round(proba*100,2)} %")
 
        except Exception as e:
            st.error(f"Erreur lors de la requête au serveur : {e}")
 
if __name__=='__main__':
    main()