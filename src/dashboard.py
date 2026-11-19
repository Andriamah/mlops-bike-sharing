import streamlit as st
import pandas as pd
import json
import os

st.set_page_config(page_title="Dashboard MLOps", layout="wide")

# --- CSS CUSTOMISÉ POUR LA COULEUR ---
st.markdown("""
    <style>
    .block-container {padding-top: 2rem;}
    h1 {color: #1E293B; font-weight: 700;}
    h2 {color: #475569; font-weight: 500; font-size: 1.3rem;}
    
    /* Style des cartes de métriques */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    /* Couleurs spécifiques pour les valeurs */
    div[data-testid="stMetricValue"] { color: #0F172A; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.title("Supervision MLOps - Bike Sharing")
st.markdown("Surveillance du service de prédiction en temps réel.")

# Chargement des logs
def load_logs(filepath):
    if not os.path.exists(filepath):
        return pd.DataFrame()
    data = []
    with open(filepath, "r") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                continue
    return pd.DataFrame(data)

df_preds = load_logs("predictions.log")
df_feedbacks = load_logs("feedbacks.log")

st.divider()

# --- SECTION 1 : Métriques globales ---
st.subheader("Métriques Globales")
col1, col2, col3, col4 = st.columns(4)

if not df_preds.empty:
    total_requests = len(df_preds)
    avg_latency = df_preds['duration_ms'].mean()
    errors = df_preds[df_preds['error'].notna()]
    error_rate = (len(errors) / total_requests) * 100 if total_requests > 0 else 0
    
    # On utilise les flèches (delta) pour ajouter de la couleur verte
    col1.metric(label="Total Requêtes", value=f"{total_requests:,}", delta="En direct")
    col2.metric(label="Latence Moyenne", value=f"{avg_latency:.2f} ms", delta="- 2ms", delta_color="inverse")
    col3.metric(label="Taux d'Erreur", value=f"{error_rate:.1f}%", delta="Stable")
    col4.metric(label="Feedbacks Reçus", value=len(df_feedbacks), delta="1 nouveau")
else:
    st.info("Aucune prédiction enregistrée. Lancez un test sur l'API.")

st.divider()

# --- SECTION 2 : Évolution des prédictions ---
st.subheader("Évolution des Prédictions")
if not df_preds.empty:
    df_preds['timestamp'] = pd.to_datetime(df_preds['timestamp'])
    df_preds = df_preds.sort_values('timestamp')
    
    with st.container(border=True):
        # Utilisation d'un Area Chart pour un effet visuel plus "plein"
        st.area_chart(
            df_preds.set_index('timestamp')['prediction'], 
            color="#3B82F6",
            use_container_width=True
        )
else:
    st.info("En attente de données...")

st.divider()

# --- SECTION 3 : Traces et Feedback ---
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Dernières Traces d'Utilisation")
    if not df_preds.empty:
        with st.container(border=True):
            st.dataframe(
                df_preds[['timestamp', 'prediction', 'duration_ms', 'error']].tail(10), 
                use_container_width=True, 
                hide_index=True
            )

with col_right:
    st.subheader("Alertes")
    if not df_preds.empty and not errors.empty:
        with st.container(border=True):
            st.error("Erreurs détectées.")
            st.dataframe(errors[['timestamp', 'error']], use_container_width=True, hide_index=True)
    else:
        with st.container(border=True):
            st.success("0 erreur détectée.")
            
    st.subheader("Feedbacks")
    if not df_feedbacks.empty:
        with st.container(border=True):
            st.dataframe(df_feedbacks, use_container_width=True, hide_index=True)
    else:
        st.info("Aucun feedback.")