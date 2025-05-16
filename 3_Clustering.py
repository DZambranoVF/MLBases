import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

st.title("🔍 Clustering con K-Means")
st.markdown("""
🔍 **¿Qué es Clustering y KMeans?**

Clustering busca **agrupar datos sin etiquetas**. No predice algo específico, sino que organiza.

KMeans agrupa los datos según similitud (distancia), y ubica "centros" para definir cada grupo.

🧠 Útil para:
- Segmentar clientes
- Detectar anomalías o patrones
- Recomendaciones sin datos previos
    """)

st.markdown("""
El **clustering** permite descubrir grupos naturales en tus datos sin tener etiquetas.

🧠 En este ejemplo agrupamos puntos simulados en el espacio con **KMeans**, visualizando cómo se forman los clústeres.
""")

# Simular datos
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=200, centers=4, cluster_std=1.5, random_state=42)

# Modelo
k = st.slider("Número de clústeres", 2, 6, 4)
modelo = KMeans(n_clusters=k, n_init=10)
y_kmeans = modelo.fit_predict(X)

# Visual
fig, ax = plt.subplots()
scatter = ax.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
centros = modelo.cluster_centers_
ax.scatter(centros[:, 0], centros[:, 1], c='red', s=200, alpha=0.75, label='Centros')
ax.legend()
st.pyplot(fig)