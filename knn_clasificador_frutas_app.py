
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# Titulo de la app
st.title("🔍 Clasificador de Frutas con KNN")
st.markdown("""
Esta app educativa te muestra cómo funciona un algoritmo de Machine Learning para clasificar frutas 
según sus características (peso y textura). 

Puedes experimentar con tus propias frutas misteriosas y ver qué predice el modelo. 🍎🍌🍇
""")

# Dataset simulado
data = {
    "peso": [150, 180, 200, 120, 170, 130, 160, 110, 190, 210],
    "textura": [1, 1, 1, 0, 1, 0, 1, 0, 1, 1],  # 1 = Suave, 0 = Rugosa
    "fruta": ["manzana", "manzana", "manzana", "naranja", "manzana",
              "naranja", "manzana", "naranja", "manzana", "manzana"]
}
df = pd.DataFrame(data)

# Entrenar el modelo
X = df[["peso", "textura"]]
y = df["fruta"]
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# Inputs del usuario
st.sidebar.header("🔧 Tu fruta misteriosa")
peso = st.sidebar.slider("Peso (g)", 100, 220, 150)
textura = st.sidebar.radio("Textura", ["Suave", "Rugosa"])
textura_val = 1 if textura == "Suave" else 0

# Prediccion
prediccion = knn.predict([[peso, textura_val]])[0]
st.subheader("🧠 Predicción del modelo:")
st.success(f"Tu fruta misteriosa es probablemente una **{prediccion}**")

# Visualización
fig, ax = plt.subplots()
for fruta in df["fruta"].unique():
    subset = df[df["fruta"] == fruta]
    ax.scatter(subset["peso"], subset["textura"], label=fruta)

ax.scatter(peso, textura_val, color="red", label="Tu fruta", marker="x", s=100)
ax.set_xlabel("Peso (g)")
ax.set_ylabel("Textura (1=Suave, 0=Rugosa)")
ax.set_title("Visualización de la Clasificación")
ax.legend()
st.pyplot(fig)

st.markdown("""
---
Esta app usa el algoritmo **KNN** que clasifica según la cercanía a otros datos.
Puedes ajustar el peso y la textura para ver cómo cambia la predicción. 
¡Así funciona el aprendizaje supervisado en ML! 🧠
""")
