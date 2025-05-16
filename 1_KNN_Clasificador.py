import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

st.title("📊 Clasificación con KNN")
st.markdown("""
📊 **¿Qué es KNN?**

KNN (K-Nearest Neighbors) es un algoritmo de clasificación que **no entrena un modelo complejo**, sino que guarda todos los datos.

Cuando llega un nuevo dato (ej. una fruta desconocida), KNN:
1. Calcula la distancia entre ese dato y todos los anteriores
2. Toma los *k* más cercanos
3. Asigna la categoría más común entre ellos

🧠 Útil para:
- Clasificar imágenes, textos o productos
- Casos donde los datos no cambian tanto
    """)

st.markdown("""
El algoritmo **K-Nearest Neighbors (KNN)** clasifica elementos según sus vecinos más cercanos.

🔍 Funciona comparando nuevas observaciones con las ya conocidas, usando distancia (normalmente Euclídea).

En este ejemplo, clasificamos frutas según su peso y textura.
""")

# Dataset
data = {
    "peso": [150, 180, 200, 120, 170, 130, 160, 110, 190, 210],
    "textura": [1, 1, 1, 0, 1, 0, 1, 0, 1, 1],  # 1 = suave, 0 = rugosa
    "fruta": ["manzana", "manzana", "manzana", "naranja", "manzana", "naranja", "manzana", "naranja", "manzana", "manzana"]
}
df = pd.DataFrame(data)

X = df[["peso", "textura"]]
y = df["fruta"]
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

# Input
peso = st.slider("Peso (g)", 100, 220, 150)
textura = st.radio("Textura", ["Suave", "Rugosa"])
textura_val = 1 if textura == "Suave" else 0

# Predicción
pred = model.predict([[peso, textura_val]])[0]
st.success(f"La fruta probablemente es: **{pred}**")

# Visualización
fig, ax = plt.subplots()
for label in df["fruta"].unique():
    subset = df[df["fruta"] == label]
    ax.scatter(subset["peso"], subset["textura"], label=label)
ax.scatter(peso, textura_val, color="red", marker="x", s=100, label="Tu fruta")
ax.set_xlabel("Peso")
ax.set_ylabel("Textura (1=Suave, 0=Rugosa)")
ax.legend()
st.pyplot(fig)