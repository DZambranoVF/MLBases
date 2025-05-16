import streamlit as st
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

st.title("🧠 Red Neuronal para Clasificación de Dígitos")
st.markdown("""
🧠 **¿Qué es una red neuronal?**

Es un modelo inspirado en el cerebro humano. Está compuesto por neuronas artificiales conectadas en capas.

Cada neurona aplica una transformación matemática, y el resultado se propaga. Se entrena ajustando los "pesos" entre neuronas.

El modelo que usamos (MLP) clasifica imágenes de dígitos (0–9) a partir de pixeles.

🧠 Útil para:
- Reconocimiento de imágenes
- Análisis de texto
- Diagnósticos médicos, juegos, predicciones complejas
    """)

st.markdown("""
Aquí usamos una **red neuronal multicapa (MLP)** para reconocer dígitos del 0 al 9 🧮.

Usamos el dataset `digits` de sklearn con imágenes de 8x8 pixeles.
""")

# Datos
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, random_state=42)

# Modelo
modelo = MLPClassifier(hidden_layer_sizes=(50,), max_iter=500)
modelo.fit(X_train, y_train)

# Interfaz
idx = st.slider("Ver imagen del dígito (0–1796)", 0, len(digits.images)-1, 10)
fig, ax = plt.subplots()
ax.imshow(digits.images[idx], cmap='gray')
st.pyplot(fig)

pred = modelo.predict([digits.data[idx]])[0]
st.success(f"Predicción del modelo: **{pred}** (etiqueta real: {digits.target[idx]})")