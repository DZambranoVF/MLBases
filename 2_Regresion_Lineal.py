import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

st.title("📈 Regresión Lineal: Predicción de Precios")
st.markdown("""
📈 **¿Qué es la regresión lineal?**

Es uno de los algoritmos más antiguos y conocidos del ML. Su objetivo es **ajustar una línea recta** a los datos para poder predecir una variable numérica.

En nuestro ejemplo, la relación es:  
**precio = m * tamaño + b**

🧠 Útil para:
- Predecir precios, ingresos, demanda, temperaturas
- Analizar tendencias
    """)

st.markdown("""
La **regresión lineal** permite predecir valores continuos, como precios, en función de variables como el tamaño de una casa.

🧠 Este modelo ajusta una línea a los datos, minimizando el error cuadrático.
""")

# Simular datos
np.random.seed(0)
metros = np.random.randint(40, 200, 50)
precio = metros * 3000 + np.random.normal(0, 20000, 50).astype(int)
df = pd.DataFrame({"m2": metros, "precio": precio})

X = df[["m2"]]
y = df["precio"]
modelo = LinearRegression()
modelo.fit(X, y)

# Input
m2_input = st.slider("Tamaño de la casa (m2)", 40, 200, 100)
pred = modelo.predict([[m2_input]])[0]
st.success(f"Precio estimado: ${int(pred):,}")

# Plot
fig, ax = plt.subplots()
sns.regplot(data=df, x="m2", y="precio", ax=ax)
ax.scatter(m2_input, pred, color="red", label="Tu casa")
ax.legend()
st.pyplot(fig)