import streamlit as st

st.set_page_config(page_title="Aprende ML Jugando", layout="wide")

st.title("🧠 Aprende Machine Learning Jugando")

st.markdown("""
Bienvenido a esta app educativa sobre **Machine Learning**. Aquí aprenderás cómo funcionan los algoritmos más comunes de forma didáctica, visual y con ejemplos interactivos.

---  
## 🔍 ¿Qué es Machine Learning?

Machine Learning (ML) es una rama de la inteligencia artificial que permite que las computadoras aprendan **a partir de datos**, sin ser explícitamente programadas para cada tarea.

En lugar de decirle “haz A, luego B, luego C”, le damos **muchos ejemplos de entrada y salida**, y el modelo aprende a predecir nuevas salidas.

### 📘 Tipos principales de aprendizaje:

- **Supervisado:** Los datos están etiquetados. Ej: saber que X = [200m²] → Y = $400M.
- **No supervisado:** El modelo agrupa o descubre patrones sin conocer los resultados esperados.
- **Por refuerzo:** El modelo toma decisiones a prueba y error y aprende de recompensas.

---

## 🎯 ¿Qué aprenderás aquí?

✅ Cómo funciona KNN para clasificar objetos  
✅ Cómo usar regresión para predecir precios  
✅ Cómo los algoritmos agrupan sin etiquetas (clustering)  
✅ Qué hace una red neuronal y cómo clasifica imágenes

---

## 📚 Módulos disponibles:

Usa el menú lateral (⬅️) para acceder a cada módulo:

1. 📊 KNN: Clasificación de frutas
2. 📈 Regresión lineal: Predicción de precios
3. 🔍 Clustering con KMeans
4. 🧠 Redes neuronales (MLP)

Cada módulo incluye:
- Explicación clara del algoritmo
- Ejemplos visuales y fáciles de usar
- Experimentos interactivos

---

👩‍🏫 *Este recorrido está diseñado para estudiantes, curiosos, profesionales y autodidactas que quieren entender de forma simple lo que hay detrás de la inteligencia artificial que usamos todos los días.*

¡Empieza tu viaje por el mundo de los algoritmos! 🚀
""")