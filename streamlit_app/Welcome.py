import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Datos de ejemplo (se pueden reemplazar con los datos reales)
# Cargar los datos

# Configuración de la página
st.set_page_config(
    page_title="Bienvenida",
    page_icon="⚽",
    layout="centered"
)
@st.cache_data
def load_data():
    df = pd.read_csv('data.csv')
    df = df[df.season!='2023-2024']
    return df
df = load_data()
# Página de bienvenida
st.title("Bienvenido a la Herramienta de Análisis de Rendimiento de Jugadores")
st.image("streamlit_app/imagen.png")

st.markdown("""
### ¿Qué puedes hacer con esta herramienta?
Esta aplicación te permitirá realizar análisis detallados del rendimiento de jugadores a lo largo de las temporadas. Podrás responder preguntas como:
- ¿En qué posiciones tiene tu equipo menos potencial comparado con sus rivales?
- ¿Cuáles son los puntos débiles de tus rivales?
- ¿Cómo ha sido el rendimiento detallado de un jugador específico?
- ¿Qué jugadores son los más similares a un jugador específico?
- ¿Cuál es la predicción del rendimiento de un jugador para la próxima temporada y porqué?

### Parámetros Ajustables
Podrás ajustar diversos parámetros para personalizar tus análisis, incluyendo:
- Selección de estadísticas específicas
- Filtros por equipo, liga, temporada, edad de los jugadores, y más
- Configuración de los modelos de predicción

### Datos de Muestra
A continuación, puedes ver una muestra de los datos con los que trabajarás:
""")

# Mostrar una muestra de los datos
st.write(df[['player', 'season', 'team', 'pos'] + [col for col in df.columns if col not in ['player', 'season', 'team', 'pos']]].sample(10))

# Comentario para agregar imágenes o recursos externos
# Puedes agregar una imagen de bienvenida usando st.image('ruta_de_la_imagen.jpg')
# Explicación breve sobre las funcionalidades de la herramienta
st.markdown("""
### Pestañas de la aplicación

#### Identificación de Puntos Débiles
En esta sección, podrás comparar a tu equipo con otros para identificar áreas de mejora.

#### Detalles del Jugador
Analiza en profundidad el rendimiento de un jugador específico a lo largo de la última temporada.

#### Comparación de Jugadores
Encuentra jugadores similares al que estás analizando y compara sus estadísticas para tomar decisiones informadas.

#### Predicción del Rendimiento
Utiliza modelos de predicción para estimar el rendimiento futuro de los jugadores y tomar decisiones estratégicas.
""")

st.markdown("""
### ¿Listo para empezar?
Navega por las páginas de la barra lateral en la izquierda de la pantalla para comenzar tu análisis.
""")

# Comentario para el desarrollador
# Puedes personalizar más la interfaz según las necesidades específicas del usuario y los datos disponibles.
