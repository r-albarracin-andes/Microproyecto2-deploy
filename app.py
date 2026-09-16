# ==============================================================================
# Aplicación Web Interactiva de Clasificación en los ODS (Streamlit)
# Maestría en Inteligencia Artificial Aplicada (MAIA) - Universidad de los Andes
# Microproyecto 2: Procesamiento de Lenguaje Natural y Machine Learning No Supervisado
# Autores: Rafael Albarracin (r.albarracin@uniandes.edu.co)
#          Juan Carlos Ramirez (jc.ramirezg1@uniandes.edu.co)
# ==============================================================================
import os
import sys
import base64
import platform
import pathlib
import numpy as np
import pandas as pd
import streamlit as st

import importlib

# ==============================================================================
# Parche de Compatibilidad Multiplataforma para Despliegue en Azure Linux
# Permite deserializar artefactos generados en Windows que contengan 'WindowsPath'
# ==============================================================================
if 'pathlib._local' not in sys.modules:
    try:
        importlib.import_module('pathlib._local')
    except ImportError:
        sys.modules['pathlib._local'] = pathlib

if platform.system() != 'Windows' or os.name != 'nt':
    pathlib.WindowsPath = pathlib.PosixPath
    if hasattr(pathlib, '_local'):
        pathlib._local.WindowsPath = pathlib.PosixPath
else:
    pathlib.PosixPath = pathlib.WindowsPath
    if hasattr(pathlib, '_local'):
        pathlib._local.PosixPath = pathlib.WindowsPath

# Importar catálogo, funciones y cargador robusto del módulo modelo_ods
from modelo_ods import (
    diccionario_ods,
    colores_ods,
    iconos_ods,
    aplicar_parche_compatibilidad_pathlib,
    cargar_pipeline_ods,
    predecir_ods_con_probabilidad
)

# ------------------------------------------------------------------------------
# Configuración de Rutas de Recursos
# ------------------------------------------------------------------------------
BASE_DIR = pathlib.Path(__file__).resolve().parent
RESOURCES_DIR = BASE_DIR / "resources"
LOGO_PATH = RESOURCES_DIR / "logo-uniandes.png"
BANNER_PATH = RESOURCES_DIR / "Banner-Machine-Learning-No-Supervisado.jpg"

def cargar_imagen_base64(ruta_archivo):
    try:
        if isinstance(ruta_archivo, (str, pathlib.Path)) and os.path.exists(str(ruta_archivo)):
            with open(ruta_archivo, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        pass
    return None

# ------------------------------------------------------------------------------
# Configuración Global de la Página de Streamlit
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Clasificador ODS - Agenda 2030 | MAIA UniAndes",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------------------
# Estilos CSS Personalizados Premium
# ------------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* -------------------------------------------------------------------------
       Tema Blanco Permanente (Deshabilitación Estricta de Modo Oscuro)
       ------------------------------------------------------------------------- */
    :root {
        color-scheme: light !important;
    }
    
    html, body, [class*="css"], [data-testid="stAppViewContainer"], .main {
        font-family: 'Inter', sans-serif;
        background-color: #ffffff !important;
        color: #1e293b !important;
    }
    
    [data-testid="stSidebar"] {
        background-color: #f8fafc !important;
    }

    /* Ocultar únicamente el menú de 3 puntos (ajustes de tema) y footer, dejando intactos los controles del sidebar */
    #MainMenu, footer, div[class*="StyledHeaderRightSection"], [data-testid="stToolbarActions"] {
        visibility: hidden !important;
        display: none !important;
    }

    /* Asegurar que el header y la barra que contiene el botón de apertura del sidebar permanezcan activos */
    header[data-testid="stHeader"],
    .stAppHeader {
        display: block !important;
        visibility: visible !important;
        background: transparent !important;
        z-index: 99999 !important;
    }

    [data-testid="stToolbar"],
    .stAppToolbar {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
    }

    @media (prefers-color-scheme: dark) {
        html, body, [data-testid="stAppViewContainer"], .main {
            background-color: #ffffff !important;
            color: #1e293b !important;
        }
        [data-testid="stSidebar"] {
            background-color: #f8fafc !important;
        }
    }

    /* -------------------------------------------------------------------------
       Control de Expansión y Colapso de Barra Lateral (Sidebar)
       Garantiza que el botón siempre sea visible, clickable y claro al colapsar
       ------------------------------------------------------------------------- */
    div[class*="StyledOpenSidebarButton"],
    div[class*="StyledHeaderLeftSection"],
    [data-testid="stSidebarCollapsedControl"],
    button[data-testid="stSidebarCollapseButton"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
    }

    div[class*="StyledOpenSidebarButton"] button,
    [data-testid="stSidebarCollapsedControl"] button,
    button[data-testid="stSidebarCollapseButton"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        background-color: #ffffff !important;
        color: #0d3b66 !important;
        border: 1.5px solid #cbd5e1 !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.12) !important;
        cursor: pointer !important;
    }

    div[class*="StyledOpenSidebarButton"] button:hover,
    [data-testid="stSidebarCollapsedControl"] button:hover,
    button[data-testid="stSidebarCollapseButton"]:hover {
        background-color: #f1f5f9 !important;
        border-color: #0d3b66 !important;
        color: #001e3d !important;
    }

    div[class*="StyledOpenSidebarButton"] svg,
    [data-testid="stSidebarCollapsedControl"] svg,
    button[data-testid="stSidebarCollapseButton"] svg {
        fill: #0d3b66 !important;
        stroke: #0d3b66 !important;
        color: #0d3b66 !important;
    }

    .main-header {
        background: linear-gradient(135deg, #0d3b66 0%, #001e3d 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 25px rgba(0,0,0,0.12);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        color: #ffffff;
    }
    
    .hero-subtitle {
        font-size: 1.02rem;
        font-weight: 400;
        color: #b0c7de;
        line-height: 1.5;
    }
    
    .ods-card-dominant {
        color: white;
        padding: 1.8rem 2rem;
        border-radius: 14px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.16);
        transition: transform 0.2s ease-in-out;
    }
    
    .ods-card-dominant:hover {
        transform: translateY(-2px);
    }
    
    .ods-badge {
        display: inline-block;
        padding: 0.4rem 0.85rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        color: white;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }
    
    .metric-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0d3b66;
    }
    
    .metric-label {
        font-size: 0.82rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# Carga del Modelo Serializado con Caché de Alta Eficiencia
# ------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando modelo serializado de machine learning (model/pipeline_ods.joblib)...")
def obtener_pipeline_ods():
    try:
        aplicar_parche_compatibilidad_pathlib()
        return cargar_pipeline_ods()
    except Exception as e:
        st.error(f"⚠️ Error al cargar el pipeline serializado: {e}")
        return None

pipeline = obtener_pipeline_ods()

# ------------------------------------------------------------------------------
# Barra Lateral (Sidebar)
# ------------------------------------------------------------------------------
with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), use_container_width=True)
    elif os.path.exists("resources/logo-uniandes.png"):
        st.image("resources/logo-uniandes.png", use_container_width=True)
    
    st.markdown("---")
    st.markdown("### ⚙️ Parámetros de Inferencia")
    
    top_k = st.slider(
        "Candidatos Top-k a extraer:",
        min_value=1,
        max_value=5,
        value=3,
        step=1,
        help="Número de ODS más probables a desplegar ordenados por confianza."
    )
    
    umbral_afinidad = st.slider(
        "Umbral de Afinidad Multietiqueta (%):",
        min_value=5,
        max_value=40,
        value=15,
        step=5,
        help="Porcentaje mínimo para que un ODS secundario sea considerado afinidad significativa (15% según rúbrica)."
    ) / 100.0
    
    st.markdown("---")
    st.markdown("### 🎓 Proyecto Académico")
    st.markdown("""
    **Programa:** Maestría en IA Aplicada (MAIA)  
    **Institución:** Universidad de los Andes  
    **Materia:** Machine Learning No Supervisado  
    **Autores:**  
    - Rafael Albarracin  
    - Juan Carlos Ramirez  
    
    **Pipeline Desplegado:**  
    `spaCy NLP` ➔ `TF-IDF (5,000)` ➔ `LSA SVD (k=20)` ➔ `Regresión Logística (C=10.0, balanced)`
    """)
    
    st.markdown("---")
    with st.expander("📚 Catálogo Oficial de los ODS"):
        for cod, nom in diccionario_ods.items():
            color = colores_ods.get(cod, '#333333')
            icono = iconos_ods.get(cod, '📌')
            st.markdown(
                f"<span style='color:{color}; font-weight:700;'>ODS {cod}</span>: {icono} {nom}",
                unsafe_allow_html=True
            )

# ------------------------------------------------------------------------------
# Banner Principal y Encabezado Secundario
# ------------------------------------------------------------------------------
banner_b64 = cargar_imagen_base64(BANNER_PATH) or cargar_imagen_base64("resources/Banner-Machine-Learning-No-Supervisado.jpg")
if banner_b64:
    st.markdown(f"""
    <div style="display: flex; width: 100%; margin-bottom: 1.2rem;">
        <img src="data:image/jpeg;base64,{banner_b64}" 
             alt="Banner Machine Learning No Supervisado" 
             style="width: 60%; height: auto; border-radius: 12px; box-shadow: 0 4px 18px rgba(0,0,0,0.08);">
    </div>
    """, unsafe_allow_html=True)
elif BANNER_PATH.exists():
    st.image(str(BANNER_PATH), width=860)

st.markdown("""
<div class="main-header">
    <div class="hero-title">🌍 Clasificador Inteligente de Textos en los ODS</div>
    <div class="hero-subtitle">
        Herramienta analítica basada en <b>Procesamiento de Lenguaje Natural (NLP)</b>, <b>Modelado de Tópicos (LSA)</b> 
        y <b>Regresión Logística Calibrada</b> para categorizar automáticamente textos, políticas públicas y proyectos sociales dentro de la <b>Agenda 2030</b>.
    </div>
</div>
""", unsafe_allow_html=True)

if pipeline is None:
    st.error("⚠️ No se pudo cargar el archivo serializado `model/pipeline_ods.joblib`. Por favor verifica que el archivo exista en la subcarpeta `model/`.")
    st.stop()

# ------------------------------------------------------------------------------
# Pestañas de la Aplicación
# ------------------------------------------------------------------------------
tab_individual, tab_demo, tab_metricas = st.tabs([
    "✍️ Clasificación Individual",
    "🎲 Demostrador Aleatorio",
    "📊 Métricas del Modelo (Test Set)"
])

# ==============================================================================
# PESTAÑA 1: Clasificación Individual
# ==============================================================================
with tab_individual:
    st.markdown("#### 💡 Selecciona un texto de ejemplo predefinido:")
    
    # Inicializar estado para el texto
    if 'texto_actual' not in st.session_state:
        st.session_state.texto_actual = ""

    col_e1, col_e2, col_e3, col_e4 = st.columns(4)
    with col_e1:
        if st.button("💧 ODS 6: Agua y Saneamiento", use_container_width=True):
            st.session_state.texto_actual = "La construcción de plantas comunitarias de tratamiento de aguas residuales y el acceso a saneamiento básico previene enfermedades bacterianas e hídricas en la población infantil."
        if st.button("🪙 ODS 1: Fin de la Pobreza", use_container_width=True):
            st.session_state.texto_actual = "Otorgar microcréditos productivos, asistencia técnica y transferencias directas condicionadas a familias vulnerables crea empleo digno y combate la pobreza extrema."
    
    with col_e2:
        if st.button("📚 ODS 4: Educación de Calidad", use_container_width=True):
            st.session_state.texto_actual = "Capacitar a los docentes en pedagogías digitales y dotar de infraestructura tecnológica a las escuelas públicas garantiza una educación inclusiva y de alta calidad."
        if st.button("⚖️ ODS 5: Igualdad de Género", use_container_width=True):
            st.session_state.texto_actual = "Eliminar las brechas salariales y promover leyes estrictas contra la violencia de género fortalece el empoderamiento femenino en cargos de liderazgo político y empresarial."
            
    with col_e3:
        if st.button("☀️ ODS 7: Energía Limpia", use_container_width=True):
            st.session_state.texto_actual = "Invertir en parques solares comunitarios y redes de energía eólica acelera la descarbonización económica y promueve el uso de fuentes limpias y no contaminantes."
        if st.button("🌱 ODS 13: Acción por el Clima", use_container_width=True):
            st.session_state.texto_actual = "Implementar estrategias municipales de adaptación al cambio climático, mitigación de riesgos de inundación y reducción de emisiones de gases de efecto invernadero."
            
    with col_e4:
        if st.button("🕊️ ODS 16: Paz y Justicia", use_container_width=True):
            st.session_state.texto_actual = "Combatir la impunidad, reforzar la independencia judicial ante tribunales y garantizar el acceso universal a la justicia consolida un estado de derecho transparente."
        if st.button("🏥 ODS 3: Salud y Bienestar", use_container_width=True):
            st.session_state.texto_actual = "Garantizar el acceso universal a servicios de atención médica primaria, vacunas esenciales y centros de salud mental previene la mortalidad materna y protege el bienestar general."

    texto_input = st.text_area(
        "Ingresa o edita el texto a clasificar:",
        value=st.session_state.texto_actual,
        height=125,
        placeholder="Escribe o pega aquí un párrafo en español para clasificarlo en los ODS..."
    )

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        btn_clasificar = st.button("🚀 Clasificar Texto", type="primary", use_container_width=True)

    if btn_clasificar:
        if not texto_input.strip():
            st.warning("⚠️ Por favor ingresa o selecciona un texto para realizar la clasificación.")
        else:
            with st.spinner("Procesando texto con spaCy, vectorizando TF-IDF y proyectando en LSA..."):
                try:
                    resultado = predecir_ods_con_probabilidad(
                        pipeline,
                        texto_input,
                        k=top_k,
                        umbral_pertenencia=umbral_afinidad,
                        diccionario_ods=diccionario_ods
                    )[0]
                except Exception as e:
                    st.error(f"Error durante la inferencia: {e}")
                    st.stop()

            # Despliegue de Resultados
            dom_cod = resultado['ODS_Dominante']
            dom_nom = resultado['Nombre_Dominante']
            dom_prob = resultado['Probabilidad_Dominante (%)']
            dom_color = colores_ods.get(dom_cod, '#0d3b66')
            dom_icono = iconos_ods.get(dom_cod, '📌')

            st.markdown("---")
            st.markdown("### 📊 Diagnóstico y Resultados de Clasificación")

            # Tarjeta Destacada del ODS Dominante
            st.markdown(f"""
            <div class="ods-card-dominant" style="background-color: {dom_color};">
                <div style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1.2px; opacity: 0.9;">
                    Predicción Principal Dominante
                </div>
                <div style="font-size: 2.2rem; font-weight: 800; margin-top: 0.3rem;">
                    {dom_icono} ODS {dom_cod}: {dom_nom}
                </div>
                <div style="font-size: 1.15rem; margin-top: 0.6rem; opacity: 0.95;">
                    Confianza Estimada del Modelo: <b>{dom_prob:.2f}%</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

            col_izq, col_der = st.columns([3, 2])

            with col_izq:
                st.markdown(f"#### 📈 Distribución de Probabilidades Top-{top_k}:")
                df_chart = pd.DataFrame(resultado['Top_ODS'])
                df_chart['Etiqueta'] = [f"ODS {row['ODS']} - {row['Nombre'][:26]}" for _, row in df_chart.iterrows()]
                
                st.bar_chart(
                    df_chart.set_index('Etiqueta')['Probabilidad (%)'],
                    color=dom_color,
                    use_container_width=True
                )

            with col_der:
                st.markdown("#### 🎯 Pertenencia Multietiqueta:")
                if len(resultado['ODS_Multiples']) > 1:
                    st.info(f"Se detectó afinidad multidimensional en **{len(resultado['ODS_Multiples'])} ODS** (Probabilidad ≥ {umbral_afinidad*100:.0f}%):")
                    for c in resultado['ODS_Multiples']:
                        c_col = colores_ods.get(c, '#555555')
                        c_ico = iconos_ods.get(c, '📌')
                        st.markdown(f"""
                        <div class="ods-badge" style="background-color: {c_col};">
                            {c_ico} ODS {c}: {diccionario_ods.get(c)}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.success(f"El texto presenta un enfoque unívoco concentrado exclusivamente en el **ODS {dom_cod}**.")

                st.markdown("---")
                st.markdown("#### 📋 Detalle Numérico:")
                for item in resultado['Top_ODS']:
                    st.markdown(f"**ODS {item['ODS']} ({item['Nombre']}):** `{item['Probabilidad (%)']:.2f}%`")
                    st.progress(min(float(item['Probabilidad (%)']) / 100.0, 1.0))

            # Sección de Trazabilidad Técnica Interna
            with st.expander("🔍 Ver trazabilidad interna del pipeline de NLP y LSA"):
                try:
                    texto_df = pd.DataFrame({'textos': [texto_input]})
                    paso_prep = pipeline.named_steps.get('preprocesamiento')
                    if paso_prep:
                        texto_limpio = paso_prep.transform(texto_df).iloc[0]
                        st.markdown("**1. Texto Normalizado y Lematizado (spaCy):**")
                        st.code(texto_limpio, language='text')

                    paso_tfidf = pipeline.named_steps.get('tfidf')
                    if paso_tfidf and paso_prep:
                        matriz_tf = paso_tfidf.transform([texto_limpio])
                        fn = np.array(paso_tfidf.get_feature_names_out())
                        indices_activos = matriz_tf.indices
                        pesos_activos = matriz_tf.data
                        if len(indices_activos) > 0:
                            orden_term = np.argsort(pesos_activos)[::-1][:10]
                            top_terminos = [f"`{fn[indices_activos[i]]}` ({pesos_activos[i]:.3f})" for i in orden_term]
                            st.markdown(f"**2. Términos con mayor peso TF-IDF:** {', '.join(top_terminos)}")

                    paso_lsa = pipeline.named_steps.get('lsa')
                    if paso_lsa:
                        k_val = getattr(paso_lsa, 'mejor_k_', 20)
                        st.markdown(f"**3. Dimensiones de Proyección Semántica Latente (LSA):** `{k_val} componentes espectrales`")
                except Exception as e_insp:
                    st.caption(f"Detalle técnico no disponible: {e_insp}")


# ==============================================================================
# PESTAÑA 2: Demostrador Aleatorio del Pipeline
# ==============================================================================
with tab_demo:
    st.markdown("### 🎲 Demostración Probabilística con Banco de Textos Coherentes")
    st.markdown(
        "Esta sección ejecuta directamente la lógica del paso **`DemostradorTextosAleatoriosODS`** "
        "incorporado en la 5ª etapa del pipeline, evaluando muestras representativas en español con coherencia semántica."
    )
    
    paso_demo = pipeline.named_steps.get('demostrador_aleatorio')
    banco = getattr(paso_demo, 'banco_textos', [
        "La construcción de plantas de tratamiento de aguas residuales y el acceso a saneamiento básico reducen bacterias.",
        "Fortalecer programas de formación pedagógica y dotar de aulas interactivas asegura educación de calidad.",
        "Invertir en parques solares y redes eólicas acelera la descarbonización con fuentes renovables.",
        "Eliminar brechas salariales y garantizar paridad femenina fortalece la igualdad de género.",
        "Garantizar acceso a atención médica primaria y salud mental previene la mortalidad materna."
    ])
    
    if st.button("🔄 Generar y Clasificar 5 Textos Aleatorios", type="primary"):
        muestras = np.random.choice(banco, size=min(5, len(banco)), replace=False)
        resultados_demo = predecir_ods_con_probabilidad(
            pipeline,
            list(muestras),
            k=top_k,
            umbral_pertenencia=umbral_afinidad,
            diccionario_ods=diccionario_ods
        )
        
        for i, (txt, res) in enumerate(zip(muestras, resultados_demo), start=1):
            cod = res['ODS_Dominante']
            col = colores_ods.get(cod, '#0d3b66')
            ico = iconos_ods.get(cod, '📌')
            
            with st.container():
                st.markdown(f"""
                <div style="border-left: 5px solid {col}; background: #f8fafc; padding: 1.2rem; border-radius: 8px; margin-bottom: 1.2rem;">
                    <div style="font-weight: 700; color: #334155; margin-bottom: 0.4rem;">Texto Aleatorio #{i}:</div>
                    <div style="font-style: italic; color: #1e293b; margin-bottom: 0.8rem;">"{txt}"</div>
                    <div style="display: flex; gap: 1rem; align-items: center; flex-wrap: wrap;">
                        <span class="ods-badge" style="background-color: {col};">
                            {ico} Predicción: ODS {cod} - {res['Nombre_Dominante']} ({res['Probabilidad_Dominante (%)']}%)
                        </span>
                        <span style="font-size: 0.85rem; color: #64748b;">
                            Afinidades (≥{umbral_afinidad*100:.0f}%): {', '.join([f'ODS {c}' for c in res['ODS_Multiples']])}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)


# ==============================================================================
# PESTAÑA 3: Métricas del Modelo y Tópicos LSA
# ==============================================================================
with tab_metricas:
    st.markdown("### 📊 Desempeño Evaluativo sobre el Conjunto de Prueba (Test Set)")
    st.markdown(
        "El pipeline serializado fue evaluado sobre el conjunto de test independiente ($N=1,932$ textos, $20\\%$) "
        "con partición estratificada (`random_state=77`), obteniendo los siguientes resultados reportados en `Entregable2.ipynb`:"
    )
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-value">81.00%</div>
            <div class="metric-label">Accuracy Global</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m2:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-value">76.63%</div>
            <div class="metric-label">Balanced Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m3:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-value" style="color: #c5192d;">75.95%</div>
            <div class="metric-label">Macro F1-Score (Principal)</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m4:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-value">81.42%</div>
            <div class="metric-label">Weighted F1-Score</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    st.markdown("#### 🧠 Interpretación Cualitativa de los Tópicos Descubiertos por LSA:")
    
    df_topicos = pd.DataFrame([
        {
            'Componente': 'Componente 2',
            'Términos Clave LSA': 'derecho, derechos, internacional, humanos, articulo, tribunal, ley',
            'ODS Relacionado': 'ODS 16: Paz, justicia e instituciones sólidas',
            'Correspondencia': 'Marco jurídico internacional, Estado de derecho y acceso universal a la justicia.'
        },
        {
            'Componente': 'Componente 3',
            'Términos Clave LSA': 'mujer, genero, ingreso, hombre, laboral, pobreza, educacion, trabajo',
            'ODS Relacionado': 'ODS 8 / ODS 10: Trabajo decente y Desigualdades',
            'Correspondencia': 'Mercado laboral, empleo formal, brechas salariales y condiciones socioeconómicas.'
        },
        {
            'Componente': 'Componente 4',
            'Términos Clave LSA': 'escuela, educacion, docente, escolar, aprendizaje, estudiant, programa',
            'ODS Relacionado': 'ODS 4: Educación de calidad',
            'Correspondencia': 'Entorno formativo escolar, pedagogía docente y calidad educativa.'
        },
        {
            'Componente': 'Componente 5',
            'Términos Clave LSA': 'mujer, genero, igualdad, hombre, igualdad genero, politica, desarrollo',
            'ODS Relacionado': 'ODS 5: Igualdad de género',
            'Correspondencia': 'Igualdad sustantiva de género, empoderamiento femenino e inclusión social.'
        },
        {
            'Componente': 'Componente 6',
            'Términos Clave LSA': 'salud, atencion, mental, paciente, salud mental, medico, enfermedad',
            'ODS Relacionado': 'ODS 3: Salud y bienestar',
            'Correspondencia': 'Sistema asistencial primario, cobertura de salud mental y prevención sanitaria.'
        },
        {
            'Componente': 'Componente 7',
            'Términos Clave LSA': 'agua, rio, agua residual, hidrico, gestion, residual, cuenca',
            'ODS Relacionado': 'ODS 6: Agua limpia y saneamiento',
            'Correspondencia': 'Recursos hídricos, saneamiento básico y tratamiento de efluentes residuales.'
        },
        {
            'Componente': 'Componente 10',
            'Términos Clave LSA': 'climatico, cambio, adaptacion, efecto, impacto, riesgo, ambiental',
            'ODS Relacionado': 'ODS 13: Acción por el clima',
            'Correspondencia': 'Mitigación y adaptación frente al cambio climático y eventos meteorológicos extremos.'
        }
    ])
    st.dataframe(df_topicos, use_container_width=True)
    
    st.markdown(r"""
    > **Conclusiones Metodológicas Destacadas:**
    > - **Regularización L2 ($C=10.0$):** La optimización interna en `fit()` mediante `StratifiedKFold` (5 folds) demostró que $C=10.0$ previene el sobreajuste en el espacio de 20 dimensiones de LSA sin perder capacidad discriminante.
    > - **Ponderación `class_weight='balanced'`:** Resultó indispensable para equilibrar la recuperación (*Recall*) en clases minoritarias como ODS 12 ($3.23\%$) y ODS 15 ($3.42\%$) frente a clases dominantes como ODS 16 ($11.18\%$).
    > - **Pertenencia Multietiqueta:** El umbral del $15\%$ captura de manera rigurosa la interconexión intrínseca de las metas de sostenibilidad (ej. agua residual impacta tanto a ODS 6 como a ODS 3).
    """)

# ------------------------------------------------------------------------------
# Pie de Página
# ------------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888888; font-size: 0.85rem; padding-bottom: 1rem;'>"
    "Desarrollado para el Microproyecto 2 de Machine Learning No Supervisado | Maestría en Inteligencia Artificial Aplicada (MAIA) | Universidad de los Andes<br>"
    "Autores: <b>Rafael Albarracín</b> & <b>Juan Carlos Ramírez</b> | Año: 2026"
    "</div>",
    unsafe_allow_html=True
)
