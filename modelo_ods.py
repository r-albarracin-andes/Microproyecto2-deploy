# ==============================================================================
# Módulo de Ingeniería y Despliegue: Transformadores y Pipeline ODS
# Maestría en Inteligencia Artificial Aplicada (MAIA) - Universidad de los Andes
# Microproyecto 2: Clasificación de Textos en los Objetivos de Desarrollo Sostenible
# Autores: Rafael Albarracin (r.albarracin@uniandes.edu.co)
#          Juan Carlos Ramirez (jc.ramirezg1@uniandes.edu.co)
# ==============================================================================
import os
import re
import sys
import string
import unicodedata
import platform
import pathlib
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import spacy
import matplotlib.pyplot as plt

import importlib

# ==============================================================================
# 0. Parche de Compatibilidad Multiplataforma (Windows <-> Linux / Azure App Service)
# ==============================================================================
def aplicar_parche_compatibilidad_pathlib():
    """
    Asegura compatibilidad de deserialización multiplataforma para objetos pathlib
    serializados con joblib/pickle entre Windows y Linux (Azure Web Apps, Docker).

    Resuelve:
      1. "cannot instantiate 'WindowsPath' on your system" al cargar en entornos Linux
         artefactos entrenados y serializados en Windows.
      2. "ModuleNotFoundError: No module named 'pathlib._local'" cuando un modelo
         serializado en Python 3.12+ se ejecuta en Python 3.11 o inferior.
    """
    import pathlib

    # 1. Garantizar resolución del submódulo pathlib._local de Python 3.12+
    if 'pathlib._local' not in sys.modules:
        try:
            importlib.import_module('pathlib._local')
        except ImportError:
            sys.modules['pathlib._local'] = pathlib

    # 2. Asignar alias dinámico según el sistema operativo de destino
    if platform.system() != 'Windows' or os.name != 'nt':
        # En Linux / macOS / Azure Web App: mapear WindowsPath a PosixPath
        pathlib.WindowsPath = pathlib.PosixPath
        if hasattr(pathlib, '_local'):
            pathlib._local.WindowsPath = pathlib.PosixPath
    else:
        # En Windows: mapear PosixPath a WindowsPath
        pathlib.PosixPath = pathlib.WindowsPath
        if hasattr(pathlib, '_local'):
            pathlib._local.PosixPath = pathlib.WindowsPath

# Ejecutar el parche de compatibilidad inmediatamente al importar el módulo
aplicar_parche_compatibilidad_pathlib()

# ==============================================================================
# Catálogos Oficiales de los Objetivos de Desarrollo Sostenible (ONU - Agenda 2030)
# ==============================================================================
diccionario_ods = {
    1: 'Fin de la pobreza',
    2: 'Hambre cero',
    3: 'Salud y bienestar',
    4: 'Educación de calidad',
    5: 'Igualdad de género',
    6: 'Agua limpia y saneamiento',
    7: 'Energía asequible y no contaminante',
    8: 'Trabajo decente y crecimiento económico',
    9: 'Industria, innovación e infraestructura',
    10: 'Reducción de las desigualdades',
    11: 'Ciudades y comunidades sostenibles',
    12: 'Producción y consumo responsables',
    13: 'Acción por el clima',
    14: 'Vida submarina',
    15: 'Vida de ecosistemas terrestres',
    16: 'Paz, justicia e instituciones sólidas',
    17: 'Alianzas para lograr los objetivos'
}

# Paleta cromática oficial de las Naciones Unidas para los ODS
colores_ods = {
    1: '#E5243B',  2: '#DDA63A',  3: '#4C9F38',  4: '#C5192D',
    5: '#FF3A21',  6: '#26BDE2',  7: '#FCC30B',  8: '#A21942',
    9: '#FD6925',  10: '#DD1367', 11: '#FD9D24', 12: '#BF8B2E',
    13: '#3F7E44', 14: '#0A97D9', 15: '#56C02B', 16: '#00689D',
    17: '#19486A'
}

# Iconos representativos de cada ODS
iconos_ods = {
    1: '🪙', 2: '🌾', 3: '🏥', 4: '📚',
    5: '⚖️', 6: '💧', 7: '☀️', 8: '💼',
    9: '🏭', 10: '🤝', 11: '🏙️', 12: '🔄',
    13: '🌱', 14: '🐟', 15: '🌳', 16: '🕊️',
    17: '🌐'
}


# ==============================================================================
# 1. Transformador Unificado de Preprocesamiento NLP
# ==============================================================================
class PreprocesadorTextosODS(BaseEstimator, TransformerMixin):
    """
    Transformador de Scikit-Learn para el preprocesamiento de textos del corpus ODS.
    
    Recibe directamente las variables predictoras X (DataFrame, Series o array) y opcionalmente y,
    ejecutando en memoria la secuencia óptima de limpieza morfológica, normalización Unicode,
    filtrado de stopwords y lematización en español con spaCy (o SnowballStemmer como respaldo).
    """
    def __init__(self,
                 columna_texto='textos',
                 columna_salida='textos_limpios',
                 columna_tokens='tokens',
                 retornar_texto_plano=False,
                 convertir_minusculas=True,
                 eliminar_puntuacion=True,
                 remover_acentos=True,
                 eliminar_numeros=True,
                 limpiar_artefactos=True,
                 tokenizar=True,
                 longitud_minima_token=2,
                 eliminar_stopwords=True,
                 stop_words_adicionales=None,
                 lematizar=True,
                 metodo_lematizacion='spacy',
                 mostrar_reportes=False):
        self.columna_texto = columna_texto
        self.columna_salida = columna_salida
        self.columna_tokens = columna_tokens
        self.retornar_texto_plano = retornar_texto_plano
        self.convertir_minusculas = convertir_minusculas
        self.eliminar_puntuacion = eliminar_puntuacion
        self.remover_acentos = remover_acentos
        self.eliminar_numeros = eliminar_numeros
        self.limpiar_artefactos = limpiar_artefactos
        self.tokenizar = tokenizar
        self.longitud_minima_token = longitud_minima_token
        self.eliminar_stopwords = eliminar_stopwords
        self.stop_words_adicionales = stop_words_adicionales
        self.lematizar = lematizar
        self.metodo_lematizacion = metodo_lematizacion
        self.mostrar_reportes = mostrar_reportes

        # Carga y normalización Unicode del catálogo de Stop Words en español
        try:
            sw_base = set(stopwords.words('spanish'))
        except Exception:
            nltk.download('stopwords', quiet=True)
            sw_base = set(stopwords.words('spanish'))
            
        if self.remover_acentos:
            sw_normalizadas = set()
            for w in sw_base:
                w_norm = unicodedata.normalize('NFD', w)
                w_norm = "".join([c for c in w_norm if unicodedata.category(c) != 'Mn'])
                sw_normalizadas.add(w_norm)
            sw_base = sw_normalizadas

        if self.stop_words_adicionales:
            for w in self.stop_words_adicionales:
                w_str = str(w).lower()
                if self.remover_acentos:
                    w_str = unicodedata.normalize('NFD', w_str)
                    w_str = "".join([c for c in w_str if unicodedata.category(c) != 'Mn'])
                sw_base.add(w_str)

        self.stopwords_ = sw_base
        
        # Inicialización del motor morfológico (spaCy con fallback a SnowballStemmer)
        self.nlp_ = None
        self.stemmer_ = None
        if self.lematizar:
            if self.metodo_lematizacion == 'spacy':
                try:
                    self.nlp_ = spacy.load('es_core_news_sm', disable=['parser', 'ner'])
                    # Sanitizar ruta interna para evitar acoplamiento a SO específico
                    if hasattr(self.nlp_, '_path'):
                        self.nlp_._path = None
                except Exception:
                    self.stemmer_ = SnowballStemmer('spanish')
            else:
                self.stemmer_ = SnowballStemmer('spanish')

    def fit(self, X, y=None):
        """Método fit conforme al estándar de Scikit-Learn."""
        if y is not None:
            self.y_ = np.array(y)
        return self

    def _limpiar_texto(self, texto):
        """Aplica limpieza de caracteres especiales, acentos, números y puntuación."""
        if not isinstance(texto, str):
            texto = str(texto) if pd.notnull(texto) else ""

        # Limpieza de artefactos de traducción y caracteres especiales
        if self.limpiar_artefactos:
            texto = texto.replace('\ufffd', ' ')
            texto = re.sub(r'[\r\n\t]+', ' ', texto)

        # Conversión a minúsculas
        if self.convertir_minusculas:
            texto = texto.lower()

        # Remoción de acentos (tildes) mediante Unicode NFD
        if self.remover_acentos:
            texto = unicodedata.normalize('NFD', texto)
            texto = "".join([c for c in texto if unicodedata.category(c) != 'Mn'])

        # Eliminación de dígitos numéricos
        if self.eliminar_numeros:
            texto = re.sub(r'\b\d+\b|\d+', ' ', texto)

        # Eliminación de signos de puntuación
        if self.eliminar_puntuacion:
            puntuacion = string.punctuation + '¿¡«»“”‘’—–…'
            texto = texto.translate(str.maketrans({c: ' ' for c in puntuacion}))

        # Normalización de espacios múltiples
        texto = re.sub(r'\s+', ' ', texto).strip()
        return texto

    def _tokenizar(self, texto_limpio):
        """Segmenta en tokens aplicando longitud mínima."""
        if not isinstance(texto_limpio, str):
            return []
        return [token for token in texto_limpio.split() if len(token) >= self.longitud_minima_token]

    def _eliminar_stopwords(self, tokens):
        """Filtra stop words normalizadas de la lista de tokens."""
        if not isinstance(tokens, list):
            return []
        return [token for token in tokens if token not in self.stopwords_]

    def _lematizar_tokens(self, lista_textos_filtrados):
        """Lematiza textos procesados en lotes optimizados con spaCy."""
        lista_lemas = []
        # Reintento perezoso de inicialización de spaCy si nlp_ no fue deserializado
        if self.nlp_ is None and self.lematizar and self.metodo_lematizacion == 'spacy':
            try:
                self.nlp_ = spacy.load('es_core_news_sm', disable=['parser', 'ner'])
                if hasattr(self.nlp_, '_path'):
                    self.nlp_._path = None
            except Exception:
                pass

        if self.nlp_ is not None:
            for doc in self.nlp_.pipe(lista_textos_filtrados, batch_size=256):
                lemas_doc = []
                for token in doc:
                    lema = token.lemma_.lower()
                    if self.remover_acentos:
                        lema = unicodedata.normalize('NFD', lema)
                        lema = "".join([c for c in lema if unicodedata.category(c) != 'Mn'])
                    if len(lema) >= self.longitud_minima_token:
                        lemas_doc.append(lema)
                lista_lemas.append(lemas_doc)
        else:
            stemmer = self.stemmer_ if self.stemmer_ else SnowballStemmer('spanish')
            for texto in lista_textos_filtrados:
                lemas_doc = [
                    stemmer.stem(token) for token in texto.split()
                    if len(token) >= self.longitud_minima_token
                ]
                lista_lemas.append(lemas_doc)
        return lista_lemas

    def transform(self, X, y=None):
        """Ejecuta la secuencia integral de transformaciones sobre los datos."""
        if isinstance(X, pd.DataFrame):
            df_out = X.copy()
        elif isinstance(X, pd.Series):
            df_out = pd.DataFrame({self.columna_texto: X.copy()})
        else:
            df_out = pd.DataFrame({self.columna_texto: list(X)})

        # Si ya contiene la columna preprocesada limpia y completa, evitar recomputar
        if self.columna_salida in df_out.columns and df_out[self.columna_salida].notnull().all():
            if self.retornar_texto_plano:
                return df_out[self.columna_salida]
            return df_out

        # Asegurar coincidencia de la columna de texto
        if self.columna_texto not in df_out.columns:
            col_origen = df_out.columns[0]
            df_out[self.columna_texto] = df_out[col_origen]

        # Si se suministra la etiqueta 'y', integrarla para consistencia
        if y is not None:
            if isinstance(y, (pd.Series, np.ndarray, list)):
                df_out['ODS'] = np.array(y)

        # Limpieza morfológica de textos
        df_out[self.columna_salida] = df_out[self.columna_texto].apply(self._limpiar_texto)

        # Tokenización
        if self.tokenizar:
            df_out[self.columna_tokens] = df_out[self.columna_salida].apply(self._tokenizar)

        # Eliminación de stop words
        if self.eliminar_stopwords and self.tokenizar:
            df_out[self.columna_tokens] = df_out[self.columna_tokens].apply(self._eliminar_stopwords)
            df_out[self.columna_salida] = df_out[self.columna_tokens].apply(lambda toks: " ".join(toks))

        # Lematización léxica
        if self.lematizar:
            textos_para_lematizar = df_out[self.columna_salida].tolist()
            lemas_resultado = self._lematizar_tokens(textos_para_lematizar)
            df_out[self.columna_tokens] = lemas_resultado
            df_out[self.columna_salida] = [" ".join(lemas) for lemas in lemas_resultado]

        if self.mostrar_reportes:
            print(f"-> Preprocesamiento completado exitosamente sobre {len(df_out):,} registros.")

        # Retornar Serie de texto plano si se invoca dentro de un Pipeline unificado
        if self.retornar_texto_plano:
            return df_out[self.columna_salida]
            
        return df_out


# ==============================================================================
# 2. Transformador de Reducción Dimensional LSA y Extracción de Tópicos
# ==============================================================================
class TransformadorLSATopicos(BaseEstimator, TransformerMixin):
    """
    Transformador para Análisis Semántico Latente (LSA) y Extracción de Tópicos.
    
    Integra de forma autónoma:
      1. Búsqueda y selección del mejor k (número de componentes) dentro de fit() 
         según la máxima varianza explicada acumulada.
      2. Método de visualización gráfica (graficar_varianza).
      3. Método de impresión en consola de los términos de mayor peso por componente (imprimir_topicos).
    """
    def __init__(self, candidatos_k=(10, 15, 20), n_iter=15, random_state=77):
        self.candidatos_k = list(candidatos_k)
        self.n_iter = n_iter
        self.random_state = random_state
        
        # Atributos aprendidos durante fit()
        self.mejor_k_ = 20
        self.svd_ = None
        self.mejor_modelo_ = None
        self.components_ = None
        self.explained_variance_ratio_ = None
        self.varianza_total_explicada_ = None
        self.resultados_exploracion_ = []
        self.df_exploracion_ = None
        self.modelos_k_ = {}

    def fit(self, X, y=None):
        """
        Ajusta TruncatedSVD para cada k candidato, evalúa la varianza explicada
        acumulada y selecciona automáticamente el mejor modelo.
        """
        self.resultados_exploracion_ = []
        self.modelos_k_ = {}
        
        for k in self.candidatos_k:
            svd_k = TruncatedSVD(
                n_components=k,
                n_iter=self.n_iter,
                random_state=self.random_state
            )
            svd_k.fit(X)
            
            var_acum = float(np.sum(svd_k.explained_variance_ratio_) * 100)
            var_ult = float(svd_k.explained_variance_ratio_[-1] * 100)
            
            info_k = {
                'Componentes (k)': k,
                'Varianza Explicada Acumulada (%)': round(var_acum, 2),
                'Varianza Última Componente (%)': round(var_ult, 3),
                'Dimensiones Reducidas': f"{X.shape[0]} x {k}",
                'var_indiv': svd_k.explained_variance_ratio_ * 100,
                'svd_model': svd_k
            }
            self.resultados_exploracion_.append(info_k)
            self.modelos_k_[k] = info_k

        # Estructuración de la tabla comparativa de exploración
        self.df_exploracion_ = pd.DataFrame([
            {
                'Componentes (k)': r['Componentes (k)'],
                'Varianza Explicada Acumulada (%)': r['Varianza Explicada Acumulada (%)'],
                'Varianza Última Componente (%)': r['Varianza Última Componente (%)'],
                'Dimensiones Reducidas': r['Dimensiones Reducidas']
            }
            for r in self.resultados_exploracion_
        ])

        # Selección automática del mejor k según máxima varianza explicada acumulada
        mejor_registro = max(self.resultados_exploracion_, key=lambda x: x['Varianza Explicada Acumulada (%)'])
        self.mejor_k_ = mejor_registro['Componentes (k)']
        self.svd_ = mejor_registro['svd_model']
        self.mejor_modelo_ = self.svd_
        self.components_ = self.svd_.components_
        self.explained_variance_ratio_ = self.svd_.explained_variance_ratio_
        self.varianza_total_explicada_ = np.sum(self.explained_variance_ratio_)
        
        return self

    def transform(self, X):
        """
        Proyecta la matriz en el espacio latente del mejor modelo SVD seleccionado.
        Compatible con los atributos svd_ y mejor_modelo_.
        """
        modelo = getattr(self, 'svd_', None) or getattr(self, 'mejor_modelo_', None)
        if modelo is None:
            raise RuntimeError("El transformador LSA debe ser ajustado con fit() antes de llamar a transform().")
        return modelo.transform(X)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def graficar_varianza(self):
        """Genera gráficas de varianza explicada individual y acumulada."""
        if self.modelos_k_ is None or len(self.modelos_k_) == 0:
            raise RuntimeError("Debe ajustar el transformador con fit() antes de graficar.")
            
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        best_k = self.mejor_k_
        var_indiv_optima = self.modelos_k_[best_k]['var_indiv']
        componentes_idx = np.arange(1, best_k + 1)

        axes[0].bar(componentes_idx, var_indiv_optima, color='#1f77b4', alpha=0.8, edgecolor='black')
        axes[0].set_title(f'Varianza Explicada Individual por Componente (LSA k={best_k})', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Índice de la Componente Latente', fontsize=11)
        axes[0].set_ylabel('Varianza Explicada (%)', fontsize=11)
        axes[0].set_xticks(componentes_idx)
        axes[0].grid(axis='y', linestyle='--', alpha=0.6)

        var_acum_optima = np.cumsum(var_indiv_optima)
        axes[1].plot(componentes_idx, var_acum_optima, marker='o', color='#d62728', linewidth=2, markersize=6)
        for k_val in self.candidatos_k:
            v_val = var_acum_optima[k_val - 1]
            axes[1].axvline(x=k_val, color='gray', linestyle=':', alpha=0.7)
            axes[1].scatter([k_val], [v_val], s=100, zorder=5)
            axes[1].annotate(f"k={k_val}: {v_val:.2f}%", (k_val, v_val), textcoords="offset points", xytext=(-25, 10), fontweight='bold')

        axes[1].set_title('Varianza Explicada Acumulada vs. Número de Componentes (Pipeline)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Número de Componentes (k)', fontsize=11)
        axes[1].set_ylabel('Varianza Acumulada (%)', fontsize=11)
        axes[1].set_xticks(componentes_idx)
        axes[1].grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.show()

    def imprimir_topicos(self, feature_names, n_top_words=10, n_componentes=None):
        """Imprime en consola los términos de mayor peso para cada componente."""
        if self.components_ is None:
            raise RuntimeError("Debe ajustar el transformador con fit() antes de imprimir los tópicos.")
            
        n_comp = n_componentes if n_componentes is not None else len(self.components_)
        print("=" * 85)
        print(f"TÉRMINOS CON MAYOR PESO POR COMPONENTE LATENTE (LSA k={self.mejor_k_})")
        print("=" * 85)
        
        for idx in range(min(n_comp, len(self.components_))):
            comp = self.components_[idx]
            indices_ordenados = np.argsort(comp)[::-1][:n_top_words]
            palabras_peso = [f"{feature_names[i]} ({comp[i]:.3f})" for i in indices_ordenados]
            var_pct = self.explained_variance_ratio_[idx] * 100
            print(f"\nComponente {idx + 1:2d} (Varianza explicada: {var_pct:.3f}%):")
            print(f"  {', '.join(palabras_peso)}")


# ==============================================================================
# 3. Transformador / Estimador de Clasificación Probabilística ODS
# ==============================================================================
class ClasificadorProbabilisticoODS(BaseEstimator, ClassifierMixin, TransformerMixin):
    """
    Transformador y Estimador de Clasificación Supervisada para el corpus ODS.
    
    Ejecuta dentro de su método fit() la búsqueda del hiperparámetro de regularización C
    mediante validación cruzada estratificada (StratifiedKFold, 5 folds) optimizando el F1-Macro.
    Permite obtener predicciones discretas (predict), probabilidades completas (predict_proba),
    transformación a espacio de probabilidades (transform) y reportes de pertenencia múltiple (predecir_top_k).
    """
    def __init__(self, candidatos_C=(0.01, 0.1, 1.0, 10.0, 100.0), penalty='l2',
                 solver='lbfgs', max_iter=1000, class_weight='balanced', random_state=77):
        self.candidatos_C = list(candidatos_C)
        self.penalty = penalty
        self.solver = solver
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.random_state = random_state
        
        # Atributos aprendidos en fit
        self.mejor_C_ = 10.0
        self.mejor_score_ = None
        self.modelo_ = None
        self.resultados_busqueda_ = []
        self.df_busqueda_ = None
        self.classes_ = None

    def fit(self, X, y):
        """Ejecuta la búsqueda de hiperparámetros de C y ajusta el clasificador definitivo."""
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        self.resultados_busqueda_ = []
        
        for c_val in self.candidatos_C:
            clf_temp = LogisticRegression(
                C=c_val,
                penalty=self.penalty,
                solver=self.solver,
                max_iter=self.max_iter,
                class_weight=self.class_weight,
                random_state=self.random_state
            )
            scores = cross_val_score(clf_temp, X, y, cv=cv, scoring='f1_macro', n_jobs=-1)
            mean_f1 = float(np.mean(scores) * 100)
            std_f1 = float(np.std(scores) * 100)
            
            res_c = {
                'Parámetro C': c_val,
                'F1-Macro Medio (%)': round(mean_f1, 2),
                'Desviación Estándar (%)': round(std_f1, 2)
            }
            self.resultados_busqueda_.append(res_c)
            
        self.df_busqueda_ = pd.DataFrame(self.resultados_busqueda_)
        mejor = max(self.resultados_busqueda_, key=lambda x: x['F1-Macro Medio (%)'])
        self.mejor_C_ = mejor['Parámetro C']
        self.mejor_score_ = mejor['F1-Macro Medio (%)']
        
        self.modelo_ = LogisticRegression(
            C=self.mejor_C_,
            penalty=self.penalty,
            solver=self.solver,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            random_state=self.random_state
        )
        self.modelo_.fit(X, y)
        self.classes_ = self.modelo_.classes_
        return self

    def predict(self, X):
        """Retorna las clases dominantes predichas."""
        if self.modelo_ is None:
            raise RuntimeError("Debe llamar a fit antes de predict.")
        return self.modelo_.predict(X)

    def predict_proba(self, X):
        """Retorna las probabilidades de pertenencia para cada uno de los ODS."""
        if self.modelo_ is None:
            raise RuntimeError("Debe llamar a fit antes de predict_proba.")
        return self.modelo_.predict_proba(X)

    def transform(self, X):
        """Retorna la matriz de probabilidades calibradas de pertenencia a cada ODS."""
        return self.predict_proba(X)

    def predecir_top_k(self, X, k=3, umbral_pertenencia=0.15, diccionario_ods_ref=None):
        """Retorna la pertenencia probabilística a uno o varios ODS para cada texto."""
        probas = self.predict_proba(X)
        dicc = diccionario_ods_ref or diccionario_ods
        resultados = []
        for proba_fila in probas:
            indices_ordenados = np.argsort(proba_fila)[::-1]
            top_k_indices = indices_ordenados[:k]
            
            top_ods = []
            for idx in top_k_indices:
                ods_cod = int(self.classes_[idx])
                ods_nom = dicc.get(ods_cod, f"ODS {ods_cod}")
                top_ods.append({
                    'ODS': ods_cod,
                    'Nombre': ods_nom,
                    'Probabilidad (%)': round(float(proba_fila[idx]) * 100, 2)
                })
                
            ods_secundarios = [
                int(self.classes_[i]) for i in indices_ordenados[1:] 
                if float(proba_fila[i]) >= umbral_pertenencia
            ]
            
            dom_idx = indices_ordenados[0]
            dom_cod = int(self.classes_[dom_idx])
            dom_nom = dicc.get(dom_cod, f"ODS {dom_cod}")
            
            resultados.append({
                'ODS_Dominante': dom_cod,
                'Nombre_Dominante': dom_nom,
                'Probabilidad_Dominante (%)': round(float(proba_fila[dom_idx]) * 100, 2),
                'Top_ODS': top_ods,
                'ODS_Multiples': [dom_cod] + ods_secundarios
            })
        return resultados


# ==============================================================================
# 4. Transformador Final de Demostración Probabilística sobre Textos Aleatorios
# ==============================================================================
class DemostradorTextosAleatoriosODS(BaseEstimator, ClassifierMixin, TransformerMixin):
    """
    Transformador de Demostración Probabilística sobre Textos Aleatorios en español.
    Se integra al final del Pipeline y evalúa un banco de textos coherentes sobre los ODS.
    """
    def __init__(self, n_textos=5, random_state=77, k=3, umbral_pertenencia=0.15, diccionario_ods=None, pipeline=None):
        self.n_textos = n_textos
        self.random_state = random_state
        self.k = k
        self.umbral_pertenencia = umbral_pertenencia
        self.diccionario_ods = diccionario_ods
        self.pipeline = pipeline
        
        # Banco temático de textos coherentes en español sobre las metas de los ODS
        self.banco_textos = [
            "La construcción de plantas de tratamiento de aguas residuales y el acceso a redes de saneamiento básico reducen la propagación de bacterias y protegen la salud pública infantil.",
            "Fortalecer los programas de formación pedagógica continua y dotar de aulas interactivas a las escuelas rurales asegura una educación equitativa y de calidad para todos.",
            "Invertir en parques solares y redes de energía eólica acelera la descarbonización económica y promueve el uso de fuentes renovables y no contaminantes.",
            "Eliminar las brechas salariales y garantizar la paridad femenina en cargos directivos y parlamentarios fortalece la igualdad de género y la justicia laboral.",
            "Garantizar el acceso universal a la atención médica primaria, vacunas esenciales y centros de salud mental previene la mortalidad materna y protege el bienestar general.",
            "Combatir la impunidad, reforzar la independencia judicial y proteger a los defensores de derechos humanos consolidan instituciones transparentes y un estado de derecho sólido.",
            "Otorgar microcréditos productivos y asistencia técnica a pequeños agricultores familiares erradica la pobreza extrema y crea oportunidades de empleo digno.",
            "Fomentar el transporte público eléctrico masivo y la gestión integral de residuos sólidos urbanos transforma las ciudades en espacios sostenibles e inclusivos.",
            "Prohibir la pesca indiscriminada de arrastre y frenar el vertido de plásticos en las costas preserva los arrecifes de coral y la biodiversidad marina.",
            "Reforestar cuencas hídricas con especies de árboles autóctonos combate la desertificación y protege la fauna silvestre de los ecosistemas terrestres."
        ]
        
        self.classes_ = None
        self.is_fitted_ = False
        self.resultados_demostracion_ = []

    def _obtener_pipeline_padre(self):
        """Detecta automáticamente el Pipeline que contiene a este transformador."""
        if self.pipeline is not None:
            return self.pipeline
        frame = sys._getframe(2)
        while frame:
            obj = frame.f_locals.get('self')
            if isinstance(obj, Pipeline):
                return obj
            frame = frame.f_back
        return None

    def fit(self, X, y=None):
        pipe = self._obtener_pipeline_padre()
        if pipe is not None:
            self.pipeline_ = pipe
            clf = pipe.named_steps['clasificador']
            self.classes_ = clf.classes_
        else:
            self.classes_ = np.array(range(1, 17))
        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Retorna la matriz de probabilidades calibradas del paso previo intacta."""
        return X

    def predict(self, X):
        """Retorna las clases dominantes predichas."""
        classes = self.classes_ if self.classes_ is not None else np.array(range(1, 17))
        return classes[np.argmax(X, axis=1)]

    def predict_proba(self, X):
        """Retorna la matriz de probabilidades calibradas."""
        return X


# ==============================================================================
# 5. Registro Dinámico en __main__ y Carga del Pipeline Serializado
# ==============================================================================
def registrar_clases_en_main():
    """
    Registra las clases del pipeline en sys.modules['__main__'] para garantizar
    la deserialización transparente con joblib/pickle de artefactos exportados
    desde entornos interactivos (Jupyter Notebooks).
    """
    clases = {
        'PreprocesadorTextosODS': PreprocesadorTextosODS,
        'TransformadorLSATopicos': TransformadorLSATopicos,
        'ClasificadorProbabilisticoODS': ClasificadorProbabilisticoODS,
        'DemostradorTextosAleatoriosODS': DemostradorTextosAleatoriosODS
    }
    
    # Inyectar en el módulo __main__ actual
    if '__main__' in sys.modules:
        main_mod = sys.modules['__main__']
        for nombre, cls in clases.items():
            setattr(main_mod, nombre, cls)

# Ejecución automática al importar el módulo
registrar_clases_en_main()


def cargar_pipeline_ods(ruta_archivo=None):
    """
    Carga el pipeline serializado 'pipeline_ods.joblib' garantizando la resolución
    de dependencias de clases en __main__, compatibilidad multiplataforma de pathlib
    (Windows <-> Linux en Azure Web App) y resolviendo rutas relativas flexibles.
    """
    aplicar_parche_compatibilidad_pathlib()
    registrar_clases_en_main()

    if ruta_archivo and os.path.exists(ruta_archivo):
        return joblib.load(ruta_archivo)

    directorio_modulo = os.path.dirname(os.path.abspath(__file__))
    candidatos_rutas = [
        os.path.join(directorio_modulo, 'model', 'pipeline_ods.joblib'),
        os.path.join(directorio_modulo, 'pipeline_ods.joblib'),
        os.path.join('model', 'pipeline_ods.joblib'),
        os.path.join('Microproyecto2', 'model', 'pipeline_ods.joblib'),
        'pipeline_ods.joblib'
    ]

    for ruta in candidatos_rutas:
        if os.path.exists(ruta):
            return joblib.load(ruta)

    raise FileNotFoundError(
        f"No se encontró el pipeline serializado en ninguna de las rutas esperadas:\n"
        + "\n".join(f"  - {r}" for r in candidatos_rutas)
    )


# ==============================================================================
# 6. Función de Inferencia de Alto Nivel para Despliegue
# ==============================================================================
def predecir_ods_con_probabilidad(pipeline, textos, k=3, umbral_pertenencia=0.15, diccionario_ods=None):
    """
    Función de inferencia integral para clasificar textos individuales, listas o DataFrames
    a través del pipeline completo, retornando probabilidades calibradas y pertenencia múltiple.
    Ideal para el despliegue interactivo en Streamlit y evaluación por lotes.
    """
    if isinstance(textos, str):
        df_in = pd.DataFrame({'textos': [textos]})
    elif isinstance(textos, list):
        df_in = pd.DataFrame({'textos': textos})
    elif isinstance(textos, pd.Series):
        df_in = pd.DataFrame({'textos': textos})
    elif isinstance(textos, pd.DataFrame):
        df_in = textos
    else:
        df_in = pd.DataFrame({'textos': list(textos)})

    # Obtener probabilidades mediante predict_proba
    probas = pipeline.predict_proba(df_in)
    
    # Extraer catálogo de clases aprendidas
    if 'clasificador' in pipeline.named_steps:
        classes = pipeline.named_steps['clasificador'].classes_
    else:
        classes = getattr(pipeline, 'classes_', np.array(range(1, 17)))

    dicc = diccionario_ods or globals().get('diccionario_ods', {})

    resultados = []
    for proba_fila in probas:
        indices_ordenados = np.argsort(proba_fila)[::-1]
        top_k_indices = indices_ordenados[:k]
        
        top_ods = []
        for idx in top_k_indices:
            ods_cod = int(classes[idx])
            ods_nom = dicc.get(ods_cod, f"ODS {ods_cod}")
            top_ods.append({
                'ODS': ods_cod,
                'Nombre': ods_nom,
                'Probabilidad (%)': round(float(proba_fila[idx]) * 100, 2)
            })
            
        # ODS secundarios que superan el umbral de afinidad
        ods_secundarios = [
            int(classes[i]) for i in indices_ordenados[1:] 
            if float(proba_fila[i]) >= umbral_pertenencia
        ]
        dom_idx = indices_ordenados[0]
        dom_cod = int(classes[dom_idx])
        dom_nom = dicc.get(dom_cod, f"ODS {dom_cod}")
        
        resultados.append({
            'ODS_Dominante': dom_cod,
            'Nombre_Dominante': dom_nom,
            'Probabilidad_Dominante (%)': round(float(proba_fila[dom_idx]) * 100, 2),
            'Top_ODS': top_ods,
            'ODS_Multiples': [dom_cod] + ods_secundarios
        })
    return resultados
