import pandas as pd
import joblib

# Cargar el modelo y las columnas
modelo = joblib.load('modelo_ENO_rf.pkl')
columnas = joblib.load('columnas_modelo.pkl')

def predecir_paciente(input_dict):
    # Convertimos el diccionario de la interfaz a un DataFrame de una sola fila
    df = pd.DataFrame([input_dict])

    # Definir las variables exactas de tu estudio
    features_num = [
        'edad', 'LAB_V_num_CHOL', 'LAB_V_num_GLUC', 'LAB_V_num_HDL',
        'LAB_V_num_TRIG', 'LAB_V_num_PLT', 'LAB_V_num_AST', 'LAB_V_num_ALT',
        'TyG', 'FIB4', 'tiempo_seguimiento'
    ]

    features_cat = [
        'GENDER', 'MODE_cat', 'Country_origin', 'EDU_cat_label',
        'VHC_ab', 'VHB_ag', 'carga_inicial_cat', 'CD4_cat',
        'ALCOHOL', 'SMOKING', 'Year_of_ART_initiation', 'tipo_primerTAR',
        'AIDS_Y', 'DEATH_Y'
    ]

    # Tratamiento de nulos (Mediana para números, 'Unknown' para categorías)
    for col in features_num:
        if col in df.columns:
            # Nota: En la app, lo ideal es que no haya nulos porque el usuario los rellena,
            # pero esto es una red de seguridad.
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0) 

    for col in features_cat:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown').astype(str)

    # Transformar categorías a columnas (One-hot encoding)
    df_processed = pd.get_dummies(df)

    # ALINEAR COLUMNAS
    # Esto añade las columnas que faltan y elimina las que no estaban en el entrenamiento
    df_final = df_processed.reindex(columns=columnas, fill_value=0)

    # Ejecutar la predicción
    prob = modelo.predict_proba(df_final)[:, 1][0] # Probabilidad de "Sí ENO"
    pred = modelo.predict(df_final)[0]             # Clase (0 o 1)

    return prob, pred

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(page_title="Predictor Riesgo ENO", layout="wide")

# CARGA DE MODELO Y MAPA DE COLUMNAS
@st.cache_resource
def cargar_recursos():
    # Asegúrate de haber ejecutado antes el script que genera estos archivos
    modelo = joblib.load('modelo_ENO_rf.pkl')
    columnas = joblib.load('columnas_modelo.pkl')
    return modelo, columnas

try:
    modelo, columnas_entrenamiento = cargar_recursos()
except FileNotFoundError:
    st.error("❌ No se encontraron los archivos .pkl. Ejecuta primero tu script de entrenamiento.")
    st.stop()

# 2. INTERFAZ DE USUARIO
st.title("Herramienta de Predicción de Eventos No-SIDA (ENO)")
st.markdown("Esta calculadora utiliza un modelo de **Random Forest** para estimar la probabilidad de aparición de eventos no-SIDA.")

st.divider()

# Organización por Pestañas
tab1, tab2, tab3 = st.tabs(["Sociodemográficos", "Analítica y Scores", "Parámetros VIH"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        edad = st.number_input("Edad (años)", 18, 90, 35)
        gender = st.selectbox("Género", ["Hombre", "Mujer"])
        country = st.selectbox("País de origen", ["Spain", "No Spain"])
        edu = st.selectbox("Nivel Educativo", ["No or compulsory", "Upper secondary or university", "Unknown"])
    with c2:
        mode = st.selectbox("Modo de transmisión", ["Homo/Bisexual", "Heterosexual", "UDI", "Other/Unknown"])
        alcohol = st.selectbox("Consumo de Alcohol", ["0", "1", "Unknown"], help="1: Sí, 0: No")
        smoking = st.selectbox("Tabaquismo", ["0", "1", "Unknown"], help="1: Sí, 0: No")

with tab2:
    c1, c2, c3 = st.columns(3)
    with c1:
        chol = st.number_input("Colesterol (mg/dL)", 0, 500, 162)
        gluc = st.number_input("Glucosa (mg/dL)", 0.0, 500.0, 90.0)
        hdl = st.number_input("HDL (mg/dL)", 0, 180, 40)
    with c2:
        trig = st.number_input("Triglicéridos (mg/dL)", 0, 500, 102)
        plt = st.number_input("Plaquetas", 0, 1000000, 217000)
        ast = st.number_input("AST (U/L)", 0, 500, 24)
    with c3:
        alt = st.number_input("ALT (U/L)", 0, 500, 24)
        tyg = st.number_input("Índice TyG", 1.0, 12.0, 8.4)
        fib4 = st.number_input("FIB-4 Score", 0.0, 75.0, 0.8)

with tab3:
    c1, c2 = st.columns(2)
    with c1:
        cd4 = st.selectbox("Categoría CD4", ["≥200", "<200", "Unknown"])
        carga = st.selectbox("Carga Viral Inicial", ["<100.000", "≥100.000", "Unknown"])
        aids = st.selectbox("Evento SIDA previo (AIDS_Y)", ["No", "Si", "Desconocido"])
        vhc = st.selectbox("VHC (Hepatitis C)", ["Negativo", "Positivo", "Unknown"])
    with c2:
        vhb = st.selectbox("VHB (Hepatitis B)", ["Negativo", "Positivo", "Unknown"])
        tar = st.selectbox("Tipo primer TAR", ["2NRTI+1NNRTI", "2NRTI+1PI", "2NRTI+1II", "Other/Unknown"])
        year_art = st.selectbox("Periodo inicio ART", ['2004–2007', '2008–2011', '2012–2015', '2016–2019', '2020–2024'])
        seguimiento = st.number_input("Tiempo seguimiento (días)", 180, 8000, 2674)
        death = st.selectbox("Muerte (DEATH_Y)", ["No", "Si"])

# PROCESAMIENTO Y PREDICCIÓN
st.divider()
if st.button("CALCULAR RIESGO", type="primary", use_container_width=True):
    
    # Crear diccionario con nombres de columnas exactos del entrenamiento
    input_dict = {
        'edad': edad, 'LAB_V_num_CHOL': chol, 'LAB_V_num_GLUC': gluc, 
        'LAB_V_num_HDL': hdl, 'LAB_V_num_TRIG': trig, 'LAB_V_num_PLT': plt, 
        'LAB_V_num_AST': ast, 'LAB_V_num_ALT': alt, 'TyG': tyg, 'FIB4': fib4, 
        'tiempo_seguimiento': seguimiento, 'GENDER': gender, 'MODE_cat': mode,
        'Country_origin': country, 'EDU_cat_label': edu, 'VHC_ab': vhc, 
        'VHB_ag': vhb, 'carga_inicial_cat': carga, 'CD4_cat': cd4, 
        'ALCOHOL': alcohol, 'SMOKING': smoking, 'Year_of_ART_initiation': year_art, 
        'tipo_primerTAR': tar, 'AIDS_Y': aids, 'DEATH_Y': death
    }
    
    # Convertir a DataFrame y aplicar One-Hot Encoding
    df_input = pd.DataFrame([input_dict])
    
    # IMPORTANTE: Convertir a str las columnas que el modelo espera como categóricas
    cols_cat = ['GENDER', 'MODE_cat', 'Country_origin', 'EDU_cat_label', 'VHC_ab', 
                'VHB_ag', 'carga_inicial_cat', 'CD4_cat', 'ALCOHOL', 'SMOKING', 
                'Year_of_ART_initiation', 'tipo_primerTAR', 'AIDS_Y', 'DEATH_Y']
    df_input[cols_cat] = df_input[cols_cat].astype(str)
    
    # Generar dummies y alinear con el modelo
    df_dummies = pd.get_dummies(df_input)
    df_final = df_dummies.reindex(columns=columnas_entrenamiento, fill_value=0)
    
    # Predicción
    prob = modelo.predict_proba(df_final)[0][1]
    
    # Presentación de resultados
    st.subheader("Resultado de la Evaluación")
    
    col_score, col_text = st.columns([1, 2])
    
    with col_score:
        st.metric("Riesgo Estimado", f"{prob:.1%}")
        
    with col_text:
        if prob < 0.20:
            st.success("✅ RIESGO BAJO: El perfil clínico sugiere una baja probabilidad de eventos ENO.")
        elif prob < 0.45:
            st.warning("⚠️ RIESGO INTERMEDIO: Se recomienda seguimiento clínico estrecho.")
        else:
            st.error("🚨 RIESGO ALTO: El modelo identifica múltiples factores predictores de eventos ENO.")

st.sidebar.markdown("### Información del Modelo")
st.sidebar.info("Este modelo ha sido entrenado con datos de pacientes en seguimiento por VIH. "
                "La probabilidad mostrada es orientativa y no sustituye el juicio clínico.")
