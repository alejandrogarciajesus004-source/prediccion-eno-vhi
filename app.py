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
st.set_page_config(page_title="NAE Risk Predictor", layout="wide")

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
st.title("Non-AIDS-Defining Event (NAE) Prediction Tool")
st.markdown("This calculator uses a **Random Forest** model to estimate the probability of developing a Non-AIDS Event.")

st.divider()

# Organización por Pestañas
tab1, tab2, tab3 = st.tabs(["Sociodemographics", "Labs & Scores", "HIV Parameters"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        edad = st.number_input("Age (years)", 18, 90, 35)
        gender = st.selectbox("Gender", ["Hombre", "Mujer"], format_func=lambda x: "Male" if x == "Hombre" else "Female")
        country = st.selectbox("Country of origin", ["Spain", "No Spain"])
        edu = st.selectbox("Education level", ["No or compulsory", "Upper secondary or university", "Unknown"])
    with c2:
        mode = st.selectbox("Transmission Mode", ["Homo/Bisexual", "Heterosexual", "UDI", "Other/Unknown"])
        alcohol = st.selectbox("Alcohol", ["0", "1", "Unknown"], format_func=lambda x: "No" if x == "0" else ("Yes" if x == "1" else "Unknown"), help="Weekly alcohol consumption in Standard Drink Units (SDU). Does not drink = 0; SDU 1 beer, 1 glass of wine =1; SDU 1 shot of spirits, 1 mixed drink = 2 SDU. If your SDU is 0 select No. If else, select Yes")
        smoking = st.selectbox("Smoking", ["0", "1", "Unknown"], format_func=lambda x: "No" if x == "0" else ("Yes" if x == "1" else "Unknown"), help=" If you never smoke, select No. If you are an active smoker, smoke ocasionally or ex-smoker select yes")

with tab2:
    c1, c2, c3 = st.columns(3)
    with c1:
        chol = st.number_input("Cholesterol (mg/dL)", 0, 500, 162)
        gluc = st.number_input("Glucose (mg/dL)", 0.0, 500.0, 90.0)
        hdl = st.number_input("HDL (mg/dL)", 0, 180, 40)
    with c2:
        trig = st.number_input("Triglycerides (mg/dL)", 0, 500, 102)
        plt = st.number_input("Platelets ", 0, 1000, 217)
        ast = st.number_input("AST (U/L)", 0, 500, 24)
    with c3:
        alt = st.number_input("ALT (U/L)", 0, 500, 24)
        tyg_calc = np.log((trig * gluc) / 2)
        fib4_calc = (edad * ast) / (plt * np.sqrt(alt))
        st.write(f"**Calculated TyG:** {tyg_calc:.2f}")
        st.write(f"**Calculated FIB-4:** {fib4_calc:.2f}")

with tab3:
    c1, c2 = st.columns(2)
    with c1:
        cd4 = st.selectbox("CD4 category", ["≥200", "<200", "Unknown"])
        carga = st.selectbox("Initial viral load", ["<100.000", "≥100.000", "Unknown"])
        aids = st.selectbox("Previous AIDS event", ["No", "Si", "Desconocido"], format_func=lambda x: "Yes" if x == "Si" else ("No" if x == "No" else "Unknown"))
        vhc = st.selectbox("HCV (Hepatitis C)", ["Negativo", "Positivo", "Unknown"], format_func=lambda x: "Negative" if x == "Negativo" else ("Positive" if x == "Positivo" else "Unknown"))
    with c2:
        vhb = st.selectbox("HBV (Hepatitis B)", ["Negativo", "Positivo", "Unknown"], format_func=lambda x: "Negative" if x == "Negativo" else ("Positive" if x == "Positivo" else "Unknown"))
        tar = st.selectbox("First ART regimen", ["2NRTI+1NNRTI", "2NRTI+1PI", "2NRTI+1II", "Other/Unknown"])
        year_art = st.selectbox("Year of ART initiation", ['2004–2007', '2008–2011', '2012–2015', '2016–2019', '2020–2024'])
        seguimiento_anios = st.number_input("Follow-up time (years)", 0.5, 25.0, 7.0)
        seguimiento_dias = seguimiento_anios * 365.25
        

# PROCESAMIENTO Y PREDICCIÓN
st.divider()
if st.button("CALCULATE RISK", type="primary", use_container_width=True):
    
    # 1. Diccionario con nombres y valores EXACTOS
   input_dict = {
        'edad': edad, 
        'LAB_V_num_CHOL': chol, 
        'LAB_V_num_HDL': hdl, 
        'TyG': tyg_calc, 
        'FIB4': fib4_calc, 
        'tiempo_seguimiento': seguimiento_anios, 
        'LAB_V_num_PLT': plt * 1000, 
        'GENDER': gender, 
        'MODE_cat': mode,
        'Country_origin': country, 
        'EDU_cat_label': edu, 
        'VHC_ab': vhc, 
        'VHB_ag': vhb, 
        'carga_inicial_cat': carga, 
        'CD4_cat': cd4, 
        'ALCOHOL': alcohol, 
        'SMOKING': smoking, 
        'Year_of_ART_initiation': year_art, 
        'tipo_primerTAR': tar, 
        'AIDS_Y': aids,
        'DEATH_Y': 'No'
    }
    
    df_input = pd.DataFrame([input_dict])
    
    # 2. Lista de categóricas (Asegúrate de incluir DEATH_Y aquí también)
    cols_cat = ['GENDER', 'MODE_cat', 'Country_origin', 'EDU_cat_label', 'VHC_ab', 
                'VHB_ag', 'carga_inicial_cat', 'CD4_cat', 'ALCOHOL', 'SMOKING', 
                'Year_of_ART_initiation', 'tipo_primerTAR', 'AIDS_Y', 'DEATH_Y']
    
    df_input[cols_cat] = df_input[cols_cat].astype(str)
    
    # 3. Generar dummies
    df_dummies = pd.get_dummies(df_input)
    
    # 4. Alinear con las columnas del entrenamiento
    df_final = df_dummies.reindex(columns=columnas_entrenamiento, fill_value=0)
    
    # 5. Predicción
    prob = modelo.predict_proba(df_final)[0][1]
    
    # Presentación de resultados
    st.subheader("Evaluation result")
    
    col_score, col_text = st.columns([1, 2])
    
    with col_score:
        st.metric("Estimated risk", f"{prob:.1%}")
        
    with col_text:
        if prob < 0.20:
            st.success("✅ LOW RISK: The clinical profile suggests a low probability of NAE.")
        elif prob < 0.45:
            st.warning("⚠️ INTERMEDIATE RISK: Close clinical monitoring is recommended.")
        else:
            st.error("🚨 HIGH RISK: The model identifies multiple predictive factors for NAE.")

st.sidebar.markdown("### Model information")
st.sidebar.info("This model was trained with data from HIV patients under follow-up. "
                "The probability shown is for guidance and does not replace clinical judgment.")
