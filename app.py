import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(page_title="NAE Risk Predictor", layout="wide")

# 2. CARGA DE RECURSOS Y CONSTANTES
@st.cache_resource
def cargar_recursos():
    modelo = joblib.load('modelo_ENO_rf.pkl')
    columnas = joblib.load('columnas_modelo.pkl')
    return modelo, columnas

try:
    modelo, columnas_entrenamiento = cargar_recursos()
except FileNotFoundError:
    st.error("❌ No se encontraron los archivos .pkl. Ejecuta primero tu script de entrenamiento.")
    st.stop()

# Medianas reales extraídas de BaseENOS_para_ML.csv
MEDIANAS = {
    'edad': 35.6,
    'LAB_V_num_CHOL': 162.0,
    'LAB_V_num_HDL': 40.0,
    'LAB_V_num_PLT': 217000.0,
    'LAB_V_num_GLUC': 90.0,
    'LAB_V_num_TRIG': 102.0,
    'LAB_V_num_AST': 24.0,
    'LAB_V_num_ALT': 24.0,
    'TyG': 8.4,
    'FIB4': 0.81,
    'tiempo_seguimiento': 2674.5
}

# 3. INTERFAZ DE USUARIO
st.title("Non-AIDS-Defining Event (NAE) Prediction Tool")
st.markdown("This calculator uses a **Random Forest** model to estimate the probability of developing a Non-AIDS Event.")

st.divider()

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
        alcohol = st.selectbox("Alcohol", ["0", "1", "Unknown"], format_func=lambda x: "No" if x == "0" else ("Yes" if x == "1" else "Unknown"))
        smoking = st.selectbox("Smoking", ["0", "1", "Unknown"], format_func=lambda x: "No" if x == "0" else ("Yes" if x == "1" else "Unknown"))

with tab2:
    c1, c2, c3 = st.columns(3)
    with c1:
        chol = st.number_input("Cholesterol (mg/dL)", 0, 500, 162)
        gluc = st.number_input("Glucose (mg/dL)", 0.0, 500.0, 90.0)
        hdl = st.number_input("HDL (mg/dL)", 0, 180, 40)
    with c2:
        trig = st.number_input("Triglycerides (mg/dL)", 0, 500, 102)
        # IMPORTANTE: Escala corregida a miles para coincidir con el modelo
        plt = st.number_input("Platelets (cells/µL)", 0, 1000000, 217000)
        ast = st.number_input("AST (U/L)", 0, 500, 24)
    with c3:
        alt = st.number_input("ALT (U/L)", 0, 500, 24)
        # Cálculos visuales inmediatos
        tyg_visual = np.log((trig * gluc) / 2)
        fib4_visual = (edad * ast) / (plt * np.sqrt(alt)) if plt > 0 and alt > 0 else 0
        st.write(f"**Calculated TyG:** {tyg_visual:.2f}")
        st.write(f"**Calculated FIB-4:** {fib4_visual:.2f}")

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
        year_art = st.selectbox("Periodo inicio ART", ['2004–2007', '2008–2011', '2012–2015', '2016–2019', '2020–2024'])
        seguimiento_anios = st.number_input("Follow-up time (years)", 0.5, 25.0, 7.0)
        seguimiento_dias = seguimiento_anios * 365.25

# 4. PROCESAMIENTO Y PREDICCIÓN
st.divider()
if st.button("CALCULATE RISK", type="primary", use_container_width=True):
    
    # Validación de seguridad para fórmulas
    plt_s = plt if plt > 0 else MEDIANAS['LAB_V_num_PLT']
    alt_s = alt if alt > 0 else MEDIANAS['LAB_V_num_ALT']
    
    # Scores finales
    tyg_f = np.log((trig * gluc) / 2)
    fib4_f = (edad * ast) / (plt_s * np.sqrt(alt_s))
    
    input_dict = {
        'edad': edad, 'LAB_V_num_CHOL': chol, 'LAB_V_num_HDL': hdl, 
        'TyG': tyg_f, 'FIB4': fib4_f, 'tiempo_seguimiento': seguimiento_dias, 
        'GENDER': gender, 'MODE_cat': mode, 'Country_origin': country, 
        'EDU_cat_label': edu, 'VHC_ab': vhc, 'VHB_ag': vhb, 
        'carga_inicial_cat': carga, 'CD4_cat': cd4, 'ALCOHOL': alcohol, 
        'SMOKING': smoking, 'Year_of_ART_initiation': year_art, 
        'tipo_primerTAR': tar, 'AIDS_Y': aids
    }
    
    df_input = pd.DataFrame([input_dict])
    
    # Forzar tipo string para categóricas
    cols_cat = ['GENDER', 'MODE_cat', 'Country_origin', 'EDU_cat_label', 'VHC_ab', 
                'VHB_ag', 'carga_inicial_cat', 'CD4_cat', 'ALCOHOL', 'SMOKING', 
                'Year_of_ART_initiation', 'tipo_primerTAR', 'AIDS_Y']
    df_input[cols_cat] = df_input[cols_cat].astype(str)
    
    # Dummies y alineación con entrenamiento
    df_dummies = pd.get_dummies(df_input)
    df_final = df_dummies.reindex(columns=columnas_entrenamiento, fill_value=0)
    
    # Predicción
    prob = modelo.predict_proba(df_final)[0][1]
    
    # Mostrar resultados
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
st.sidebar.info("This model was trained with data from HIV patients under follow-up.")
