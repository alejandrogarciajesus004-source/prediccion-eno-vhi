import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN DE LA PÁGINA Y ESTILOS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="HIV Multi-Risk Predictor (NAE, Cancer & CVD)",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🩺 HIV Non-AIDS Event (NAE) Multi-Risk Calculator")
st.markdown("""
This clinical decision support tool estimates the **multi-risk profile** for patients living with HIV on ART, 
evaluating **Overall NAE Risk**, **Non-AIDS Cancer Risk**, and **Cardiovascular Event Risk**.
""")

# Medianas de referencia para imputación por defecto
MEDIANAS = {
    'LAB_V_num_CHOL': 162.0,
    'LAB_V_num_HDL': 40.0,
    'LAB_V_num_GLUC': 90.0,
    'LAB_V_num_TRIG': 102.0,
    'LAB_V_num_PLT': 217000.0,
    'LAB_V_num_AST': 24.0,
    'LAB_V_num_ALT': 24.0
}

# -----------------------------------------------------------------------------
# 2. CARGA DE MODELOS (.PKL) CON CACHÉ DE STREAMLIT
# -----------------------------------------------------------------------------
@st.cache_resource
def load_all_models():
    try:
        model_global = joblib.load('mejor_modelo_ENO.pkl')
        model_cancer = joblib.load('Cancer_modelo.pkl')
        model_cardio = joblib.load('Cardiovascular_modelo.pkl')
        metadata = joblib.load('metadatos_columnas.pkl')
        return model_global, model_cancer, model_cardio, metadata, None
    except Exception as e:
        return None, None, None, None, str(e)

model_global, model_cancer, model_cardio, metadata, error = load_all_models()

if error:
    st.error(f"⚠️ **Error loading model files (.pkl):** {error}")
    st.info("Make sure 'mejor_modelo_ENO.pkl', 'Cancer_modelo.pkl', 'Cardiovascular_modelo.pkl', and 'metadatos_columnas.pkl' are in the root directory.")
    st.stop()

# -----------------------------------------------------------------------------
# 3. CAPTURA DE DATOS DEL PACIENTE (BARRA LATERAL Y PESTAÑAS)
# -----------------------------------------------------------------------------
st.sidebar.header("📋 Patient Profile")

edad = st.sidebar.number_input("Age (Years)", min_value=18, max_value=90, value=48)
gender = st.sidebar.selectbox("Gender", options=['Hombre', 'Mujer', 'Unknown'], index=0)
seguimiento_anos = st.sidebar.slider("Follow-up Time (Years)", min_value=0.5, max_value=25.0, value=7.3, step=0.1)
seguimiento_dias = seguimiento_anos * 365.25

tab1, tab2, tab3 = st.tabs(["👤 Demographics & History", "🧪 Labs & Metabolic Scores", "💊 HIV & ART Profile"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        country = st.selectbox('Country of Origin', ['España', 'Latinoamérica', 'Europa Central/Oriental', 'Sub-Saharan Africa', 'Otros'], index=0)
        edu = st.selectbox('Education Level', ['Secundaria/FP', 'Universitaria', 'Primaria o menor', 'Unknown'], index=0)
        mode = st.selectbox('Transmission Mode', ['MSM', 'Heterosexual', 'IDU', 'Otros/Unknown'], index=0)
    with c2:
        smoking = st.selectbox('Smoking Status', ['0', '1', 'Unknown'], index=0, help="0: Non-smoker, 1: Smoker")
        alcohol = st.selectbox('Alcohol Intake', ['0', '1', 'Unknown'], index=0, help="0: No/Moderate, 1: Heavy")

with tab2:
    c1, c2, c3 = st.columns(3)
    with c1:
        chol = st.number_input('Cholesterol (mg/dL)', 0, 500, 162)
        gluc = st.number_input('Glucose (mg/dL)', 0.0, 500.0, 90.0)
        hdl = st.number_input('HDL (mg/dL)', 0, 180, 40)
    with c2:
        trig = st.number_input('Triglycerides (mg/dL)', 0, 500, 102)
        plt_input = st.number_input('Platelets (cells/µL)', 0, 1000000, 217000)
        ast = st.number_input('AST (U/L)', 0, 500, 24)
    with c3:
        alt = st.number_input('ALT (U/L)', 0, 500, 24)
        
        # Escala adecuada para la ecuación de FIB-4 (Plaquetas en miles)
        plt_thousands = (plt_input / 1000.0) if plt_input > 0 else (MEDIANAS['LAB_V_num_PLT'] / 1000.0)
        
        # Cálculos de visores
        tyg_visual = np.log((trig * gluc) / 2) if (trig > 0 and gluc > 0) else 0.0
        fib4_visual = (edad * ast) / (plt_thousands * np.sqrt(alt)) if (plt_thousands > 0 and alt > 0) else 0.0
        
        st.info(f"**Calculated TyG:** {tyg_visual:.2f}\n\n**Calculated FIB-4:** {fib4_visual:.2f}")

with tab3:
    c1, c2 = st.columns(2)
    with c1:
        vhc = st.selectbox('HCV Antibodies (VHC_ab)', ['Negativo', 'Positivo', 'Unknown'], index=0)
        vhb = st.selectbox('HBV Antigen (VHB_ag)', ['Negativo', 'Positivo', 'Unknown'], index=0)
        aids = st.selectbox('AIDS History (AIDS_Y)', ['No', 'Sí', 'Unknown'], index=0)
    with c2:
        cd4 = st.selectbox('CD4 Category at baseline', ['>=500', '200-499', '<200', 'Unknown'], index=0)
        carga = st.selectbox('Viral Load Category', ['<100.000', '>=100.000', 'Unknown'], index=0)
        year_art = st.selectbox('Year of ART Initiation', ['2016-2019', '2012-2015', '2020-2024', '2004-2007', '2008-2011', '<2004'], index=0)
        tar = st.selectbox('First ART Regimen', ['2NRTI+1II', '2NRTI+1NNRTI', '2NRTI+1PI', 'Otros'], index=0)

# -----------------------------------------------------------------------------
# 4. PROCESAMIENTO Y PREDICCIÓN MULTI-RIESGO
# -----------------------------------------------------------------------------
st.divider()

if st.button('🚀 CALCULATE MULTI-RISK PROFILE', type='primary', use_container_width=True):
    
    # Preparar valores procesados
    plt_s = plt_input if plt_input > 0 else MEDIANAS['LAB_V_num_PLT']
    alt_s = alt if alt > 0 else MEDIANAS['LAB_V_num_ALT']
    plt_thousands_for_fib4 = plt_s / 1000.0
    
    tyg_f = np.log((trig * gluc) / 2)
    fib4_f = (edad * ast) / (plt_thousands_for_fib4 * np.sqrt(alt_s))
    
    input_dict = {
        'edad': edad,
        'LAB_V_num_CHOL': chol,
        'LAB_V_num_HDL': hdl,
        'TyG': tyg_f,
        'FIB4': fib4_f,
        'tiempo_seguimiento': seguimiento_dias,
        'GENDER': str(gender),
        'MODE_cat': str(mode),
        'Country_origin': str(country),
        'EDU_cat_label': str(edu),
        'VHC_ab': str(vhc),
        'VHB_ag': str(vhb),
        'carga_inicial_cat': str(carga),
        'CD4_cat': str(cd4),
        'ALCOHOL': str(alcohol),
        'SMOKING': str(smoking),
        'Year_of_ART_initiation': str(year_art),
        'tipo_primerTAR': str(tar),
        'AIDS_Y': str(aids)
    }
    
    df_patient = pd.DataFrame([input_dict])
    
    # Cálculo de las 3 probabilidades
    prob_global = model_global.predict_proba(df_patient)[0][1]
    prob_cancer = model_cancer.predict_proba(df_patient)[0][1]
    prob_cardio = model_cardio.predict_proba(df_patient)[0][1]
    
    # -------------------------------------------------------------------------
    # 5. DESPLIEGUE DE RESULTADOS EN TARJETAS COMPARATIVAS
    # -------------------------------------------------------------------------
    st.subheader("📊 Multi-Risk Assessment Results")
    
    col_g, col_c, col_v = st.columns(3)
    
    # A. Riesgo Global
    with col_g:
        st.metric("Overall NAE Risk", f"{prob_global:.1%}")
        if prob_global < 0.15:
            st.success("✅ **LOW RISK** (<15%)\n\nBelow population baseline.")
        elif prob_global < 0.30:
            st.warning("⚠️ **MODERATE RISK** (15-30%)\n\nClose monitoring advised.")
        else:
            st.error("🚨 **HIGH RISK** (>30%)\n\nHigh risk accumulation.")

    # B. Cáncer No-SIDA (Basal ~5.4%)
    with col_c:
        st.metric("Non-AIDS Cancer Risk", f"{prob_cancer:.1%}")
        if prob_cancer < 0.08:
            st.success("✅ **LOW RISK** (<8%)\n\nStandard screening.")
        elif prob_cancer < 0.18:
            st.warning("⚠️ **MODERATE RISK** (8-18%)\n\nEnhanced oncological screening.")
        else:
            st.error("🚨 **HIGH RISK** (>18%)\n\nPriority cancer workup.")

    # C. Evento Cardiovascular (Basal ~3.2%)
    with col_v:
        st.metric("Cardiovascular Event Risk", f"{prob_cardio:.1%}")
        if prob_cardio < 0.05:
            st.success("✅ **LOW RISK** (<5%)\n\nStandard CV measures.")
        elif prob_cardio < 0.12:
            st.warning("⚠️ **MODERATE RISK** (5-12%)\n\nTarget lipid & glycemic control.")
        else:
            st.error("🚨 **HIGH RISK** (>12%)\n\nIntensive CV intervention.")

    # -------------------------------------------------------------------------
    # 6. EXPLICACIÓN CLÍNICA ADICIONAL
    # -------------------------------------------------------------------------
    st.divider()
    with st.expander("ℹ️ **Clinical Interpretation & Benchmarks**"):
        st.markdown(f"""
        * **Patient Profile:** {edad} years old | Follow-up horizon: {seguimiento_anos:.1f} years.
        * **Calculated Biomarkers:** TyG Index = `{tyg_f:.2f}` | FIB-4 Index = `{fib4_f:.2f}`.
        * **Methodological Basis:**
            * *Overall NAE Risk Model:* Evaluates general non-AIDS morbidity (Prevalence: ~9.2%).
            * *Non-AIDS Cancer Model (XGBoost):* ROC-AUC = `0.935`. Key predictors include age, HCV status, CD4 levels, and smoking history.
            * *Cardiovascular Model (LightGBM):* ROC-AUC = `0.930`. Key predictors include TyG Index, total cholesterol, age, and ART initiation era.
        """)
# SIDEBAR INFORMATIVA
st.sidebar.markdown("### Model Information")
st.sidebar.info(
    'This predictive tool runs on the optimal pipeline trained across LightGBM architecture.'
)
