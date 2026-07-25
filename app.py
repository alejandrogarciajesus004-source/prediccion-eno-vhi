import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & TITLE
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

# Reference medians for missing/default lab imputations
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
# 2. LOAD TRAINED MODELS (.PKL)
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
    st.info("Make sure 'mejor_modelo_ENO.pkl', 'Cancer_modelo.pkl', 'Cardiovascular_modelo.pkl', and 'metadatos_columnas.pkl' are in the working directory.")
    st.stop()

# -----------------------------------------------------------------------------
# 3. SIDEBAR & PATIENT PROFILE INPUTS (EXACT DATASET CATEGORIES)
# -----------------------------------------------------------------------------
st.sidebar.header("📋 Patient Demographics")

edad = st.sidebar.number_input("Age (Years)", min_value=18, max_value=90, value=48)
gender = st.sidebar.selectbox("GENDER", options=['Male', 'Female', 'Unknown'], index=0)
seguimiento_anos = st.sidebar.slider("Follow-up Time (Years)", min_value=0.5, max_value=25.0, value=7.3, step=0.1)
seguimiento_dias = seguimiento_anos * 365.25

tab1, tab2, tab3 = st.tabs(["👤 Demographics & History", "🧪 Labs & Metabolic Scores", "💊 HIV & ART Profile"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        country = st.selectbox('Country_origin', ['Spain', 'Latin America', 'Central/Eastern Europe', 'Sub-Saharan Africa', 'Others', 'Unknown'], index=0)
        edu = st.selectbox('EDU_cat_label', ['Secondary/Vocational', 'University', 'Primary or less', 'Unknown'], index=0)
        mode = st.selectbox('MODE_cat', ['MSM', 'Heterosexual', 'IDU', 'Others/Unknown'], index=0)
    with c2:
        smoking = st.selectbox('SMOKING', ['0', '1', 'Unknown'], index=0, help="0: Non-smoker, 1: Smoker")
        alcohol = st.selectbox('ALCOHOL', ['0', '1', 'Unknown'], index=0, help="0: No/Moderate, 1: Heavy")

with tab2:
    c1, c2, c3 = st.columns(3)
    with c1:
        chol = st.number_input('Cholesterol (LAB_V_num_CHOL, mg/dL)', 0, 500, 162)
        gluc = st.number_input('Glucose (LAB_V_num_GLUC, mg/dL)', 0.0, 500.0, 90.0)
        hdl = st.number_input('HDL (LAB_V_num_HDL, mg/dL)', 0, 180, 40)
    with c2:
        trig = st.number_input('Triglycerides (LAB_V_num_TRIG, mg/dL)', 0, 500, 102)
        plt_input = st.number_input('Platelets (LAB_V_num_PLT, cells/µL)', 0, 1000000, 217000)
        ast = st.number_input('AST (LAB_V_num_AST, U/L)', 0, 500, 24)
    with c3:
        alt = st.number_input('ALT (LAB_V_num_ALT, U/L)', 0, 500, 24)
        
        # Scale platelets (in thousands) solely for the FIB-4 index formula
        plt_thousands = (plt_input / 1000.0) if plt_input > 0 else (MEDIANAS['LAB_V_num_PLT'] / 1000.0)
        
        # Visual calculated metrics
        tyg_visual = np.log((trig * gluc) / 2) if (trig > 0 and gluc > 0) else 0.0
        fib4_visual = (edad * ast) / (plt_thousands * np.sqrt(alt)) if (plt_thousands > 0 and alt > 0) else 0.0
        
        st.info(f"**Calculated TyG Index:** {tyg_visual:.2f}\n\n**Calculated FIB-4 Index:** {fib4_visual:.2f}")

with tab3:
    c1, c2 = st.columns(2)
    with c1:
        vhc = st.selectbox('VHC_ab', ['Negative', 'Positive', 'Unknown'], index=0)
        vhb = st.selectbox('VHB_ag', ['Negative', 'Positive', 'Unknown'], index=0)
        aids = st.selectbox('AIDS_Y', ['No', 'Yes', 'Unknown'], index=0)
    with c2:
        cd4 = st.selectbox('CD4_cat', ['>=500', '200-499', '<200', 'Unknown'], index=0)
        carga = st.selectbox('carga_inicial_cat', ['<100.000', '>=100.000', 'Unknown'], index=0)
        year_art = st.selectbox('Year_of_ART_initiation', ['2016-2019', '2012-2015', '2020-2024', '2004-2007', '2008-2011', '<2004', 'Unknown'], index=0)
        tar = st.selectbox('tipo_primerTAR', ['2NRTI+1II', '2NRTI+1NNRTI', '2NRTI+1PI', 'Others', 'Unknown'], index=0)

# -----------------------------------------------------------------------------
# 4. PROCESSING & MULTI-RISK PREDICTION
# -----------------------------------------------------------------------------
st.divider()

if st.button('🚀 CALCULATE MULTI-RISK PROFILE', type='primary', use_container_width=True):
    
    # Process lab values
    plt_s = plt_input if plt_input > 0 else MEDIANAS['LAB_V_num_PLT']
    alt_s = alt if alt > 0 else MEDIANAS['LAB_V_num_ALT']
    plt_thousands_for_fib4 = plt_s / 1000.0
    
    tyg_f = np.log((trig * gluc) / 2)
    fib4_f = (edad * ast) / (plt_thousands_for_fib4 * np.sqrt(alt_s))
    
    # Construct input dataframe with exact feature names and categories from the CSV
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
    
    # Calculate probabilities across the 3 models
    prob_global = model_global.predict_proba(df_patient)[0][1]
    prob_cancer = model_cancer.predict_proba(df_patient)[0][1]
    prob_cardio = model_cardio.predict_proba(df_patient)[0][1]
    
    # -------------------------------------------------------------------------
    # 5. DISPLAY RESULTS IN COMPARATIVE METRIC CARDS
    # -------------------------------------------------------------------------
    st.subheader("📊 Multi-Risk Assessment Results")
    
    col_g, col_c, col_v = st.columns(3)
    
    # A. Overall NAE Risk
    with col_g:
        st.metric("Overall NAE Risk", f"{prob_global:.1%}")
        if prob_global < 0.15:
            st.success("✅ **LOW RISK** (<15%)\n\nBelow population baseline.")
        elif prob_global < 0.30:
            st.warning("⚠️ **MODERATE RISK** (15-30%)\n\nClose monitoring advised.")
        else:
            st.error("🚨 **HIGH RISK** (>30%)\n\nHigh risk accumulation.")

    # B. Non-AIDS Cancer Risk (Baseline ~5.4%)
    with col_c:
        st.metric("Non-AIDS Cancer Risk", f"{prob_cancer:.1%}")
        if prob_cancer < 0.08:
            st.success("✅ **LOW RISK** (<8%)\n\nStandard screening.")
        elif prob_cancer < 0.18:
            st.warning("⚠️ **MODERATE RISK** (8-18%)\n\nEnhanced oncological screening.")
        else:
            st.error("🚨 **HIGH RISK** (>18%)\n\nPriority cancer workup.")

    # C. Cardiovascular Event Risk (Baseline ~3.2%)
    with col_v:
        st.metric("Cardiovascular Event Risk", f"{prob_cardio:.1%}")
        if prob_cardio < 0.05:
            st.success("✅ **LOW RISK** (<5%)\n\nStandard CV measures.")
        elif prob_cardio < 0.12:
            st.warning("⚠️ **MODERATE RISK** (5-12%)\n\nTarget lipid & glycemic control.")
        else:
            st.error("🚨 **HIGH RISK** (>12%)\n\nIntensive CV intervention.")

    # -------------------------------------------------------------------------
    # 6. CLINICAL SUMMARY
    # -------------------------------------------------------------------------
    st.divider()
    with st.expander("ℹ️ **Clinical Interpretation & Model Details**"):
        st.markdown(f"""
        * **Patient Parameters:** {edad} years old | Follow-up time horizon: {seguimiento_anos:.1f} years.
        * **Calculated Indexes:** TyG Index = `{tyg_f:.2f}` | FIB-4 Index = `{fib4_f:.2f}`.
        * **Model Architecture & Performance:**
            * *Overall NAE Risk:* Random Forest (ROC-AUC: `0.923`).
            * *Non-AIDS Cancer Risk:* XGBoost (ROC-AUC: `0.935`).
            * *Cardiovascular Event Risk:* LightGBM (ROC-AUC: `0.930`).
        """)
# SIDEBAR INFORMATIVA
st.sidebar.markdown("### Model Information")
st.sidebar.info(
    'This predictive tool runs on the optimal pipelines for different machine learning algorithms'
)
