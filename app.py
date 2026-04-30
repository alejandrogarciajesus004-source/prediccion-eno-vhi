import streamlit as st
import pandas as pd
import joblib

# PAGE CONFIGURATION
st.set_page_config(page_title="ENO Risk Predictor", layout="wide")[cite: 1]

@st.cache_resource
def load_resources():
    # Load the existing model and columns
    model = joblib.load('modelo_ENO_rf.pkl')[cite: 1]
    training_columns = joblib.load('columnas_modelo.pkl')[cite: 1]
    return model, training_columns

try:
    model, training_columns = load_resources()
except FileNotFoundError:
    st.error("❌ Model files (.pkl) not found. Please ensure they are in the same directory.")
    st.stop()

# 1. USER INTERFACE (English)
st.title("Non-AIDS Events (NAE) Prediction Tool")
st.markdown("This tool estimates the probability of non-AIDS events using a Random Forest model.")[cite: 1]

st.divider()

# Organizing inputs in Tabs
tab1, tab2, tab3 = st.tabs(["Sociodemographics", "Metabolic & Hepatic Scores", "HIV Parameters"])[cite: 1]

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age (years)", 18, 90, 35)[cite: 1]
        gender = st.selectbox("Gender", ["Male", "Female"]) 
        country = st.selectbox("Country of Origin", ["Spain", "No Spain"])[cite: 1]
    with c2:
        mode = st.selectbox("Transmission Mode", ["Homo/Bisexual", "Heterosexual", "UDI", "Other/Unknown"])[cite: 1]
        smoking = st.selectbox("Smoking Status", ["0", "1", "Unknown"], help="1: Yes, 0: No")[cite: 1]

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        chol = st.number_input("Total Cholesterol (mg/dL)", 0, 500, 162)[cite: 1]
        hdl = st.number_input("HDL Cholesterol (mg/dL)", 0, 180, 40)[cite: 1]
    with c2:
        tyg = st.number_input("TyG Index", 1.0, 12.0, 8.4)[cite: 1]
        fib4 = st.number_input("FIB-4 Score", 0.0, 75.0, 0.8)[cite: 1]

with tab3:
    c1, c2 = st.columns(2)
    with c1:
        cd4 = st.selectbox("CD4 Category", ["≥200", "<200", "Unknown"])[cite: 1]
        vl = st.selectbox("Initial Viral Load", ["<100,000", "≥100,000", "Unknown"])[cite: 1]
        aids = st.selectbox("Previous AIDS Event", ["No", "Yes", "Unknown"])[cite: 1]
    with c2:
        follow_up = st.number_input("Follow-up time (days)", 180, 8000, 2674)[cite: 1]
        alcohol = st.selectbox("Alcohol Consumption", ["0", "1", "Unknown"])[cite: 1]

# 2. DATA PROCESSING & PREDICTION
if st.button("CALCULATE RISK", type="primary", use_container_width=True):
    
    # We create the dictionary including the hidden variables with neutral values (medians)
    # so the model doesn't crash, but the user doesn't have to fill them.
    input_dict = {
        # User-provided numericals
        'edad': age, 
        'LAB_V_num_CHOL': chol, 
        'LAB_V_num_HDL': hdl, 
        'TyG': tyg, 
        'FIB4': fib4, 
        'tiempo_seguimiento': follow_up,
        # Hidden numericals (using medians from your training script to keep it neutral)
        'LAB_V_num_GLUC': 90.0, 
        'LAB_V_num_TRIG': 102.0, 
        'LAB_V_num_PLT': 217000, 
        'LAB_V_num_AST': 24, 
        'LAB_V_num_ALT': 24,
        # Categoricals (mapped to training labels)
        'GENDER': gender, 
        'MODE_cat': mode,
        'Country_origin': country, 
        'EDU_cat_label': "Unknown", 
        'VHC_ab': "Unknown", 
        'VHB_ag': "Unknown", 
        'carga_inicial_cat': vl, 
        'CD4_cat': cd4, 
        'ALCOHOL': alcohol, 
        'SMOKING': smoking, 
        'Year_of_ART_initiation': '2016–2019', 
        'tipo_primerTAR': 'Other/Unknown', 
        'AIDS_Y': aids
    }
    
    # Convert to DataFrame
    df_input = pd.DataFrame([input_dict])
    
    # Ensure categoricals are strings[cite: 1]
    cat_cols = ['GENDER', 'MODE_cat', 'Country_origin', 'EDU_cat_label', 'VHC_ab', 
                'VHB_ag', 'carga_inicial_cat', 'CD4_cat', 'ALCOHOL', 'SMOKING', 
                'Year_of_ART_initiation', 'tipo_primerTAR', 'AIDS_Y']
    df_input[cat_cols] = df_input[cat_cols].astype(str)
    
    # One-hot encoding and alignment[cite: 1]
    df_dummies = pd.get_dummies(df_input)
    df_final = df_dummies.reindex(columns=training_columns, fill_value=0)[cite: 1]
    
    # Prediction
    probability = model.predict_proba(df_final)[0][1][cite: 1]
    
    # RESULTS DISPLAY
    st.subheader("Results")
    col_metric, col_desc = st.columns([1, 2])
    
    with col_metric:
        st.metric("Estimated Risk", f"{probability:.1%}")[cite: 1]
        
    with col_desc:
        if probability < 0.20:
            st.success("✅ LOW RISK: Clinical profile suggests a low probability of NAE.")[cite: 1]
        elif probability < 0.45:
            st.warning("⚠️ INTERMEDIATE RISK: Close clinical monitoring is recommended.")[cite: 1]
        else:
            st.error("🚨 HIGH RISK: Multiple predictors for NAE identified.")[cite: 1]
