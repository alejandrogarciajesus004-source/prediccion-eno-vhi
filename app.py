import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

# 1. CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(
    page_title="NAE Risk Predictor", layout="wide", page_icon="🩺"
)


# 2. CARGA DE RECURSOS Y CONSTANTES
@st.cache_resource
def cargar_recursos():
  pipeline = joblib.load('mejor_modelo_ENO.pkl')
  metadatos = joblib.load('metadatos_columnas.pkl')
  return pipeline, metadatos


try:
  pipeline, metadatos = cargar_recursos()
except FileNotFoundError:
  st.error(
      "❌ No se encontraron los archivos '.pkl'. Ejecuta primero tu script de"
      ' entrenamiento.'
  )
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
    'tiempo_seguimiento': 2674.5,
}

# 3. INTERFAZ DE USUARIO
st.title("Non-AIDS-Defining Event (NAE) Risk Calculator")
st.markdown(
    'Esta herramienta estima la probabilidad de desarrollar un evento no'
    ' SIDA (ENO) en pacientes con VIH mediante el modelo óptimo seleccionado.'
)

st.divider()

tab1, tab2, tab3 = st.tabs(
    ['Sociodemographics', 'Labs & Scores', 'HIV Parameters']
)

with tab1:
  c1, c2 = st.columns(2)
  with c1:
    edad = st.number_input('Age (years)', 18, 90, 35)
    gender = st.selectbox(
        'Gender',
        ['Hombre', 'Mujer'],
        format_func=lambda x: 'Male' if x == 'Hombre' else 'Female',
    )
    country = st.selectbox('Country of origin', ['Spain', 'No Spain'])
    edu = st.selectbox(
        'Education level',
        ['No or compulsory', 'Upper secondary or university', 'Unknown'],
    )
  with c2:
    mode = st.selectbox(
        'Transmission Mode',
        ['Homo/Bisexual', 'Heterosexual', 'UDI', 'Other/Unknown'],
    )
    alcohol = st.selectbox(
        'Alcohol consumption',
        ['0', '1', 'Unknown'],
        format_func=lambda x: (
            'No' if x == '0' else ('Yes' if x == '1' else 'Unknown')
        ),
    )
    smoking = st.selectbox(
        'Smoking status',
        ['0', '1', 'Unknown'],
        format_func=lambda x: (
            'No' if x == '0' else ('Yes' if x == '1' else 'Unknown')
        ),
    )

with tab2:
  c1, c2, c3 = st.columns(3)
  with c1:
    chol = st.number_input('Cholesterol (mg/dL)', 0, 500, 162)
    gluc = st.number_input('Glucose (mg/dL)', 0.0, 500.0, 90.0)
    hdl = st.number_input('HDL (mg/dL)', 0, 180, 40)
  with c2:
    trig = st.number_input('Triglycerides (mg/dL)', 0, 500, 102)
    plt = st.number_input('Platelets (cells/µL)', 0, 1000000, 217000)
    ast = st.number_input('AST (U/L)', 0, 500, 24)
  with c3:
    alt = st.number_input('ALT (U/L)', 0, 500, 24)

    # Cálculos visuales informativos
    tyg_visual = np.log((trig * gluc) / 2)
    fib4_visual = (
        (edad * ast) / (plt * np.sqrt(alt)) if plt > 0 and alt > 0 else 0
    )
    st.info(
        f"**Calculated TyG:** {tyg_visual:.2f}\n\n**Calculated FIB-4:**"
        f' {fib4_visual:.4f}'
    )

with tab3:
  c1, c2 = st.columns(2)
  with c1:
    cd4 = st.selectbox('CD4 category', ['≥200', '<200', 'Unknown'])
    carga = st.selectbox(
        'Initial viral load', ['<100.000', '≥100.000', 'Unknown']
    )
    aids = st.selectbox(
        'Previous AIDS event',
        ['No', 'Si', 'Desconocido'],
        format_func=lambda x: (
            'Yes' if x == 'Si' else ('No' if x == 'No' else 'Unknown')
        ),
    )
    vhc = st.selectbox(
        'HCV (Hepatitis C)',
        ['Negativo', 'Positivo', 'Unknown'],
        format_func=lambda x: (
            'Negative'
            if x == 'Negativo'
            else ('Positive' if x == 'Positivo' else 'Unknown')
        ),
    )
  with c2:
    vhb = st.selectbox(
        'HBV (Hepatitis B)',
        ['Negativo', 'Positivo', 'Unknown'],
        format_func=lambda x: (
            'Negative'
            if x == 'Negativo'
            else ('Positive' if x == 'Positivo' else 'Unknown')
        ),
    )
    tar = st.selectbox(
        'First ART regimen',
        ['2NRTI+1NNRTI', '2NRTI+1PI', '2NRTI+1II', 'Other/Unknown'],
    )
    year_art = st.selectbox(
        'Year of ART initiation',
        ['2004–2007', '2008–2011', '2012–2015', '2016–2019', '2020–2024'],
    )
    seguimiento_anios = st.number_input(
        'Follow-up time (years)', 0.5, 25.0, 7.0
    )
    seguimiento_dias = seguimiento_anios * 365.25

# 4. PROCESAMIENTO Y PREDICCIÓN
st.divider()
if st.button('CALCULATE RISK', type='primary', use_container_width=True):

  # Protección contra división por cero
  plt_s = plt if plt > 0 else MEDIANAS['LAB_V_num_PLT']
  alt_s = alt if alt > 0 else MEDIANAS['LAB_V_num_ALT']

  # Scores finales
  tyg_f = np.log((trig * gluc) / 2)
  fib4_f = (edad * ast) / (plt_s * np.sqrt(alt_s))

  # Construir DataFrame idéntico al original
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
      'AIDS_Y': str(aids),
  }

  df_patient = pd.DataFrame([input_dict])

  # Predicción con el Pipeline
  prob = pipeline.predict_proba(df_patient)[0][1]

  # MOSTRAR RESULTADOS
  st.subheader('Evaluation Result')
  col_score, col_text = st.columns([1, 2])

  with col_score:
    st.metric('Estimated Risk', f'{prob:.1%}')

  with col_text:
    if prob < 0.20:
      st.success(
          '✅ **LOW RISK:** The clinical profile suggests a low probability of'
          ' NAE.'
      )
    elif prob < 0.45:
      st.warning(
          '⚠️ **INTERMEDIATE RISK:** Close clinical monitoring is recommended.'
      )
    else:
      st.error(
          '🚨 **HIGH RISK:** The model identifies multiple predictive factors'
          ' for NAE.'
      )

  # 5. EXPLICACIÓN INDIVIDUALIZADA SHAP (Waterfall Plot)
  st.divider()
  st.subheader('Individualized Explanation (SHAP Analysis)')
  st.write(
      'This chart shows how each patient feature pushed the probability higher'
      ' (red) or lower (blue).'
  )

  try:
    # Extraer componentes del Pipeline
    preprocessor = pipeline.named_steps['preprocessor']
    classifier = pipeline.named_steps['classifier']

    # Preprocesar la muestra del paciente
    X_patient_prep = preprocessor.transform(df_patient)
    feature_names = preprocessor.get_feature_names_out()

    # Formatear nombres de columnas para lectura limpia
    clean_feature_names = [
        col.replace('num__', '').replace('cat__', '') for col in feature_names
    ]

    X_patient_df = pd.DataFrame(
        X_patient_prep, columns=clean_feature_names, index=[0]
    )

    # Crear explainer
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer(X_patient_df)

    # Extraer clase 1 (Sí ENO) si el modelo devuelve estructura multiclase
    if len(shap_values.shape) == 3:
      shap_single = shap_values[0, :, 1]
    else:
      shap_single = shap_values[0]

    # Graficar Waterfall Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.plots.waterfall(shap_single, max_display=10, show=False)
    plt.tight_layout()
    st.pyplot(fig)

  except Exception as e:
    st.info(
        'Individual SHAP visualization is only available for tree-based'
        f' models. ({e})'
    )

# SIDEBAR INFORMATIVA
st.sidebar.markdown("### Model Information")
st.sidebar.info(
    'This predictive tool runs on the optimal pipeline trained across'
    ' Random Forest, LightGBM, XGBoost, and SVM architectures.'
)
