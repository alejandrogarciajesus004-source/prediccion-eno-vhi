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
    ' This tool estimates the risk of suffering a non-AIDS defining event with a '
    ' machine learning model'
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
    # Plaquetas en unidades absolutas (ej. 217000)
    plt_input = st.number_input('Platelets (cells/µL)', 0, 1000000, 217000)
    ast = st.number_input('AST (U/L)', 0, 500, 24)
  with c3:
    alt = st.number_input('ALT (U/L)', 0, 500, 24)

    # Conversión a miles ÚNICAMENTE para la ecuación del FIB-4
    plt_in_thousands = (
        plt_input / 1000.0
        if plt_input > 0
        else (MEDIANAS['LAB_V_num_PLT'] / 1000.0)
    )

    # Cálculos visuales
    tyg_visual = np.log((trig * gluc) / 2)
    fib4_visual = (
        (edad * ast) / (plt_in_thousands * np.sqrt(alt))
        if plt_in_thousands > 0 and alt > 0
        else 0
    )

    st.info(
        f"**Calculated TyG:** {tyg_visual:.2f}\n\n**Calculated FIB-4:**"
        f' {fib4_visual:.2f}'
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

# --- 4. PROCESAMIENTO Y PREDICCIÓN ---
st.divider()
if st.button('CALCULATE RISK', type='primary', use_container_width=True):

  # Plaquetas en unidades absolutas para el diccionario del modelo
  plt_s = plt_input if plt_input > 0 else MEDIANAS['LAB_V_num_PLT']
  alt_s = alt if alt > 0 else MEDIANAS['LAB_V_num_ALT']

  # Plaquetas en miles SOLAMENTE para el cálculo de FIB4
  plt_thousands_for_fib4 = plt_s / 1000.0

  # Scores finales con la escala idéntica a la del CSV
  tyg_f = np.log((trig * gluc) / 2)
  fib4_f = (edad * ast) / (plt_thousands_for_fib4 * np.sqrt(alt_s))

  # Construir DataFrame
  input_dict = {
      'edad': edad,
      'LAB_V_num_CHOL': chol,
      'LAB_V_num_HDL': hdl,
      'TyG': tyg_f,  # ~8.4
      'FIB4': fib4_f,  # ~0.81
      'tiempo_seguimiento': seguimiento_dias,  # en DÍAS
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
    st.metric('Estimated Risk Index', f'{prob:.1%}')

  with col_text:
    if prob < 0.30:
      st.success(
          '✅ **LOW RISK:** The clinical profile suggests a low probability of'
          ' NAE.'
      )
    elif prob < 0.55:
      st.warning(
          '⚠️ **INTERMEDIATE RISK:** Moderate predictive score. Close clinical'
          ' monitoring recommended.'
      )
    else:
      st.error(
          '🚨 **HIGH RISK:** The model identifies multiple strong risk'
          ' factors for NAE.'
      )
 # 5. EXPLICACIÓN INDIVIDUALIZADA SHAP (CORREGIDA Y BLINDEADA)
    st.divider()
    st.subheader("Individualized Explanation (SHAP Analysis)")
    st.write(
        "This waterfall plot illustrates how each clinical factor pushes the prediction "
        "higher (red) or lower (blue) relative to the baseline risk."
    )

    try:
        # Extraer preprocesador y clasificador del pipeline
        preprocessor = pipeline.named_steps['preprocessor']
        classifier = pipeline.named_steps['classifier']

        # Preprocesar datos del paciente
        X_prep = preprocessor.transform(df_patient)
        feature_names = preprocessor.get_feature_names_out()

        # Nombres de variables limpios para la interfaz
        clean_names = [col.replace('num__', '').replace('cat__', '') for col in feature_names]
        X_df = pd.DataFrame(X_prep, columns=clean_names)

        # SHAP Explainer
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer(X_df)

        # Seleccionar la clase de interés (Clase 1: Sí ENO)
        if len(shap_values.shape) == 3:
            single_shap = shap_values[0, :, 1]
        else:
            single_shap = shap_values[0]

        # CREACIÓN EXPLÍCITA DE LA FIGURA DE MATPLOTLIB
        fig, ax = plt.subplots(figsize=(9, 5))
        
        # Le pasamos la figura/eje directamente a SHAP
        shap.plots.waterfall(single_shap, max_display=10, show=False)
        
        # Ajustamos layout y renderizamos de forma limpia
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig) # Cerramos la figura para liberar memoria

    except Exception as e:
        st.warning(
            f"Could not render individual SHAP explanation: {e}. "
            "Note that SHAP waterfall plots are optimized for tree-based estimators."
        )
# SIDEBAR INFORMATIVA
st.sidebar.markdown("### Model Information")
st.sidebar.info(
    'This predictive tool runs on the optimal pipeline trained across LightGBM architecture.'
)
