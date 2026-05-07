# ======================================================
# IA BANK CHURN PREDICTOR
# INTERFAZ MODERNA STREAMLIT
# ======================================================

import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score


# ======================================================
# CONFIGURACIÓN
# ======================================================

st.set_page_config(
    page_title="Churn Predictor",
    page_icon="🤖",
    layout="wide"
)


# ======================================================
# CSS PERSONALIZADO
# ======================================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
    color: white;
}

.stApp {
    background: linear-gradient(
        135deg,
        #0f172a,
        #111827,
        #1e293b
    );
}

h1, h2, h3 {
    color: white;
}

.block-container {
    padding-top: 2rem;
}

.metric-card {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 20px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.1);
    text-align: center;
}

.prediction-box {
    padding: 25px;
    border-radius: 20px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
}

.big-font {
    font-size: 22px !important;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)


# ======================================================
# HEADER
# ======================================================

st.markdown("""
# BANK CHURN PREDICTOR

### Sistema Inteligente de Predicción de Abandono Bancario

Machine Learning con:
- Regresión Logística
- Random Forest
""")

st.divider()


# ======================================================
# CARGAR DATASET
# ======================================================

@st.cache_data
def cargar_datos():
    return pd.read_csv(
        "Bank Customer Churn Prediction.csv"
    )


df = cargar_datos()


# ======================================================
# SELECCIÓN DE VARIABLES
# ======================================================

df = df[[
    "credit_score",
    "country",
    "gender",
    "age",
    "tenure",
    "balance",
    "products_number",
    "credit_card",
    "active_member",
    "estimated_salary",
    "churn"
]].copy()


# ======================================================
# LIMPIEZA
# ======================================================

df = df.dropna()
df = df.drop_duplicates()


# ======================================================
# LABEL ENCODER
# ======================================================

le_country = LabelEncoder()
le_gender = LabelEncoder()

df["country"] = le_country.fit_transform(
    df["country"]
)

df["gender"] = le_gender.fit_transform(
    df["gender"]
)


# ======================================================
# VARIABLES
# ======================================================

X = df[[
    "credit_score",
    "country",
    "gender",
    "age",
    "tenure",
    "balance",
    "products_number",
    "credit_card",
    "active_member",
    "estimated_salary"
]]

y = df["churn"]


# ======================================================
# TRAIN TEST SPLIT
# ======================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=101
)


# ======================================================
# SCALER
# ======================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# ======================================================
# MODELO LOGÍSTICO
# ======================================================

log_model = LogisticRegression(
    max_iter=1000
)

log_model.fit(
    X_train_scaled,
    y_train
)


# ======================================================
# RANDOM FOREST
# ======================================================

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=101
)

rf_model.fit(
    X_train,
    y_train
)


# ======================================================
# ACCURACY
# ======================================================

log_accuracy = accuracy_score(
    y_test,
    log_model.predict(X_test_scaled)
)

rf_accuracy = accuracy_score(
    y_test,
    rf_model.predict(X_test)
)


# ======================================================
# SIDEBAR
# ======================================================

st.sidebar.title("Métricas")

st.sidebar.metric(
    "Logistic Regression",
    f"{log_accuracy:.2%}"
)

st.sidebar.metric(
    "Random Forest",
    f"{rf_accuracy:.2%}"
)

st.sidebar.divider()

st.sidebar.info("""
💡 Random Forest obtuvo
el mejor desempeño general
en el proyecto.
""")


# ======================================================
# COLUMNAS
# ======================================================

col1, col2 = st.columns([1,1])


# ======================================================
# FORMULARIO
# ======================================================

with col1:

    st.markdown("## Datos del Cliente")

    credit_score = st.slider(
        "Credit Score",
        300,
        900,
        650
    )

    country = st.selectbox(
        "País",
        ["France", "Germany", "Spain"]
    )

    gender = st.selectbox(
        "Género",
        ["Female", "Male"]
    )

    age = st.slider(
        "Edad",
        18,
        100,
        35
    )

    tenure = st.slider(
        "Años como cliente",
        0,
        10,
        5
    )

    balance = st.number_input(
        "Balance",
        0.0,
        300000.0,
        50000.0
    )

    products_number = st.slider(
        "Productos contratados",
        1,
        4,
        2
    )

    credit_card = st.selectbox(
        "¿Tiene tarjeta?",
        [0, 1]
    )

    active_member = st.selectbox(
        "¿Es miembro activo?",
        [0, 1]
    )

    estimated_salary = st.number_input(
        "Salario estimado",
        0.0,
        300000.0,
        60000.0
    )


# ======================================================
# CODIFICACIÓN
# ======================================================

country_encoded = le_country.transform(
    [country]
)[0]

gender_encoded = le_gender.transform(
    [gender]
)[0]


# ======================================================
# NUEVO CLIENTE
# ======================================================

nuevo_cliente = pd.DataFrame({

    "credit_score": [credit_score],
    "country": [country_encoded],
    "gender": [gender_encoded],
    "age": [age],
    "tenure": [tenure],
    "balance": [balance],
    "products_number": [products_number],
    "credit_card": [credit_card],
    "active_member": [active_member],
    "estimated_salary": [estimated_salary]

})


# ======================================================
# RESULTADOS
# ======================================================

with col2:

    st.markdown("## Predicción")

    st.markdown("""
    <div class='prediction-box'>
    Sistema de análisis
    de abandono bancario.
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    if st.button(
        "Analizar Cliente",
        use_container_width=True
    ):

        # ==========================================
        # REGRESIÓN LOGÍSTICA
        # ==========================================

        nuevo_cliente_scaled = scaler.transform(
            nuevo_cliente
        )

        pred_log = log_model.predict(
            nuevo_cliente_scaled
        )[0]

        prob_log = log_model.predict_proba(
            nuevo_cliente_scaled
        )[0][1]

        # ==========================================
        # RANDOM FOREST
        # ==========================================

        pred_rf = rf_model.predict(
            nuevo_cliente
        )[0]

        prob_rf = rf_model.predict_proba(
            nuevo_cliente
        )[0][1]

        # ==========================================
        # RESULTADOS IA
        # ==========================================

        st.markdown("## Resultado Final")

        # RANDOM FOREST
        st.markdown("### Random Forest")

        if pred_rf == 1:

            st.error(
                "⚠️ Riesgo ALTO de abandono"
            )

        else:

            st.success(
                "✅ Cliente probablemente permanecerá"
            )

        st.progress(float(prob_rf))

        st.write(
            f"Probabilidad de abandono: "
            f"{prob_rf:.2%}"
        )

        st.divider()

        # LOGISTIC REGRESSION
        st.markdown("### Regresión Logística")

        if pred_log == 1:

            st.warning(
                "⚠️ Posible abandono detectado"
            )

        else:

            st.success(
                "✅ Cliente estable"
            )

        st.progress(float(prob_log))

        st.write(
            f"Probabilidad de abandono: "
            f"{prob_log:.2%}"
        )

        st.divider()

        # COMPARACIÓN FINAL

        st.markdown("## Conclusión ")

        if prob_rf > prob_log:

            st.info("""
            El modelo Random Forest detecta
            un mayor riesgo de abandono.
            """)

        else:

            st.info("""
            La Regresión Logística detecta
            un mayor riesgo de abandono.
            """)

        # SCORE VISUAL

        riesgo = int(prob_rf * 100)

        st.markdown(
            f"""
            <div class='metric-card'>
            <p class='big-font'>
            Riesgo de Churn
            </p>
            <h1>{riesgo}%</h1>
            </div>
            """,
            unsafe_allow_html=True
        )