import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Hairify – AI Hair Disease Detection",
    layout="wide",
    page_icon="🧑‍🦱"
)

# ===============================
# SESSION STATE
# ===============================
if "page" not in st.session_state:
    st.session_state.page = "home"
if "disease" not in st.session_state:
    st.session_state.disease = None
if "confidence" not in st.session_state:
    st.session_state.confidence = None
if "severity" not in st.session_state:
    st.session_state.severity = None

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
# ===============================
# LOAD MODEL
# ===============================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        r"C:\Users\thont\OneDrive\Documents\Hair_Diease_Detection\Hair_Diease_Detection\MobileNetV2_hair_model.h5",
        compile=False
    )

if "model" not in st.session_state:
    st.session_state.model = load_model()

model = st.session_state.model

# ===============================
# LOAD PRODUCTS CSV
# ===============================
@st.cache_data
def load_products():
    df = pd.read_csv(r"C:\Users\thont\OneDrive\Documents\Hair_Diease_Detection\Hair_Diease_Detection\products.csv")
    df.columns = df.columns.str.strip().str.lower()
    return df

products_df = load_products()

# ===============================
# LOAD REMEDIES CSV
# ===============================
@st.cache_data
def load_remedies():
    df = pd.read_csv(r"C:\Users\thont\OneDrive\Documents\Hair_Diease_Detection\Hair_Diease_Detection\ayurveda_remedies.csv")
    df.columns = df.columns.str.strip().str.lower()
    return df

remedies_df = load_remedies()

# ===============================
# CONSTANTS
# ===============================
LOGO_URL = "https://img.freepik.com/free-vector/women-logo-design-template_474888-1838.jpg"
BG_URL = "https://img.freepik.com/free-photo/watercolor-pastel-background_23-2151891320.jpg"

DISEASES = [
"Alopecia Areata",
"Contact Dermatitis",
"Folliculitis",
"Head Lice",
"Lichen Planus",
"Male Pattern Baldness",
"Psoriasis",
"Seborrheic Dermatitis",
"Telogen Effluvium",
"Tinea Capitis"
]

DISEASE_INFO = {
"Alopecia Areata": "An autoimmune condition causing sudden hair loss in patches.",
"Contact Dermatitis": "Scalp irritation caused by allergic reactions.",
"Folliculitis": "Inflammation of hair follicles due to infection.",
"Head Lice": "Parasitic infestation causing itching and irritation.",
"Lichen Planus": "Inflammatory condition affecting scalp.",
"Male Pattern Baldness": "Genetic hair thinning condition.",
"Psoriasis": "Chronic autoimmune skin condition causing scaly patches.",
"Seborrheic Dermatitis": "Common scalp condition causing dandruff and redness.",
"Telogen Effluvium": "Temporary hair shedding due to stress.",
"Tinea Capitis": "Fungal infection of the scalp."
}

# ===============================
# IMAGE PREPROCESSING
# ===============================
def preprocess_image(img):

    img = img.convert("RGB")
    img = img.resize((128,128))

    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    return img

# ===============================
# GLOBAL STYLES
# ===============================
st.markdown(f"""
<style>

[data-testid="stAppViewContainer"] {{
background-image: url('{BG_URL}');
background-size: cover;
background-attachment: fixed;
}}

.header {{
display:flex;
align-items:center;
justify-content:center;
gap:15px;
margin-bottom:30px;
}}

.header img {{
height:70px;
border-radius:15px;
}}

.header h1 {{
color:white;
font-size:50px;
}}

.prediction-box {{
font-size:24px;
color:white;
padding:20px;
border-radius:15px;
background:#ff5e62;
text-align:center;
margin-top:20px;
}}

.treatment-box {{
background:white;
padding:12px;
border-left:5px solid #3498db;
margin-top:10px;
border-radius:8px;
}}

</style>
""", unsafe_allow_html=True)

# =====================================================
# HOME PAGE
# =====================================================
if st.session_state.page == "home":

    st.markdown(f"""
    <div class="header">
        <img src="{LOGO_URL}">
        <h1>Hairify</h1>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.2,1])

    with col1:

        st.markdown(
        "<h1 style='color:black;'>Combat Hair Loss<br>with Confidence.</h1>",
        unsafe_allow_html=True
        )

        st.markdown(
        "<h3 style='color:black;'>Empowering You with AI for Healthier Hair</h3>",
        unsafe_allow_html=True
        )

        if st.button("Start Analysis →"):

            st.session_state.page="detect"
            st.rerun()

    with col2:

        st.image(
        "https://hairify-ai.vercel.app/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fhero_img.ccd4ef64.png&w=3840&q=75",
        width=420
        )

# =====================================================
# DETECT PAGE
# =====================================================
elif st.session_state.page == "detect":

    st.markdown(f"""
    <div class="header">
        <img src="{LOGO_URL}">
        <h1>Hair Disease Detection</h1>
    </div>
    """, unsafe_allow_html=True)

    if st.button("⬅ Back to Home"):

        st.session_state.page="home"
        st.rerun()

    uploaded_file = st.file_uploader(
        "Upload scalp image",
        type=["jpg","jpeg","png"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file)

        st.image(image,width=500)

        processed = preprocess_image(image)

        prediction = model.predict(processed, verbose=0)

        st.write("Prediction Shape:", prediction.shape)

        confidence = float(np.max(prediction))*100
        pred_index = int(np.argmax(prediction))

        if pred_index < len(DISEASES):
            disease = DISEASES[pred_index]
        else:
            disease = "Healthy"

        severity = (
        "High" if confidence>85
        else "Moderate" if confidence>60
        else "Low"
        )

        st.session_state.disease = disease
        st.session_state.confidence = confidence
        st.session_state.severity = severity

        st.markdown(f"""
        <div class='prediction-box'>
        Predicted Disease: {disease}<br>
        Confidence: {confidence:.2f}%<br>
        Severity: {severity}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🩺 Disease Overview")

        st.write(DISEASE_INFO.get(disease,"Description not available."))

        st.markdown("---")

        col1,col2 = st.columns(2)

        with col1:

            if st.button("🌿 Natural Remedies"):

                st.session_state.page="remedies"
                st.rerun()

        with col2:

            if st.button("🧴 Personalized Recommendations"):

                st.session_state.page="products"
                st.rerun()

# =====================================================
# REMEDIES PAGE
# =====================================================
elif st.session_state.page == "remedies":

    st.markdown(
    "<h2 style='color:black;'>🌿 Natural Remedies</h2>",
    unsafe_allow_html=True
    )

    if st.button("⬅ Back"):

        st.session_state.page="detect"
        st.rerun()

    disease = st.session_state.disease
    severity = st.session_state.severity.lower()

    filtered_remedies = remedies_df[
    (remedies_df["disease"].str.strip()==disease) &
    (remedies_df["severity"].str.strip().str.lower()==severity)
    ]

    if not filtered_remedies.empty:

        for _,row in filtered_remedies.iterrows():

            st.markdown(
            f"<div class='treatment-box'>✔ {row['remedy']}</div>",
            unsafe_allow_html=True
            )

    else:

        st.write("No remedies available")

# =====================================================
# PRODUCTS PAGE
# =====================================================
elif st.session_state.page == "products":

    st.markdown(
    "<h2 style='color:black;'>🧴 Personalized Recommendations</h2>",
    unsafe_allow_html=True
    )

    if st.button("⬅ Back"):

        st.session_state.page="detect"
        st.rerun()

    disease = st.session_state.disease
    severity = st.session_state.severity.lower()

    filtered = products_df[
    (products_df["disease"].str.strip()==disease) &
    (products_df["severity"].str.strip().str.lower()==severity)
    ]

    if not filtered.empty:

        for _,product in filtered.iterrows():

            st.image(product["image"],width=250)

            st.write(f"### {product['name']}")

            st.markdown(f"[Buy Now]({product['link']})")

            st.markdown("---")

    else:

        st.write("No products available")

# ===============================
# WARNING
# ===============================
st.warning(
"⚠️ This AI tool is for educational purposes only. Consult a dermatologist."
)