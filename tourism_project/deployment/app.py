import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Wellness Tourism Purchase Predictor",
    page_icon="✈️",
    layout="wide"
)

st.title("🌿 Visit with Us: Wellness Package Purchase Prediction")
st.markdown("""
Predict whether a prospective customer will purchase the newly introduced
**Wellness Tourism Package** before direct outreach.
""")

# Load Model
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "model.joblib")
    return joblib.load(model_path)

try:
    model = load_model()
    st.sidebar.success("Model loaded successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    st.stop()

st.sidebar.header("Customer Profile & Interaction Inputs")

# Sidebar Input Widgets
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Demographics")
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    gender = st.selectbox("Gender", ["Female", "Male"])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    marital_status = st.selectbox("Marital Status", ["Married", "Single", "Divorced"])
    monthly_income = st.number_input("Monthly Income (₹ / $)", min_value=1000.0, max_value=200000.0, value=25000.0)

with col2:
    st.subheader("Travel Preferences")
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    passport = st.selectbox("Holds Valid Passport?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    own_car = st.selectbox("Owns a Car?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    preferred_property_star = st.selectbox("Preferred Hotel Rating (Stars)", [3.0, 4.0, 5.0])
    number_of_trips = st.number_input("Average Annual Trips", min_value=0.0, max_value=30.0, value=2.0)
    number_of_person_visiting = st.number_input("Total Persons Visiting", min_value=1, max_value=10, value=3)
    number_of_children_visiting = st.number_input("Children (<5 years)", min_value=0.0, max_value=5.0, value=0.0)

with col3:
    st.subheader("Sales Pitch Details")
    typeof_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    duration_of_pitch = st.number_input("Pitch Duration (minutes)", min_value=1.0, max_value=120.0, value=15.0)
    number_of_followups = st.number_input("Number of Follow-ups", min_value=0.0, max_value=10.0, value=3.0)
    pitch_satisfaction_score = st.slider("Pitch Satisfaction Score (1 to 5)", 1, 5, 3)

# Inference Action
if st.button("Predict Purchase Probability", type="primary"):
    input_data = pd.DataFrame([{
        'Age': float(age),
        'TypeofContact': typeof_contact,
        'CityTier': int(city_tier),
        'DurationOfPitch': float(duration_of_pitch),
        'Occupation': occupation,
        'Gender': gender,
        'NumberOfPersonVisiting': int(number_of_person_visiting),
        'NumberOfFollowups': float(number_of_followups),
        'ProductPitched': product_pitched,
        'PreferredPropertyStar': float(preferred_property_star),
        'MaritalStatus': marital_status,
        'NumberOfTrips': float(number_of_trips),
        'Passport': int(passport),
        'PitchSatisfactionScore': int(pitch_satisfaction_score),
        'OwnCar': int(own_car),
        'NumberOfChildrenVisiting': float(number_of_children_visiting),
        'Designation': designation,
        'MonthlyIncome': float(monthly_income)
    }])

    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    st.markdown("---")
    st.subheader("Prediction Result")

    res_col1, res_col2 = st.columns(2)
    with res_col1:
        if pred == 1:
            st.success("🎯 **Likely to Purchase!** (Target Customer)")
        else:
            st.warning("⚠️ **Unlikely to Purchase** (Low Conversion Probability)")

    with res_col2:
        st.metric(label="Purchase Probability", value=f"{prob * 100:.2f}%")
        st.progress(float(prob))
