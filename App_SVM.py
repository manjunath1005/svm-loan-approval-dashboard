import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# ==========================
# Page Config
# ==========================
st.set_page_config(
    page_title="Smart Loan Approval System",
    page_icon="💳",
    layout="wide"
)

# ==========================
# Custom Dark Styling
# ==========================
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: #fafafa;
}
.block-container {
    padding-top: 2rem;
}
.card {
    background-color: #161b22;
    padding: 1.5rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
}
.badge-approved {
    background-color: #1fdf64;
    color: black;
    padding: 0.5rem 1rem;
    border-radius: 8px;
    font-weight: bold;
}
.badge-rejected {
    background-color: #ff4b4b;
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 8px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ==========================
# Header
# ==========================
st.markdown("## 💳 Smart Loan Approval System")
st.caption("Decision support system for automated credit screening using Support Vector Machines")

# ==========================
# Load Data
# ==========================
@st.cache_data
def load_data():
    data = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")

    data = data.dropna(subset=['Gender', 'Married', 'Loan_Amount_Term'])
    data['LoanAmount'].fillna(data['LoanAmount'].mode()[0], inplace=True)
    data['Credit_History'].fillna(data['Credit_History'].mode()[0], inplace=True)
    data['Self_Employed'].fillna(data['Self_Employed'].mode()[0], inplace=True)
    data['Dependents'].fillna(data['Dependents'].mode()[0], inplace=True)

    encoders = {}
    for col in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        encoders[col] = le

    return data, encoders

data, label_encoders = load_data()

FEATURES = [
    'Gender', 'Married', 'Dependents', 'Education',
    'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome',
    'LoanAmount', 'Loan_Amount_Term', 'Credit_History',
    'Property_Area'
]

X = data[FEATURES]
y = data['Loan_Status']

x_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)

models = {
    "Linear": SVC(kernel="linear", probability=True),
    "Polynomial": SVC(kernel="poly", degree=3, probability=True),
    "RBF": SVC(kernel="rbf", probability=True)
}

for m in models.values():
    m.fit(x_train_scaled, y_train)

# ==========================
# Sidebar – Model Controls
# ==========================
st.sidebar.markdown("### ⚙️ Model Controls")
kernel_choice = st.sidebar.radio(
    "Kernel Strategy",
    ["Linear", "Polynomial", "RBF"]
)

st.sidebar.caption(
    "• Linear: Simple credit rules\n"
    "• Polynomial: Moderate complexity\n"
    "• RBF: Non-linear credit patterns"
)

# ==========================
# Applicant Snapshot
# ==========================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### 📋 Applicant Snapshot")

c1, c2 = st.columns(2)

with c1:
    income = st.number_input("Monthly Income (₹)", min_value=0)
    credit_ui = st.selectbox("Credit Record", ["Clean History", "Bad History"])
    employment_ui = st.selectbox("Employment Category", ["Salaried", "Self Employed"])

with c2:
    loan_amount = st.number_input("Requested Loan (₹)", min_value=0)
    property_ui = st.selectbox("Residence Zone", ["Urban", "Semiurban", "Rural"])
    loan_term = st.selectbox("Loan Duration (Months)", [360, 240, 180, 120])

st.markdown('</div>', unsafe_allow_html=True)

# ==========================
# Encode Inputs
# ==========================
credit = 1 if credit_ui == "Clean History" else 0
employment = "Yes" if employment_ui == "Self Employed" else "No"

input_df = pd.DataFrame([{
    "Gender": label_encoders['Gender'].transform(["Male"])[0],
    "Married": label_encoders['Married'].transform(["Yes"])[0],
    "Dependents": label_encoders['Dependents'].transform(["0"])[0],
    "Education": label_encoders['Education'].transform(["Graduate"])[0],
    "Self_Employed": label_encoders['Self_Employed'].transform([employment])[0],
    "ApplicantIncome": income,
    "CoapplicantIncome": 0,
    "LoanAmount": loan_amount,
    "Loan_Amount_Term": loan_term,
    "Credit_History": credit,
    "Property_Area": label_encoders['Property_Area'].transform([property_ui])[0]
}])

# ==========================
# Prediction
# ==========================
if st.button("🔍 Evaluate Risk", use_container_width=True):

    scaled_input = scaler.transform(input_df)
    model = models[kernel_choice]
    pred = model.predict(scaled_input)[0]
    prob = model.predict_proba(scaled_input).max() * 100

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📊 Risk Summary")

    if pred == 1:
        st.markdown('<span class="badge-approved">✔ APPROVED</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-rejected">✖ REJECTED</span>', unsafe_allow_html=True)

    st.write(f"**Kernel Used:** {kernel_choice}")
    st.write(f"**Confidence Level:** {prob:.2f}%")

    st.markdown(
        f"Based on income stability, credit behavior, and employment profile, "
        f"the applicant is **{'likely' if pred == 1 else 'unlikely'} to repay the loan**."
    )

    st.markdown('</div>', unsafe_allow_html=True)
