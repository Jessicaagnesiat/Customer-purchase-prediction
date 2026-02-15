import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

st.set_page_config(
    page_title="Customer Purchase Prediction",
    page_icon="📊",
    layout="wide"
)

# ======================
# Load Model
# ======================
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ======================
# Custom CSS (Modern Dark Theme)
# ======================
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
h1, h2, h3 {
    color: #FFFFFF;
}
.stMetric {
    background-color: #1E222B;
    padding: 15px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ======================
# Title Section
# ======================
st.title("🧠 Customer Purchase Prediction Dashboard")
st.markdown("### Predict whether a customer will make a future purchase")

st.divider()

# ======================
# Sidebar Inputs
# ======================
st.sidebar.header("📌 Customer Behavior Inputs")

total_session = st.sidebar.number_input("Total Session", 0, 200, 10)
unique_category = st.sidebar.number_input("Unique Category Viewed", 0, 50, 5)
total_action = st.sidebar.number_input("Total Actions", 0, 500, 20)
recency = st.sidebar.number_input("Recency (Days)", 0, 365, 15)
total_spent = st.sidebar.number_input("Total Spent", 0.0, 1000000.0, 50000.0)
add_to_cart = st.sidebar.number_input("Add to Cart Count", 0, 200, 5)
checkout = st.sidebar.number_input("Checkout Count", 0, 200, 2)

cart_conversion_rate = checkout / add_to_cart if add_to_cart != 0 else 0
avg_spent_per_session = total_spent / total_session if total_session != 0 else 0

# ======================
# Prediction Button
# ======================
if st.sidebar.button("🔮 Predict Purchase Probability"):

    input_data = np.array([[
        total_session,
        unique_category,
        total_action,
        recency,
        total_spent,
        add_to_cart,
        checkout,
        cart_conversion_rate,
        avg_spent_per_session
    ]])

    scaled_input = scaler.transform(input_data)
    probability = model.predict_proba(scaled_input)[0][1]
    prediction = (probability > 0.35).astype(int)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Purchase Probability", f"{probability:.2%}")

    with col2:
        if prediction == 1:
            st.success("✅ Likely to Purchase")
        else:
            st.error("❌ Unlikely to Purchase")

    st.divider()

    # Gauge Chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={'text': "Purchase Likelihood"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#00BFFF"},
            'steps': [
                {'range': [0, 35], 'color': "#2E2E2E"},
                {'range': [35, 70], 'color': "#555555"},
                {'range': [70, 100], 'color': "#00BFFF"},
            ],
        }
    ))

    st.plotly_chart(fig, use_container_width=True)