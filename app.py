import streamlit as st
import pandas as pd
from pycaret.regression import load_model, predict_model

# Stylizacja i tło aplikacji
st.set_page_config(page_title="Insurance Predictor", page_icon="💸")

st.markdown("""
    <style>
    .stApp {
        background-image: url("https://raw.githubusercontent.com/Jaszczur1969/Insurance_prediction/0ce8099a63a572e91f9e9d209c0004d90a3b7a9e/tlo_reka_2.png");
        background-size: 1000px auto;
        background-position: top right;
        background-attachment: fixed;
        color: #333333;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Przezroczyste tła dla czytelności */
    .css-1d391kg, .css-1d391kg:before {
        background-color: rgba(255, 255, 255, 0.85) !important;
        border-radius: 10px;
        padding: 1rem;
    }

    .st-cf {
        background-color: rgba(255, 255, 255, 0.90);
        border-radius: 10px;
        padding: 1rem;
    }

    /* Styl dla przycisków */
    div.stButton > button {
        background-color: #0078d4;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.4em 0.8em;
        border: none;
        transition: background-color 0.3s;
    }
    div.stButton > button:hover {
        background-color: #005a9e;
    }
    </style>
""", unsafe_allow_html=True)




# Wczytujemy model regresji dla insurance
@st.cache_data
def get_model():
    return load_model("insurance_regression_model")

model = get_model()


st.markdown("<h1 style='text-align: center; color: #2c3e50;'>🛡️ Calculate Your Insurance Rate</h1>", unsafe_allow_html=True)

# Dane wejściowe
with st.sidebar:

    st.markdown("### 📝 Enter the following information:")

    sex = st.radio("Select gender", options=["female", "male"])

    age = st.slider("Select your age", min_value=1, max_value=110, value=50)

    children = st.slider("Select number of children", min_value=0, max_value=7, value=1)

    smoker = st.radio("Select your smoking status", options=["yes", "no"])

    height = st.number_input ("Enter your height (in cm)", min_value=50, value=170, step=1)

    weight = st.number_input ("Enter your weight (in kg)", min_value=6, value=70, step=1)

    BMI_raw = weight / (height * height / 10000)
    BMI = round(BMI_raw, 1)

    region = st.selectbox(
    "Select your region:",
    options=["southeast", "northwest", "northeast", "southwest"])
    
    # Tworzymy DataFrame na podstawie danych wejściowych
    input_df = pd.DataFrame([
        {
            'sex': sex,
            'age': age,
            'children': children,
            'smoker': smoker,
            'bmi': BMI,
            'region': region,
        }
    ])

# Insurance rate prediction button
if st.button("Click to check your insurance rate"):
    prediction = predict_model(model, data=input_df)
    st.session_state['prediction'] = prediction['prediction_label'][0]
    st.session_state['input_df'] = input_df
    st.session_state['BMI'] = BMI
    st.session_state['show_result'] = True  # flaga widoczności
    st.session_state['alt_prediction'] = None  # resetuj alternatywną predykcję
    st.session_state['alt_prediction_bmi'] = None  # reset alternatywnej predykcji dla BMI

# Wyświetl predykcję, jeśli już istnieje
if st.session_state.get('show_result'):
    st.success(f"💰 Estimated insurance rate: {st.session_state['prediction']:,.2f} USD")
    #st.markdown(f"<div style='background-color:#dff0d8;padding:1em;border-radius:10px;'>"
            #f"<h3 style='color:#3c763d;'>💰 Estimated Insurance Rate: {st.session_state['prediction']:,.2f} USD</h3></div>",
            #unsafe_allow_html=True)
    st.subheader("Calculation based on below data:")
    st.write(st.session_state['input_df'])

    # Get more /hide info button
    if 'show_advice' not in st.session_state:
        st.session_state['show_advice'] = False

    button_label = "Hide advice" if st.session_state['show_advice'] else "Learn how to save on your insurance"
    
    if st.button(button_label):
        st.session_state['show_advice'] = not st.session_state['show_advice']

# Funkcja do klasyfikacji BMI
def interpret_bmi(bmi):
    if bmi < 18.5:
        return "🔵 Underweight"
    elif 18.5 <= bmi < 25:
        return "🟢 Normal weight"
    elif 25 <= bmi < 30:
        return "🟡 Overweight"
    else:
        return "🔴 Obesity"


# Porady, jeśli użytkownik kliknął drugi przycisk
if st.session_state.get('show_advice'):
    st.subheader("See what features influence insurance rates the most:")
    st.image("Feature_Importance.png", use_container_width=True)
    
    st.markdown("""
As you can see from the Feature Importance Plot above, the three factors that most affect your insurance are:

- whether or not you smoke  
- your BMI  
- your age

By improving these factors, you can significantly reduce your insurance costs.
""")
    st.subheader("Smoker")

    st.markdown ("""If you are a smoker and decide to quit, your insurance cost may be reduced to the following amount (click the button below):
""")

    if st.button("🚭 Calculate cost if you quit smoking"):
        # Skopiuj dane wejściowe i ustaw smoker="no"
        modified_df = st.session_state['input_df'].copy()
        modified_df.loc[0, 'smoker'] = "no"

        alt_prediction = predict_model(model, data=modified_df)
        st.session_state['alt_prediction'] = alt_prediction['prediction_label'][0]

    # Wyświetl alternatywny wynik jeśli istnieje
    if st.session_state.get('alt_prediction') is not None:
        original = st.session_state['prediction']
        alternative = st.session_state['alt_prediction']
        delta = round(original - alternative, 2)

        st.markdown("---")
        st.success(f"✅ If you quit smoking, your insurance cost could be: {alternative:,.2f} USD")

        if delta > 0:
            st.markdown(f"💸 **Potential savings:** {delta:,.2f} USD per year")
        else:
            st.info("🎉 Widzę że nie jesteś palaczem – gratulacje!")

    st.markdown("---")

    st.subheader ("What exactly is BMI, and why does it matter?")

    st.markdown("""
BMI (Body Mass Index) is the ratio of your weight to the square of your height. It indicates whether you are overweight or obese. This factor also significantly affects your insurance cost. 
""")
    
    st.markdown("---")
    st.info(f"📏 Your height is: **{height} cm**")

    st.markdown ("""Check how your BMI and insurance cost would change if your weight decreased (select your desired weight value in the sidebar on the left, then click the "Check insurance rate" button below):                
""" )

    bmi_category = interpret_bmi(BMI)
    st.metric("Your current BMI is:", BMI)
    st.markdown(f"**BMI category:** {bmi_category}")


     # --- NOWY PRZYCISK DO PREDYKCJI NA PODSTAWIE AKTUALNEGO BMI ---
    if st.button("📉 Calculate insurance rate based on current BMI"):
        modified_df_bmi = st.session_state['input_df'].copy()
        # Podmieniamy bmi na aktualne (wyliczone z bieżącej wagi/wzrostu)
        modified_df_bmi.loc[0, 'bmi'] = BMI

        alt_pred_bmi = predict_model(model, data=modified_df_bmi)
        st.session_state['alt_prediction_bmi'] = alt_pred_bmi['prediction_label'][0]


     # Wyświetlenie wyniku dla aktualnego BMI
    if st.session_state.get('alt_prediction_bmi') is not None:
        original = st.session_state['prediction']
        alternative_bmi = st.session_state['alt_prediction_bmi']
        delta_bmi = round(original - alternative_bmi, 2)

        st.markdown("---")
        st.success(f"✅ Estimated insurance rate based on current BMI: {alternative_bmi:,.2f} USD")

        if delta_bmi > 0:
            st.markdown(f"💸 **Potential savings compared to original:** {delta_bmi:,.2f} USD per year")
        else:
            st.info("ℹ️ No savings compared to the original insurance rate based on BMI.")

    st.markdown("---")

    st.subheader ("Check your total savings")

    # --- DODATKOWY PRZYCISK DO SUMY OSZCZĘDNOŚCI ---
    if st.button("Show total potential savings"):
        delta_smoker = 0
        delta_bmi_val = 0

        if st.session_state.get('alt_prediction') is not None:
            delta_smoker = round(st.session_state['prediction'] - st.session_state['alt_prediction'], 2)

        if st.session_state.get('alt_prediction_bmi') is not None:
            delta_bmi_val = round(st.session_state['prediction'] - st.session_state['alt_prediction_bmi'], 2)

        total_savings = delta_smoker + delta_bmi_val

        if total_savings > 0:
            st.markdown(f"<h1 style='color:green;'>💰 Total Potential Savings: {total_savings:,.2f} USD per year</h1>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h2 style='color:blue;'>ℹ️ No total savings available.</h2>", unsafe_allow_html=True)