import pandas as pd
import streamlit as st 
from datetime import datetime
import joblib
import numpy as np

st.title("Crypto price prediction")

# st.image('image.jpg')

models_df = pd.read_csv('models_df.csv')

crypto = st.sidebar.selectbox('Select the currency', models_df['name'])

smape = float(models_df[models_df['name'] == crypto].reset_index()['smape'][0])

st.sidebar.write(f"{crypto} has the smape: {smape}")

if smape >0.70:
    st.sidebar.write('This model is currenlty in development mode. please select any other currency')

selected_date = st.sidebar.date_input("Select a date", datetime.today())
specific_date = pd.to_datetime(selected_date)

model_path = f'models/{crypto}.joblib'
model_image_path = f'accuracy_graph/{crypto}.png'

# st.image(model_image_path)
choice = st.sidebar.radio(f"Do you want to see the accuracy grpah for {crypto} model", ['No','Yes'])
if choice == 'Yes':    
    st.image(model_image_path, caption="Model accuracy", use_column_width=True, width=1200)

df = pd.read_csv('crypto_dataset.csv')
min = df[df['crypto_name']==crypto]['close'].min()
max = df[df['crypto_name']==crypto]['close'].max()
median = df[df['crypto_name']==crypto]['close'].median()

if st.button('Predict'):
    model = loaded_model = joblib.load(model_path)
    future_df = pd.DataFrame({'ds': [specific_date]})
    forecast = model.predict(future_df)
    pred = np.exp(float(forecast['yhat'][0]))
    lower = np.exp(float(forecast['yhat_lower'][0]))
    upper = np.exp(float(forecast['yhat_upper'][0]))
    st.write(f"Lower Limit: {lower:.2f}USD **<< Predicted Value: {pred:.2f}USD <<** Upper Limit: {upper:.2f}USD")
    st.write(f"Minimum Value: {min:.2f}USD << Average Value: {median:.2f}USD << Maximum Value: {max:.2f}USD")

# Extract the predicted value for the specific date
# predicted_value = forecast.loc[forecast['ds'] == specific_date, 'yhat'].values[0]
