import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta
import matplotlib.pyplot as plt

# load model parameter
model = joblib.load('random_forest_model.pkl')
n_lag = joblib.load('n_lag.pkl')

# load mae,rmse,r2
mae_test = joblib.load('mae.pkl')
rmse_test = joblib.load('rmse.pkl')
r2_test = joblib.load('r2.pkl')

# load data
df1 = pd.read_csv("antam_price.csv")
df2 = pd.read_csv("antam_price2025.csv")
df = pd.concat([df1, df2], axis=0, ignore_index=True)

# Bersihkan data
df['Tanggal'] = pd.to_datetime(df['Tanggal'], errors='coerce')
df.dropna(subset=['Tanggal'], inplace=True)
df = df.drop_duplicates(subset='Tanggal', keep='last')
df = df.sort_values('Tanggal')
df = df[['Tanggal', 'Harga']]
df.set_index('Tanggal', inplace=True)

# n_lag
for i in range(1, n_lag + 1):
    df[f'lag_{i}'] = df['Harga'].shift(i)

df.dropna(inplace=True)

# prediksi
last_known = df.iloc[-n_lag:]['Harga'].values.tolist()
future_predictions = []
future_dates = []
last_date = df.index[-1].to_pydatetime()
future_days = 90

for i in range(future_days):
    input_features = pd.DataFrame([last_known[-n_lag:]], 
                                  columns=[f'lag_{j}' 
                                           for j in range(1, n_lag + 1)])
    next_pred = model.predict(input_features)[0]
    future_predictions.append(next_pred)
    last_known.append(next_pred)
    future_dates.append(last_date + timedelta(days=i + 1))

# df prediksi
future_df = pd.DataFrame({
    'Tanggal': future_dates,
    'Prediksi_Harga': future_predictions
})
future_df['Tanggal_only'] = future_df['Tanggal'].dt.date

# UI
st.title("Prediksi Harga Emas Antam")
st.subheader("Harga Emas Antam Tahun 2010-2025")
st.line_chart(df['Harga'])

tgl = st.date_input(
    "Pilih tanggal prediksi",
    value=future_df['Tanggal_only'].min(),
    min_value=future_df['Tanggal_only'].min(),
    max_value=future_df['Tanggal_only'].max()
)

if tgl in future_df['Tanggal_only'].values:
    harga = future_df.loc[future_df['Tanggal_only'] == tgl, 
                          'Prediksi_Harga'].values[0]
    st.success(f"Prediksi harga emas Antam pada tanggal {tgl} adalah **Rp{harga:,.0f}**")
    st.info(f"MAE (Rata-rata selisih prediksi): Rp{mae_test:,.0f}")
    st.info(f"RMSE (Akar dari rata-rata error): Rp{rmse_test:,.0f}")
    st.info(f"R² Score (Keakuratan model): {r2_test:.4f}")
else:
    st.warning(f"Tanggal tidak tersedia dalam rentang prediksi ({future_df['Tanggal_only'].min()} s.d. {future_df['Tanggal_only'].max()})")

