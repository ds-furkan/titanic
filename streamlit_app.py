
import streamlit as st
import pandas as pd
import tensorflow as tf
import pickle
import numpy as np

# Model ve Scaler'ı yükle (Streamlit uygulamasının çalışacağı ortamda bu dosyaların olması gerekir)
try:
    model = tf.keras.models.load_model('my_keras_model.h5')
except Exception as e:
    st.error(f"Keras modelini yüklerken bir hata oluştu: {e}. Model dosyasının Streamlit uygulamasının olduğu dizinde olduğundan emin olun.")
    st.stop()

try:
    with open('standard_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except Exception as e:
    st.error(f"Scaler'ı yüklerken bir hata oluştu: {e}. Scaler dosyasının Streamlit uygulamasının olduğu dizinde olduğundan emin olun.")
    st.stop()

st.title('Titanic Hayatta Kalma Tahmini Uygulaması')
st.write('Kendi bilgilerinizi girerek Titanic kazasında hayatta kalma olasılığınızı öğrenin.')

# Kullanıcıdan bilgi alımı
pclass_options = {'1. Sınıf': 0, '2. Sınıf': 1, '3. Sınıf': 2}
pclass_display = st.selectbox('Bilet Sınıfı', list(pclass_options.keys()))
pclass_value = pclass_options[pclass_display]

sex_options = {'Kadın': 0, 'Erkek': 1}
sex_display = st.selectbox('Cinsiyet', list(sex_options.keys()))
sex_value = sex_options[sex_display]

# Sayısal alanları textbox ile al ve dönüştür
try:
    age = float(st.text_input('Yaş', '30'))
    sibsp = int(st.text_input('Yanınızdaki kardeş/eş sayısı (SibSp)', '0'))
    parch = int(st.text_input('Yanınızdaki ebeveyn/çocuk sayısı (Parch)', '0'))
    fare = float(st.text_input('Bilet Ücreti (Min: 0.0, Max: 500.0)', '30.0')) # Updated here
except ValueError:
    st.error('Lütfen sayısal alanlara geçerli sayılar girin.')
    st.stop()


embarked_options = {'Cherbourg (C)': 0, 'Queenstown (Q)': 1, 'Southampton (S)': 2}
embarked_display = st.selectbox('Biniş Limanı (Embarked)', list(embarked_options.keys()))
embarked_value = embarked_options[embarked_display]

# Tahmin yapma butonu
if st.button('Tahmin Yap'):
    # Girilen bilgileri DataFrame formatına dönüştür
    new_data = pd.DataFrame([{
        'Pclass': pclass_value,
        'Sex': sex_value,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'Embarked': embarked_value
    }])

    # Ölçeklendirme
    scaled_data = scaler.transform(new_data)

    # Tahmin yap
    prediction_probability = model.predict(scaled_data)[0][0]
    
    st.subheader('Tahmin Sonucu:')
    if prediction_probability > 0.5:
        st.success(f'Hayatta Kalma olasılığınız: **%{prediction_probability*100:.2f}** - **Yaşayacaksınız!**')
    else:
        st.error(f'Hayatta Kalma olasılığınız: **%{prediction_probability*100:.2f}** - **Yaşamayacaksınız.**')

