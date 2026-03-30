import streamlit as st
import pandas as pd
import pickle
from keras.models import load_model

st.title("Titanic Survival Prediction")

pclass = st.slider('Passenger Class', 1, 3)
sex = st.selectbox('Gender', ['male', 'female'])
sibsp = st.slider('Siblings/Spouses', 0, 8)
parch = st.slider('Parents/Children', 0, 6)
fare = st.slider('Fare', 0, 100)
embarked = st.selectbox('Embarked', ['Cherbourg', 'Queenstown', 'Southampton'])

data = pd.DataFrame({
    'Pclass': [pclass],
    'Sex': [sex],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'Embarked': [embarked]
})

@st.cache_resource
def load_my_model():
    return load_model('model_clean.h5', compile=False)

model = load_my_model()

with open('label_encoder.pkl', 'rb') as file:
    label = pickle.load(file)

with open('onehot_encoder.pkl', 'rb') as file:
    onehot = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

data['Sex'] = label.transform(data['Sex'])

embarked_encoded = onehot.transform(data[['Embarked']])
embarked_df = pd.DataFrame(embarked_encoded, columns=onehot.get_feature_names_out())

data = pd.concat([data, embarked_df], axis=1).drop('Embarked', axis=1)

data[['Pclass','SibSp','Parch','Fare']] = scaler.transform(data[['Pclass','SibSp','Parch','Fare']])

data = data.reindex(columns=[
    'Pclass','Sex','SibSp','Parch','Fare',
    'Embarked_Cherbourg','Embarked_Queenstown','Embarked_Southampton'
])

if st.button('Predict Survival'):
    y = model.predict(data)[0][0]

    st.write('Probability:', y)

    if y >= 0.5:
        st.success("Likely to survive")
    else:
        st.error("Not likely to survive")
