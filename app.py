import streamlit as st
import pandas as pd
import pickle 

st.title("Titanic Survival Prediction")
pclass = st.slider('ENTER THE PASSENGER CLASSFOR USER', 1, 3)
sex = st.selectbox('SELECT THE GENDER OF THE PASSENGER', ['male', 'female'])
sibsp = st.slider('ENTER THE NUMBER OF SIBLINGS/SPOUSES OF THE PASSENGER', 0, 8)
parch = st.slider('ENTER THE NUMBER OF PARENTS/CHILDREN OF THE PASSENGER', 0, 6)
fare = st.slider('ENTER THE FARE OF THE PASSENGER', 0, 100)
embarked = st.selectbox('SELECT THE EMBARKED LOCATION OF THE PASSENGER', ['Cherbourg', 'Queenstown', 'Southampton'])

data=pd.DataFrame({'Pclass': pclass,'Sex': sex,'SibSp': sibsp,'Parch': parch,'Fare': fare,'Embarked': embarked }, index=[0])

import tensorflow as tf
from keras.models import load_model

tf.keras.utils.disable_interactive_logging()

model = load_model('model.h5', compile=False)

with open('label_encoder.pkl', 'rb') as file:
    label=pickle.load(file)

with open('onehot_encoder.pkl', 'rb') as file:
    onehot=pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler=pickle.load(file)

data['Sex']=label.transform(data['Sex'])
embarked=onehot.transform(data[['Embarked']])
embarked=pd.DataFrame(embarked, columns=onehot.get_feature_names_out())
data=pd.concat([data, embarked], axis=1).drop('Embarked', axis=1)

data[['Pclass','SibSp','Parch','Fare']]=scaler.transform(data[['Pclass','SibSp','Parch','Fare']])

y=model.predict(data)
y=y[0][0]

def chance(y):
    if y>=0.5:
        return "The passenger is likely to survive"
    else:
        return "The passenger is likely to not survive"
    
if st.button('PREDICT SURVIVAL CHANCE'):
    st.write('Probability of Survival',y)
    st.write(chance(y))
