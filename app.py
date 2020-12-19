import numpy as np
import pickle
import pandas as pd
import os
import joblib
import streamlit as st 

from PIL import Image


@st.cache
def load_dataset(dataset):
    df = pd.read_csv(dataset)
    return df

gender_label = {'male':0, 'female':1}
race_label = {'group A':0, 'group B':1, 'group C':2, 'group D':3, 'group E':4}
parental_education_label = {'some college':0, "associate's degree":1, 
                            "high school":2, "some high school":3, "bachelor's degree":4, "master's degree":5}
lunch_label = {"standard":0, "free/reduced":1}
test_preparation_label = {"none": 0, "completed": 1}

def get_value(val, my_dict):
    for key, value in my_dict.items():
        if val == key:
            return value

def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val == value:
            return key

def load_prediction_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file), "rb"))
    return loaded_model

def student_performance(gender,race,parental_education,lunch, test_preparation):
   
    prediction = regressor.predict([[gender,race,parental_education,lunch,test_preparation]])
    return prediction



def main():
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Student Performance Prediction App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    menu = ["EDA", "Prediction", "About"]

    choices = st.sidebar.selectbox("Select Activities", menu)

    if(choices == 'EDA'):
        st.subheader("Exploratory Data Analysis")

        data = load_dataset('StudentsPerformance.csv')
        
        if st.checkbox("Show Data"):
            st.dataframe(data.head())

        if st.checkbox("Show Summary"):
            st.write(data.describe())

        if st.checkbox("Show Shape"):
            st.write(data.shape)

    if(choices == 'Prediction'):
        st.subheader('Prediction')

        gender = st.selectbox('Select Gender', tuple(gender_label.keys()))
        race = st.selectbox('Select Race/Ethnicity', tuple(race_label.keys()))
        parental_education = st.selectbox('Select Parental Level of Education', tuple(parental_education_label.keys()))
        lunch = st.selectbox('Select Lunch type', tuple(lunch_label.keys()))
        test_preparation = st.selectbox('Select Test Preparation Course', tuple(test_preparation_label.keys()))

        v_gender = get_value(gender, gender_label)
        v_race = get_value(race, race_label)
        v_parental_education = get_value(parental_education, parental_education_label)
        v_lunch = get_value(lunch, lunch_label)
        v_test_preparation = get_value(test_preparation, test_preparation_label)
        
        pretty_data = {"Gender": gender, "Race":race, "Parental Level of Education":parental_education,
                        "Lunch":lunch, "Test Preparation": test_preparation}

        st.subheader("Options Selected")
        st.json(pretty_data)

        st.subheader("Encoded Data")
        sample_data = [v_gender, v_race, v_parental_education, v_lunch, v_test_preparation]
        st.write(sample_data)

        prep_data = np.array(sample_data).reshape(1,-1)

        model_choice = st.selectbox("Select Model", ["Linear Regression", "Decision Tree Regression"])
        
        if st.button("Predict"):
            if model_choice == "Linear Regression":
                predictor = load_prediction_model("reg1.pkl")
                prediction = predictor.predict(prep_data)

            elif model_choice == "Decision Tree Regression":
                predictor = load_prediction_model("reg2.pkl")
                prediction = predictor.predict(prep_data)

            st.success(prediction)

    

    if(choices == 'About'):
        st.subheader('About')

        st.text("Built with Streamlit")

if __name__=='__main__':
    main()
    
    
    