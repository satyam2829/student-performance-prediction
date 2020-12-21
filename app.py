import numpy as np
import pickle
import pandas as pd
import os
import joblib
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from PIL import Image


@st.cache(allow_output_mutation=True)
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
        st.header("Data Analysis")

        data = load_dataset('StudentsPerformance.csv')
        data['score'] = data[['math score', 'reading score', 'writing score']].median(axis=1)
        
        if st.checkbox("Show Data"):
            st.dataframe(data.head())

        if st.checkbox("Show Summary"):
            st.write(data.describe())

        if st.checkbox("Show Shape"):
            st.write(data.shape)

        st.header("Data Visualization")

        st.subheader("Univariate Analysis")

        bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

        if st.checkbox("Math Score"):
            st.write(pd.DataFrame(data['math score'].describe()))
            sns.distplot(data['math score'], kde=False, color='red', bins=bins, hist_kws=dict(edgecolor="k"))
            plt.xticks(ticks=bins)
            st.pyplot()

        if st.checkbox("Reading Score"):
            st.write(pd.DataFrame(data['reading score'].describe()))
            sns.distplot(data['reading score'], kde=False, color='red', bins=bins, hist_kws=dict(edgecolor="k"))
            plt.xticks(ticks=bins)
            st.pyplot()

        if st.checkbox("Writing Score"):
            st.write(pd.DataFrame(data['writing score'].describe()))
            sns.distplot(data['writing score'], kde=False, color='red', bins=bins, hist_kws=dict(edgecolor="k"))
            plt.xticks(ticks=bins)
            st.pyplot()

        if st.checkbox("Gender"):
            st.write(pd.DataFrame(data['gender'].value_counts()))
            sns.countplot(data['gender'])
            st.pyplot()

        if st.checkbox("Ethnicity"):
            st.write(pd.DataFrame(data['race/ethnicity'].value_counts()))
            sns.countplot(data['race/ethnicity'])
            st.pyplot()

        if st.checkbox("Parental Level of Education"):
            st.write(pd.DataFrame(data['parental level of education'].value_counts()))
            sns.countplot(data['parental level of education'])
            plt.xticks(rotation=90)
            st.pyplot()

        if st.checkbox("Lunch"):
            st.write(pd.DataFrame(data['lunch'].value_counts()))
            sns.countplot(data['lunch'])
            st.pyplot()

        if st.checkbox("Test Preparation Course"):
            st.write(pd.DataFrame(data['test preparation course'].value_counts()))
            sns.countplot(data['test preparation course'])
            st.pyplot()

        st.subheader("Bi-Variate Analysis")

        if(st.checkbox("Gender and Score")):
            sns.boxplot(x='gender', y='score', data=data)
            st.pyplot()

        if(st.checkbox("Ethnicity and Score")):
            sns.boxplot(x='race/ethnicity', y='score', data=data)
            st.pyplot()

        if(st.checkbox("Parental Education and Score")):
            sns.boxplot(x='parental level of education', y='score', data=data)
            st.pyplot()

        if(st.checkbox("Lunch and Score")):
            sns.boxplot(x='lunch', y='score', data=data)
            st.pyplot()

        if(st.checkbox("Test Preparation Course and Score")):
            sns.boxplot(x='test preparation course', y='score', data=data)
            st.pyplot()

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
        encoded_data = {"Gender":v_gender, "Race":v_race, "Parental Level of Education": v_parental_education,
                        "Lunch": v_lunch, "Test Preparation": v_test_preparation}
        sample_data = [v_gender, v_race, v_parental_education, v_lunch, v_test_preparation]
        st.json(encoded_data)

        prep_data = np.array(sample_data).reshape(1,-1)

        model_choice = st.selectbox("Select Model", ["Linear Regression", "Decision Tree", "Random Forest"])
        
        if st.button("Predict"):
            if model_choice == "Linear Regression":
                predictor = load_prediction_model("reg1.pkl")
                prediction = predictor.predict(prep_data)
                if(prediction > 75):
                    st.balloons()

            elif model_choice == "Decision Tree":
                predictor = load_prediction_model("reg2.pkl")
                prediction = predictor.predict(prep_data)
                if(prediction > 75):
                    st.balloons()

            elif model_choice == "Random Forest":
                predictor = load_prediction_model("reg3.pkl")
                prediction = predictor.predict(prep_data)

            st.success(prediction)

    

    if(choices == 'About'):
        st.subheader('About')

        st.text("Built with Streamlit")

if __name__=='__main__':
    main()
    
    
    