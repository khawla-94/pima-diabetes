# Import the Dependencies
import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("C:\Users\user\Desktop\Projet Cloud\model.pkl", "rb") as f:
    model = pickle.load(f)

# Streamlit User Interface
def main():
    st.title('Pima Indian Diabetes Prediction')
    
    st.write('based on these features predict if you have diabetes or not:')
    pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20)
    glucose = st.number_input('Glucose', min_value=0, max_value=200)
    blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=150)
    skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=100)
    insulin = st.number_input('Insulin', min_value=0, max_value=1000)
    bmi = st.number_input('BMI', min_value=0.0, max_value=60.0)
    diabetes_pedigree_function = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.0)
    age = st.number_input('Age', min_value=0, max_value=120)

    if st.button('Predict'):
        # Prepare input features as numpy array
        features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])
        # Make prediction using loaded model
        prediction = model.predict(features)
        if prediction[0] == 1:
            st.write('Prediction: Diabetic')
        else:
            st.write('Prediction: Not Diabetic')

if __name__ == '__main__':
    main()
