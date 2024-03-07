import json
import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app=Flask(__name__)

# Load the model
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['all_data']
    print(data)
    new_Data = np.array(list(data.values())).reshape(1,-1)
    output = model.predict(new_Data)

    if (output == 1):
        output = "Heart Disease"
    elif(output == 0):
        output = "Normal"

    return jsonify(output)


@app.route('/predict', methods=['POST'])
def predict():
    # select and input name 
    gender = request.form['gender']
    ChestPainType = request.form['ChestPainType']
    ecg = request.form['ecg']
    ExerciseAngina = request.form['ExerciseAngina']
    StSlope = request.form['StSlope']

    age = int(request.form['age'])
    BloodPressure = int(request.form['BloodPressure'])
    cholesterol = int(request.form['cholesterol'])
    FastingBloodSugar = int(request.form['FastingBloodSugar'])
    MaximumHeartRate = int(request.form['MaximumHeartRate'])
    OldPeak = float(request.form['OldPeak'])

    # gender (male+female); before it contains 2 value and after 1 
    if (gender == "Male"):  # Male is option value
        gender=1       # gender(male/female) 
    elif (gender == "Female"):
        gender=0

    # Chest Pain Type (other value not assigned because label-encoding)
    if(ChestPainType == "Asymptomatic"):
        ChestPainType=0
    elif(ChestPainType == "AtypicalAngina"):
        ChestPainType=1
    elif(ChestPainType == "NonAnginalPain"):
        ChestPainType=2
    elif(ChestPainType == "TypicalAngina"):
        ChestPainType=3

    # Exercise Angina
    if(ExerciseAngina == "Yes"):
        ExerciseAngina=1
    elif(ExerciseAngina == "No"):
        ExerciseAngina=0

    # Resting Electrocardiogram result 
    if(ecg == "LVH"):
        ecg=0
    elif(ecg == "Normal"):
        ecg=1
    elif(ecg == "ST"):
        ecg=2
    
    # Slope of the peak exercise 
    if(StSlope == "Down"):
        StSlope=0
    elif(StSlope == "Flat"):
        StSlope=1
    elif(StSlope == "Up"):
        StSlope=2           

    # Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope, Age, RestingBP, Cholesterol, FastingBS,  MaxHR,  Oldpeak,  HeartDisease 
    output = model.predict([[gender, ChestPainType, ecg, ExerciseAngina, StSlope, age, BloodPressure, cholesterol, FastingBloodSugar, MaximumHeartRate, OldPeak]])
    
    if (output == 1):
        output = "Heart Disease"
    elif(output == 0):
        output = "Normal"

    return render_template("home.html", prediction_text="The prediction is {}".format(output))

if __name__=="__main__":
    app.run(debug=True)