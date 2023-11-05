from flask import Flask, request, app,render_template
from flask import Response
import pickle
import numpy as np
import pandas as pd


application = Flask(__name__)
app=application

scaler=pickle.load(open("/config/workspace/models/scaler.pkl", "rb"))
model = pickle.load(open("/config/workspace/models/scaler.pkl", "rb"))

## Route for homepage

@app.route('/')
def index():
    return render_template('index.html')

## Route for Single data point prediction
@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    result=""

    if request.method=='POST':
        ## male,age,education,currentSmoker,cigsPerDay,BPMeds,prevalentStroke,prevalentHyp,diabetes,totChol,sysBP,diaBP,BMI,heartRate,glucose,TenYearCHD
        male=int(request.form.get("male"))
        age = float(request.form.get('age'))
        education = float(request.form.get('education'))
        currentSmoker = float(request.form.get('currentSmoker'))
        cigsPereDay = float(request.form.get('cigsPerDay'))
        BPMeds = float(request.form.get('BPMeds'))
        prevalentStroke = float(request.form.get('prevalentStroke'))
        prevalentHyp = float(request.form.get('prevalentHyp'))
        diabetes = float(request.form.get('diabetes'))
        totChol = float(request.form.get('totChol'))
        sysBP = float(request.form.get('sysBP'))
        diaBP = float(request.form.get('diaBP'))
        BMI = float(request.form.get('BMI'))
        heartRate = float(request.form.get('heartRate'))
        glucose = float(request.form.get('glucose'))

        new_data=scaler.transform([[male,age,education,currentSmoker,cigsPerDay,BPMeds,prevalentStroke,prevalentHyp,diabetes,totChol,sysBP,diaBP,BMI,heartRate,glucose]])
        predict=model.predict(new_data)
       
        if predict[0] ==1 :
            result = 'Heart Diseases'
        else:
            result ='Non-Heart Diseases'
            
        return render_template('home.html',result=result)

    else:
        return render_template('home.html')


if __name__=="__main__":
    app.run(host="0.0.0.0")