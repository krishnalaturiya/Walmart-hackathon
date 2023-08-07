from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open(r"model.pkl",'rb'))

@app.route("/")
def about():
    return render_template('home.html')

@app.route("/home")
def home():
    return render_template('home.html')

@app.route("/predict")      
def predict():
    return render_template('predict.html')

@app.route("/aboutUs")      
def aboutUs():
    return render_template('aboutUs.html')

@app.route("/pred", methods=['POST','GET'])
def predi():
   x = [[x for x in request.form.values()]]
   print(x)
   x = np.array(x)
   print(x.shape)
     
   print(x)
   pred = model.predict(x)
   print(pred[0])
   return render_template('submit.html', prediction_text=str(pred))

if __name__ == '__main__':
    app.run(debug=True, port=8000)
