from flask import Flask, render_template,request
from flask import jsonify
import json
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.python.keras import regularizers
from keras.layers import Activation, Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import StandardScaler
from keras.models import load_model

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/data')
def get_data():
    data = {'message': 'This is a sample JSON response from the REST API'}
    return jsonify(data)

@app.route('/api/submit', methods=['POST'])
def submit():
    first_term_gpa = float(request.form['first_term_gpa'])
    second_term_gpa = float(request.form['second_term_gpa'])
    first_language = int(request.form['first_language'])
    funding = int(request.form['funding'])
    school = int(request.form['school'])
    fast_track = int(request.form['fast_track'])
    coop = int(request.form['coop'])
    residency = int(request.form['residency'])
    gender = int(request.form['gender'])
    prev_education = int(request.form['prev_education'])
    age_group = int(request.form['age_group'])
    high_school_average = float(request.form['high_school_average'])
    math_score = float(request.form['math_score'])
    english_grade = int(request.form['english_grade'])

    input_data = {
        'first_term_gpa': first_term_gpa,
        'second_term_gpa': second_term_gpa,
        'first_language': first_language,
        'funding': funding,
        'school': school,
        'fast_track': fast_track,
        'coop': coop,
        'residency': residency,
        'gender': gender,
        'prev_education': prev_education,
        'age_group': age_group,
        'high_school_average': high_school_average,
        'math_score': math_score,
        'english_grade': english_grade
    }
    model_pkl_file = "dropout_prediction_model.pkl"
    with open(model_pkl_file, 'rb') as file:
        model = pickle.load(file)

    scaler_pkl ="scaler.pkl"
    with open(scaler_pkl,"rb") as file:
        scaler = pickle.load(file)

    input_data = scaler.transform(np.array([list((input_data.values()))]))
    y_pred_prob = model.predict(input_data)
    y_pred = (y_pred_prob > 0.6).astype(int)

    output_data = []
    output_data.append(list(y_pred_prob[0]))
    output_data.append(list(y_pred[0]))
    data = {
        'probability': str(output_data[0][0]),
        'result': str(output_data[1][0])
    }
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)