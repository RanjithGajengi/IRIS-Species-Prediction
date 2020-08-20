# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the SVC Linear CLassifier model
filename = 'SVC-linear-model.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])
                
        data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        my_prediction = classifier.predict(data)
        
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(host="127.0.0.1",port=8080,debug=True)