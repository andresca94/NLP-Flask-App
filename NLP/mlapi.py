# importing libraries
from flask import Flask, request, render_template
import pickle
import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

"""
import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")
"""

le = LabelEncoder()

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods = ["POST","GET"])
def predict():
    # loading the dataset
    data = pd.read_csv("/Users/andrescarvajal/Desktop/Data Science/two_languages.csv")
    X = data["Text"]
    y = data["language"]
    # label encoding
    y = le.fit_transform(y)

    #loading the model and cv
    model = pickle.load(open("model.pkl", "rb"))
    cv = pickle.load(open("transform.pkl", "rb"))
    lang_predict = "Unknown"
    if request.method == "POST":
        # taking the input
        text = request.form["text"]
        # preprocessing the text
        text = re.sub(r'[!@#$(),\n"%^*?\:;~`0-9]', '', text)
        text = re.sub(r'[[]]', '', text)
        text = text.lower()
        dat = [text]
        # creating the vector
        vect = cv.transform(dat).toarray()
        # prediction
        my_pred = model.predict(vect)
        my_pred = le.inverse_transform(my_pred)
        lang_predict = my_pred[0]

    return render_template("home.html", pred=" The above text is in {}".format(lang_predict))



if __name__ == "__main__":
    app.run(debug=True)
