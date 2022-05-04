import pickle
import numpy as np

from flask import Flask, request, jsonify, render_template

from features import cookieCutter

# Set up Flask and bypass CORS
app = Flask(__name__)
model = None
table_data = {}

@app.route('/')
def display_results():
    headings = ["Cookie Names", "HTTP-Only", "Secure"]
    data = []
    for cookie in table_data:
        row_data = [
            cookie['name'],
            cookie['httpOnly'],
            cookie['secure'],
        ]
        data.append(row_data)
    return render_template('result.html', headings=headings, data=data)

# Create the receiver API POST endpoint
@app.route("/RF", methods=["POST"])
def detect_cookies():

    data = request.get_json()
    res = []
    # FIXME: There is a bug where cookieCutter is converting 
    # data to NaN. I cannot replicate the issue, will try later.
    cutData = cookieCutter(data)
    pred = detect(cutData)
    index = np.where(pred == 1)
    for i in index[0]:
        res.append(data[i])
    # Convert to JSON before returning data
    global table_data
    table_data = res
    output = jsonify(res)
    return output

# Create the receiver API POST endpoint
@app.route("/delete", methods=["POST"])
def delete_cookies():
    global table_data
    output = jsonify(table_data)
    return output

def load_model():
    model = pickle.load(open('model.sav', 'rb'))
    return model

def detect(x):
    return model.predict(x)

if __name__ == "__main__":
    model = load_model()
    app.run(host="localhost", port=5000, debug=True)