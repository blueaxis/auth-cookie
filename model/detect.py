from flask import Flask, request, jsonify
from flask_cors import CORS
from features import cookieCutter
import pickle
import numpy as np

# Set up Flask and bypass CORS
app = Flask(__name__)
cors = CORS(app)
model = None

# Create the receiver API POST endpoint
@app.route("/RF", methods=["POST"])
def detect_cookies():

    data = request.get_json()
    # Process cookies here where data is an array of cookies
    # Each cookie is a dictionary and have the following format:
    # {"domain": DOMAIN,
    # "expirationDate": EXPIRY,
    # "hostOnly": IS_JAVASCRIPT,
    # "httpOnly": IS_HTTP_ONLY,
    # "name": NAME,
    # "path": PATH,
    # "sameSite": IS_SAMESITE,
    # "secure": IS_SECURE,
    # "storeId": ID,
    # "value":VALUE}
    
    res = []
    # FIXME: There is a bug where cookieCutter is converting 
    # data to NaN. I cannot replicate the issue, will try later.
    cutData = cookieCutter(data)
    pred = detect(cutData)
    index = np.where(pred == 1)
    for i in index[0]:
        res.append(data[i])
    # Convert to JSON before returning data
    output = jsonify(res)
    return output

def load_model():
    model = pickle.load(open('model.sav', 'rb'))
    return model

def detect(x):
    return model.predict(x)

if __name__ == "__main__":
    model = load_model()
    app.run(host="localhost", port=5000, debug=True)