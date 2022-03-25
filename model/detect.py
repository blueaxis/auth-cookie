from flask import Flask, request, jsonify
from flask_cors import CORS

# Set up Flask and bypass CORS
app = Flask(__name__)
cors = CORS(app)

# Create the receiver API POST endpoint
@app.route("/RF", methods=["POST"])
def detect_cookies():

    data = request.get_json()
    # Process cookies here where data is an array of cookies
    # Each cookie is a dictionary and have the following format:
    # { name: NAME,
    # value: VAL,
    # domain: DOM,
    # expiry: EXP,
    # secure: isSEC,
    # httpOnly: isHTTP,
    # javaScript: isJS }

    # Convert to JSON before returning data
    output = jsonify(data)
    return output

if __name__ == "__main__": 
    app.run(host="localhost", port=5000, debug=True)