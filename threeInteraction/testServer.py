from flask import Flask, jsonify, request, redirect, url_for, send_from_directory
import os, argparse, json
import subprocess
from flask_cors import CORS


app = Flask(__name__)
CORS(app)



@app.route('/getInfo', methods=['POST'])
def getInfo():

    reqData = request.get_json(force=True, silent=True)

    if reqData == None:
        return app.make_response((jsonify( {'error': 'invalid json'} ), 400, None))

    print(reqData)

    elemID = reqData["elemid"]

    
    newtext = subprocess.check_output("fortune", shell=True).decode().replace("\n", " ")
    print(newtext)

    jsonStr = json.dumps({
        "text": newtext,
        "elemid": elemID
    })

    retResponse = app.make_response((jsonStr, 200, None))
    retResponse.mimetype = "application/json"

    return retResponse



if __name__ == '__main__':

    ap = argparse.ArgumentParser(description='--references genome.fa')
    ap.add_argument('--port', type=int, default=5000, required=False)
    args = ap.parse_args()

    app.run(threaded=True, host="0.0.0.0", port=args.port)