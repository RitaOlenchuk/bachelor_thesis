from flask import Flask, jsonify, request, redirect, url_for, send_from_directory
import os, argparse, json
import subprocess
from flask_cors import CORS

dataurl = str(os.path.dirname(os.path.realpath(__file__))) + "/templates/"
print(dataurl)

model = "testInteraction.html"

app = Flask(__name__, static_folder=dataurl, static_url_path='/templates')
CORS(app)

with open('config_files.json') as json_file:
    data = json.load(json_file)

#@app.route('/')
#def showModel():
#    return send_from_directory(app.static_folder, model)

@app.route('/')
def showModel():
    return send_from_directory(app.static_folder, model)

@app.route('/<path:filename>')
def base_static(filename):
    return send_from_directory(app.static_folder, filename)

@app.route('/getInfo', methods=['POST'])
def getInfo():

    reqData = request.get_json(force=True, silent=True)

    if reqData == None:
        return app.make_response((jsonify( {'error': 'invalid json'} ), 400, None))

    print(reqData)

    elemID = reqData["elemid"]

    
    #newtext = subprocess.check_output("fortune", shell=True).decode().replace("\n", " ")
    if elemID == 0:
        newtext = 'Tunica externa'
    elif elemID == 1:
        newtext = 'Tunica media'
    elif elemID == 2:
        newtext = 'Tunica intima'
    elif elemID == 3:
        newtext = 'Plaque'
    elif elemID == 4:
        newtext = 'MSI protein'
        for elem in data:
            if elem["type"] == "MSI":
                path = elem["path"]
                #f not os.path.isfile(path):
                #    os.system("minPlotImzml.py")
                
    elif elemID == 5:
        newtext = 'HE staining'
        for elem in data:
            if elem["type"] == "HE":
                path = elem["path"]
    elif elemID == 6:
        newtext = 'Merged channels'
        for elem in data:
            if elem["type"] == "Merged":
                path = elem["path"]
    elif elemID == 7:
        newtext = 'MSI lipids'
        for elem in data:
            if elem["type"] == "MSI":
                path = elem["path"]
                #if not os.path.isfile(path):
                #    os.system("minPlotImzml.py")
    else:
        newtext = 'Not named'


    print(newtext)

    if elemID>3:
        jsonStr = json.dumps({
            "text": newtext,
            "elemid": elemID,
            "path": path #os.path.relpath(path)
        })
    else:
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
    print(args.port)
    app.run(threaded=True, host="0.0.0.0", port=args.port)