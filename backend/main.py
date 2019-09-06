from flask import Flask,request, redirect, jsonify, send_file, send_from_directory
from flask_cors import CORS
import datetime
import os
app = Flask(__name__)
cors = CORS(app, resources={r"/*":{"origins": "*"}})
PATH_TO_PYTHON_INTERPRETER = 'python'
PATH_TO_DUPA = f'/root/hackathon/project/models/g12.pkl'
PATH_TO_DUPA2 = f'/root/hackathon/project/models/g21.pkl'
PATH_TO_SCRIPT = '/root/hackathon/project/test.py'
@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/xd', methods=['GET', 'POST'])
def tranform_request():
    mode = request.args.get("type")
    model_path = PATH_TO_DUPA if mode == 'green' else PATH_TO_DUPA2
    file = request.files['file']
    path_to_image = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"focia{datetime.datetime.now().timestamp()}.jpg")
    file.save(path_to_image)
    ###
    filename  =  f"focia{datetime.datetime.now().timestamp()}.jpg"
    d=os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)), "build"), filename)
    x=f'{PATH_TO_PYTHON_INTERPRETER} {PATH_TO_SCRIPT} --img_path "{path_to_image}" --gen_path {model_path} --out {d}'
    print(x)
    os.system(x)

    return jsonify({"ok":"ok", "filename":filename})

@app.route('/<path:path>')
def send_frontend(path):
    return send_from_directory('build', path)

app.run(host='0.0.0.0')
