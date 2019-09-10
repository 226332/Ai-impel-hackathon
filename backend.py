from flask import Flask,request, redirect, jsonify, send_file, send_from_directory
from flask_cors import CORS
import datetime
import os
app = Flask(__name__)
cors = CORS(app, resources={r"/*":{"origins": "*"}})
PATH_TO_PYTHON_INTERPRETER = 'python3'
PATH_TO_GENERATOR_FIRE = f'./models/g12.pkl'
PATH_TO_GENERATOR_FLORA = f'./models//g21.pkl'
PATH_TO_SCRIPT = f'./test.py'

@app.route('/inference', methods=['POST'])
def tranform_request():
    mode = request.args.get("type")
    model_path = PATH_TO_GENERATOR_FIRE if mode == 'fire' else PATH_TO_GENERATOR_FLORA
    file = request.files['file']
    path_to_image = os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)), "created"), f"received{datetime.datetime.now().timestamp()}.jpg")
    file.save(path_to_image)
    ###
    
    filename  =  f"generated{datetime.datetime.now().timestamp()}.jpg"
    d=os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)), "created"), filename)
    x=f'{PATH_TO_PYTHON_INTERPRETER} {PATH_TO_SCRIPT} --img_path "{path_to_image}" --gen_path {model_path} --out {d}'
    print(x)
    os.system(x)

    return jsonify({"filename":filename})

@app.route('/<path:path>')
def send_frontend(path):
    return send_from_directory('created', path)


app.run(host='0.0.0.0')
