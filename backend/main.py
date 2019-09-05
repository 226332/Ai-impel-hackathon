from flask import Flask,request, redirect, jsonify, send_file, send_from_directory
from flask_cors import CORS
import datetime
import os
app = Flask(__name__)
CORS(app)
PATH_TO_DUPA = f'{os.path.realpath(__file__)}/../models/g12.pkl'
@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/xd', methods=['GET', 'POST'])
def tranform_request():
    mode = request.args.get("type")

    file = request.files['file']
    path_to_image = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"focia{datetime.datetime.now().timestamp()}.jpg")
    file.save(path_to_image)
    ###
    d=os.path.join(os.path.dirname(path_to_image), "/build/img.jpg")
    os.system(f'python3 test.py --img_path "{path_to_image}" --gen_path {PATH_TO_DUPA} --out {d}')

    return jsonify({"ok":"ok"})

@app.route('/<path:path>')
def send_frontend(path):
    return send_from_directory('build', path)

app.run()