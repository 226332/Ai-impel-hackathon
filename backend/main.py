from flask import Flask,request, redirect, jsonify, send_file
from flask_cors import CORS
import datetime
import os
app = Flask(__name__)
CORS(app)
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


    return send_file(path_to_image, attachment_filename="img.jpg")

app.run()