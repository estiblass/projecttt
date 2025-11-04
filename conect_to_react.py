from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
from flask_cors import CORS
import os
from Uploading_picture import cut_image
from new_model import predict
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify(error='No file selected')
    file = request.files['image']
    if file.filename == '':
        return jsonify(error='No file selected')
    file.save('uploads/' + secure_filename(file.filename))
    avg,distances=cut_image('uploads/' + secure_filename(file.filename))
    result = predict()
    distances.reverse()
    print(distances)
    j=0
    for i in range(len(distances)):
        if distances[j]>avg:
            print(distances[j])
            print(result[i])
            result.insert(i+1,' ')
        j=j+1
    print(upload_file)
    print(result)
    return jsonify(result=result)

@app.route('/api/upload', methods=['GET'])
def get_result():

    result = predict()

    return jsonify(result=result)


if __name__ == '__main__':
    app.run()
