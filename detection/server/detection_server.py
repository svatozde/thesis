import io


from PIL import Image
from flask import Flask, request
import os
from werkzeug.utils import secure_filename
import cv2
from plastron.plastron_detector import detect_plastron
from plastron import Utils
import base64
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "images"


@app.route('/', methods=['POST'])
def upload_file():
    file = request.files['file']
    filename = secure_filename(file.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)
    img = cv2.imread(path)

    return process(img)

@app.route('/string', methods=['POST'])
def upload_string():
    data = request.get_data()
    img =  toRGB(stringToImage(data))
    return process(img)



def process(img):
    messages = []
    try:
        cropped, croped_intersections, drawn_img, original_predicted_labels, test_labels = detect_plastron(img)

        Utils.drawPolygons(img, original_predicted_labels['plastron'], color=(255, 255, 0))
        Utils.drawPolygons(img, [original_predicted_labels['bottom_part'][0]], color=(255, 0, 0))
        Utils.drawPolygons(img, original_predicted_labels['central_seam'], color=(0, 255, 0))
        Utils.drawPolygons(img, original_predicted_labels['seam_connection'], color=(0, 255, 255))

        if len(original_predicted_labels['seam_connection']) != 5:
            messages.append("Number of central seam junctions is not correct. It must be 5.")

        if len(original_predicted_labels['bottom_part']) < 1:
            messages.append("Missing bottom part")
    except:
        messages.append("Detection failed. Likely there is no plastron on the image")

    success, buffer = cv2.imencode('.jpg', img)
    image_string_b = base64.b64encode(buffer)
    image_string = image_string_b.decode("utf-8")

    return {
        "valid": len(messages) == 0,
        "messages": messages,
        "marked_image": image_string,
    }



# Take in base64 string and return PIL image
def stringToImage(base64_string):
    imgdata = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(imgdata))


def toRGB(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)



if __name__ == "__main__":
    app.run(threaded=False,host="0.0.0.0")