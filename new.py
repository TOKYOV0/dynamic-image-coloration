from flask import Flask, render_template, request
import numpy as np
import cv2
import os

app = Flask(__name__, static_folder="E:/New folder/static")
UPLOAD_FOLDER = "E:/New folder/static/uploads"
OUTPUT_FOLDER = "E:/New folder/static/outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load the pre-trained model
prototxt = "E:/New folder/colorization_deploy_v2.prototxt"
caffe_model = "E:/New folder/colorization_release_v2.caffemodel"
pts_npy = "E:/New folder/pts_in_hull.npy"

net = cv2.dnn.readNetFromCaffe(prototxt, caffe_model)
pts = np.load(pts_npy)
layer1 = net.getLayerNames().index("class8_ab")
layer2 = net.getLayerNames().index("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(layer1 + 1).blobs = [pts.astype("float32")]
net.getLayer(layer2 + 1).blobs = [np.full([1, 313], 2.606, dtype="float32")]

def colorize_image(image_path, output_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    normalized = gray.astype("float32") / 255.0
    lab_image = cv2.cvtColor(normalized, cv2.COLOR_RGB2LAB)
    resized = cv2.resize(lab_image, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))
    L = cv2.split(lab_image)[0]
    LAB_colored = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    RGB_colored = cv2.cvtColor(LAB_colored, cv2.COLOR_LAB2RGB)
    RGB_colored = np.clip(RGB_colored, 0, 1)
    RGB_colored = (255 * RGB_colored).astype("uint8")
    RGB_BGR = cv2.cvtColor(RGB_colored, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, RGB_BGR)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            output_path = os.path.join(OUTPUT_FOLDER, file.filename)
            file.save(file_path)
            colorize_image(file_path, output_path)
            return render_template("index.html", 
                                   grayscale_filename=file.filename, 
                                   colorized_filename=file.filename)
    return render_template("index.html", grayscale_filename=None, colorized_filename=None)

if __name__ == "__main__":
    app.run(debug=True)
