from flask import Flask, render_template, request
import numpy as np
import cv2
import os
from skimage.metrics import structural_similarity as ssim

app = Flask(__name__, static_folder="E:/New folder/static")
UPLOAD_FOLDER = "E:/New folder/static/uploads"
BW_FOLDER = "E:/New folder/static/bw"
OUTPUT_FOLDER = "E:/New folder/static/outputs"

# Create folders if not exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(BW_FOLDER, exist_ok=True)

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

def colorize_image(image_path, bw_path, output_path):
    image = cv2.imread(image_path)
    
    # Convert to Black & White
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(bw_path, gray_bgr)

    # Colorize Black & White
    normalized = gray_bgr.astype("float32") / 255.0
    lab_image = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)
    resized = cv2.resize(lab_image, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

    L_original = cv2.split(lab_image)[0]
    LAB_colored = np.concatenate((L_original[:, :, np.newaxis], ab), axis=2)
    RGB_colored = cv2.cvtColor(LAB_colored, cv2.COLOR_LAB2BGR)
    RGB_colored = np.clip(RGB_colored, 0, 1)
    RGB_colored = (255 * RGB_colored).astype("uint8")
    
    cv2.imwrite(output_path, RGB_colored)

def get_image_info(image_path):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    mean_L = np.mean(lab_image[:, :, 0])
    mean_A = np.mean(lab_image[:, :, 1])
    mean_B = np.mean(lab_image[:, :, 2])
    return {
        "resolution": f"{width} x {height} px",
        "lab_mean": f"L: {mean_L:.2f}, A: {mean_A:.2f}, B: {mean_B:.2f}"
    }

def calculate_ssim(original_path, colorized_path):
    original = cv2.imread(original_path)
    colorized = cv2.imread(colorized_path)

    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    colorized_gray = cv2.cvtColor(colorized, cv2.COLOR_BGR2GRAY)

    ssim_index = ssim(original_gray, colorized_gray)
    return f"{ssim_index * 100:.2f}%"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filename = file.filename
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            bw_path = os.path.join(BW_FOLDER, filename)
            output_path = os.path.join(OUTPUT_FOLDER, filename)

            file.save(file_path)

            # Process the image
            colorize_image(file_path, bw_path, output_path)

            # Get image info
            original_info = get_image_info(file_path)
            colorized_info = get_image_info(output_path)

            # Calculate SSIM accuracy
            accuracy = calculate_ssim(file_path, output_path)

            return render_template(
                "index.html",
                original_filename=filename,
                bw_filename=filename,
                colorized_filename=filename,
                resolution=original_info["resolution"],
                lab_info_original=original_info["lab_mean"],
                lab_info_colorized=colorized_info["lab_mean"],
                accuracy=accuracy
            )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
