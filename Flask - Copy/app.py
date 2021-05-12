from flask import Flask, request, current_app,render_template,send_file,redirect
from PIL import Image
import numpy as np
import io
import base64
import os
from function import build_generator,tensor_to_img
low_resolution_shape = (64,64,3)

path = 'uploaded\\'

app = Flask(__name__)

generator = build_generator();
generator.load_weights("G:\\SGP\\SRGAN\\gen6.h5")

@app.route("/", methods=["GET","POST"])
def processing_img():
	if request.method == "POST":
		file = request.files['image']
		img = Image.open(file.stream)
		img_name = file.filename
		img_low_resolution =[]
		img1 = np.asarray(img)
		img1 = img1.astype(np.float32)
		if img1.shape != low_resolution_shape:
			from skimage.transform import resize                    
			img1 = resize(img1, low_resolution_shape)
		img_low_resolution.append(img1)
		low_resolution_images = np.array(img_low_resolution)
		low_resolution_images = low_resolution_images / 127.5 - 1.
		gen_img = generator(low_resolution_images)
		low_res = tensor_to_img(img1)
		img = tensor_to_img(gen_img[0])
		img.save(path+img_name)
		low = io.BytesIO()
		high = io.BytesIO()
		low_res.save(low,'JPEG')
		img.save(high,"JPEG")
		high_img_data = base64.b64encode(high.getvalue())
		low_img_data = base64.b64encode(low.getvalue())
		return render_template("show.html", high_img_data = high_img_data.decode("utf-8"),low_img_data = low_img_data.decode("utf-8"), location = img_name)

	return render_template("choose.html")


@app.route("/download/<filename>", methods = ['GET', 'POST'])
def download(filename):
	return send_file(path+filename, as_attachment = True)


@app.route("/cancel/<filename>", methods = ['GET', 'POST'])
def cancel(filename):

	os.remove(path+filename)

	return redirect("http://127.0.0.1:5000/", code=302)

if __name__ == "__main__":
    app.run()
