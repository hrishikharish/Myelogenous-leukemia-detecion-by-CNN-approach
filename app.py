from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.utils import img_to_array
import numpy as np
import cv2
import pickle
import imutils
app = Flask(__name__)

# Load Model:-
model1 = load_model("TRAINING_EXPERIENCE.h5")
mlb = pickle.loads(open("mlb.pickle", "rb").read())



def predict_label(img_path):
    image = cv2.imread(img_path)
    #image = cv2.equalizeHist(image)
    output = imutils.resize(image,width=400)
    image = cv2.resize(image, (96, 96))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    proba = model1.predict(image)[0]
    print(proba)
    idxs = np.argsort(proba)[::-1][:1]
    for (i, j) in enumerate(idxs):
            label = "{}: {:.2f}%".format(mlb.classes_[j], proba[j] * 100)
            (mlb.classes_[j])
            return label



# routes
@app.route("/", methods=['GET', 'POST'])
def kuch_bhi():
	return render_template("home.html")


@app.route("/about")
def about_page():
	return "About You..!!!"
@app.route("/submit", methods = ['GET', 'POST'])
def get_hours():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)

	return render_template("home.html", prediction = p, img_path = img_path)





if __name__ =='__main__':
	
	app.run()
