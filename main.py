import os
from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np

for i in os.listdir('./static/uploads'):
	os.remove('./static/uploads/' + i)

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
IMAGE_SHAPE = (224, 224)
overcomes = {'Apple__Apple_scab': 'Captafol', 'Apple_Black_rot': 'Streptomycin', 'Apple_Cedar_apple_rust': 'Immunox, Captan, Mancozeb', 'Apple_healthy': 'The plant is healthy', 'Blueberry_healthy': 'The plant is healthy', 'Cherry(including_sour)__Powdery_mildew': 'Sulphur based fungicide', 'Cherry(including_sour)__healthy': 'The plant is healthy', 'Corn(maize)__Cercospora_leaf_spot Gray_leaf_spot': 'Mancozeb 75%', 'Corn(maize)__Common_rust': 'Mancozeb 75%, Pyraclostrobin', 'Corn_(maize)__Northern_Leaf_Blight': 'Zineb 75%', 'Corn(maize)__healthy': 'The plant is healthy', 'Grape_Black_rot': 'Mancozeb 37%, Myclobutanil 1.55%', 'Grape_Esca(Black_Measles)': 'Trichoderma', 'Grape__Leaf_blight(Isariopsis_Leaf_Spot)': 'Bordeaux misture, Mancozeb, Toprin', 'Grape__healthy': 'The plant is healthy', 'Orange_Haunglongbing(Citrus_greening)': 'Monocrotophos insecticide', 'Peach__Bacterial_spot': 'Copper oxytetreacycline syllit + Captan', 'Peach_healthy': 'The plant is healthy', 'Pepper,_bell_Bacterial_spot': 'Fixed copper', 'Pepper,_bell_healthy': 'The plant is healthy', 'Potato_Early_blight': 'Mancozeb, Chlorothalanil', 'Potato_Late_blight': 'Mancozeb, Chlorothalanil, Copper oxy chloride', 'Potato_healthy': 'The plant is healthy', 'Raspberry_healthy': 'The plant is healthy', 'Soybean_healthy': 'The plant is healthy', 'Squash_Powdery_mildew': 'Sulphur, Stylet oil, Neem oil', 'Strawberry_Leaf_scorch': 'No_measures', 'Strawberry_healthy': 'The plant is healthy', 'Tomato_Bacterial_spot': 'Copper fungicide', 'Tomato_Early_blight': 'Mancozeb, Chlorothalanil, Copper oxy chloride', 'Tomato_Late_blight': 'Chlorothalanil, Mancozeb', 'Tomato_Leaf_Mold': 'Chlorothalanil, Mancozeb, Copper fungicide', 'Tomato_Septoria_leaf_spot': 'Chlorothalanil, Maneb, Mancozeb, Bordeaux mixture', 'Tomato_Spider_mites Two-spotted_spider_mite': 'Neem oil, Pyrethrins, Azaridactin', 'Tomato_Target_Spot': 'No measures', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus': 'Yellow sticky trap', 'Tomato_Tomato_mosaic_virus': 'Yellow sticky trap', 'Tomato__healthy': 'The plant is healthy'}
causes = {0: 'Venturia Inaequalis', 1: 'Diplodia Seriata', 2: 'Gymnosporangium, Juniperi-virginianae',3: '',4: '', 5: 'Podosphaera Clandestina', 6: '', 7:  'Cercospora Zeae-maydis ', 8: 'Puccinia Sorghi',  9: 'Exserohilum Turcicum', 10: '', 11: 'Guignardia Bidwellii',12: 'Phaeoacremonium Aeophilum', 13:'Pseudocercospora Vitis', 14: '', 15: 'Candidatus Liberibacter',  16: 'Xanthomonas Campestris Pv. Pruni',17: '',18: 'Xanthomonas Campestris Pv. Vesicatoria', 19: '', 20: 'Alternaria Solani', 21: 'Phytophthora Infestans', 22: '', 23: '', 24: '', 25: 'Podosphaera Xanthii(infects all cucurbits - including muskmelons)', 26: 'Diplocarpon Earlianum', 27: '', 28: 'Xanthomonas Campesiris Pv, Vesicatoria', 29: 'Alternaria Solani', 30: 'Phytophthora Infestans', 31: 'Cladosporium Fulvum',32: 'Septoria Lycopersici', 33: 'Two-spotted_spider_mite', 34: 'Corynespora Cassiicola', 35: 'Curl_Virus - Tomato yellow leaf curl virus (TYLCV)',36: 'Tomato___Tomato_mosaic_virus', 37: ''}
classes = {0: 'Apple__Apple_scab', 1: 'Apple_Black_rot', 2: 'Apple_Cedar_apple_rust', 3: 'Apple_healthy', 4: 'Blueberry_healthy', 5: 'Cherry(including_sour)__Powdery_mildew', 6: 'Cherry(including_sour)__healthy', 7: 'Corn(maize)__Cercospora_leaf_spot Gray_leaf_spot', 8: 'Corn(maize)__Common_rust', 9: 'Corn_(maize)__Northern_Leaf_Blight', 10: 'Corn(maize)__healthy', 11: 'Grape_Black_rot', 12: 'Grape_Esca(Black_Measles)', 13: 'Grape__Leaf_blight(Isariopsis_Leaf_Spot)', 14: 'Grape__healthy', 15: 'Orange_Haunglongbing(Citrus_greening)', 16: 'Peach__Bacterial_spot', 17: 'Peach_healthy', 18: 'Pepper,_bell_Bacterial_spot', 19: 'Pepper,_bell_healthy', 20: 'Potato_Early_blight', 21: 'Potato_Late_blight', 22: 'Potato_healthy', 23: 'Raspberry_healthy', 24: 'Soybean_healthy', 25: 'Squash_Powdery_mildew', 26: 'Strawberry_Leaf_scorch', 27: 'Strawberry_healthy', 28: 'Tomato_Bacterial_spot', 29: 'Tomato_Early_blight', 30: 'Tomato_Late_blight', 31: 'Tomato_Leaf_Mold', 32: 'Tomato_Septoria_leaf_spot', 33: 'Tomato_Spider_mites Two-spotted_spider_mite', 34: 'Tomato_Target_Spot', 35: 'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 36: 'Tomato_Tomato_mosaic_virus', 37: 'Tomato__healthy'}
export_path = "./PlantDiseaseDet"
reloaded = tf.compat.v1.keras.experimental.load_from_saved_model(export_path, custom_objects={'KerasLayer':hub.KerasLayer})

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_reload(image):
    probabilities = reloaded.predict(np.asarray([image]))[0]
    class_idx = np.argmax(probabilities)
    return class_idx, {classes[class_idx]: probabilities[class_idx]}

def load_image(filename):
    img = cv2.imread(f'./static/uploads/{filename}')
    img = cv2.resize(img, (IMAGE_SHAPE[0], IMAGE_SHAPE[1]))
    img = img /255
    return img


@app.route('/')
def upload_form():
	return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_image():
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		print('upload_image filename: ' + filename)
		image = load_image(filename)
		id, prediction = predict_reload(image)
		if list(prediction.values())[0] > 0.60:
			label = list(prediction.keys())[0]
			if causes[id]:
				flash(f'The plant is affected with "{label}".')
				flash("Causes of this condition:")
				for i, txt in enumerate(causes[id].strip().split(',')):
					flash(str(i + 1) + '. ' + txt.strip()+'.')	
				flash("Treating_measures for this condition:")
				for i, txt in enumerate(overcomes[label].strip().split(',')):
					flash(str(i + 1) + '. ' + txt.strip()+'.')
			else:
				flash("Your Plant is coming out very well. Continue nourishing it to get greater yield from it!")
		else:
			flash("Please provide a Clear and Valid Plant Image!")	
		return render_template('upload.html', filename=filename)
	else:
		flash('Allowed image types are -> png, jpg, jpeg')
		return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run()