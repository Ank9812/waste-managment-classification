import streamlit as st
from PIL import Image


import numpy as np
from tensorflow.keras.models import load_model

from keras.applications.resnet  import preprocess_input
import requests
from bs4 import BeautifulSoup


import tensorflow as tf

model = load_model('last_model.h5')
#model test 2 was also good
labels = {'Brique_en_carton': 0,
 'Disposable': 1,
 'Garbage Bag Images': 2,
 'Masques': 3,
 'Metal_can': 4,
 'Paper Carton': 5,
 'Plastic Bag Images': 6,
 'Plastic Bottle': 7,
 'Power bank': 8,
 'bones': 9,
 'cardboard': 10,
 'clothes': 11,
 'fruits_leftover': 12,
 'glass_bottles': 13,
 'green_vege': 14,
 'leftovers': 15,
 'ointment': 16,
 'plastic cup': 17,
 'rotten_fruits': 18,
 'shoes': 19}


Recyclable_waste = ['Brique_en_carton', 'cardboard', 'glass_bottles', 'Metal_can', 'Paper Carton',
                    'Plastic Bottle','tooth_brush' ]

Other_waste=['Garbage Bag Images', 'Paper Bag Images', 'shoes',
                  'Disposable', 'Masques',  'plastic cup'
                ]

Hazardous_Waste= ['cells','ointment','Power bank', ]
Household_waste=[ 'bones',  'green_vege', 'leftovers','rotten_fruits', 'fruits_leftover']



index_to_class = {v: k for k, v in labels.items()}
def processed_img(img_path):
    # Preprocess the image
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)


    # Predict the class
    predictions = model.predict(img_array)

    # Get class with highest probability
    predicted_class = np.argmax(predictions, axis=-1)
    # Map the predicted class index back to the corresponding class label
    predicted_class_name = index_to_class[predicted_class[0]]
    return predicted_class_name



def run():
    st.title("Waste Management Classification ♻️")

    img_file = st.file_uploader("Choose an Image", type=['jpg', "png", "jpeg"])
    if img_file is not None:
        img = Image.open(img_file).resize((250, 250))
        st.image(img)
        image_path = './upload_images/' + img_file.name
        with open(image_path, "wb") as f:
            f.write(img_file.getbuffer())
        if img_file is not None:
            predicted_class_name = processed_img(image_path)
            # Map the predicted class index back to the corresponding class label

            st.success("**Predicted : " + predicted_class_name + '**')

            if predicted_class_name in Other_waste:
                st.info('**Category : Non-Recyclable Waste**')
            elif predicted_class_name in Recyclable_waste:
                st.info('**Category : Recyclable Waste**')
            elif predicted_class_name in Hazardous_Waste:
                st.info('**Category : Recyclable Hazardous Waste**')

            elif predicted_class_name in Household_waste:
                st.info('**Category : Non-Recyclable Household Waste**')












run()


