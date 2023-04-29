
# How to run acivate  3.10.8(base"conda)
# right click run code 
# 


import os
import pickle

import cv2
import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

model = load_model('model.h5' ,compile=False )

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Your Page Title
st.title("Face Mask Detector")

# UPload Image 
uploaded_image = st.file_uploader("Chose an Image")



def save_uploaded_image(uploaded_image):

    try:
        with open(os.path.join('uploads',uploaded_image.name),'wb') as f:
            f.write(uploaded_image.getbuffer())

        return (uploaded_image.name)
    except:
        return False

x = save_uploaded_image(uploaded_image)



if uploaded_image is not None:
    col1,col2 = st.columns(2)
    if save_uploaded_image(uploaded_image):
        display_image = Image.open(uploaded_image)

        with col1:
            st.header('Your uploaded image')
            
            m=st.image(display_image,width=300)
   
        def extract_features(img_path,model):
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, 1.3, 5)


            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = img[y:y+h, x:x+w]


            img = cv2.imread(img_path)
            img_crop = img[y:y+h,x:x+w]


            IMAGE_SIZE = 224
            img = cv2.resize(img_crop , (IMAGE_SIZE , IMAGE_SIZE)) # Because hmare model me esi size ke image ja rhi h to hme alwys use size me convert krna hota h

            cv2.imwrite("show/Crop_Image.jpeg", img)



    # Hm bta rhe h model ko ke hm only ek image bhej rhe h so that 

            img = img.reshape((1,IMAGE_SIZE,IMAGE_SIZE,3))
            out = model.predict(img)

            out = out.argmax(axis=-1)
    
            with col2:
                if out == 0:
                    st.header("Your Image like this")
                    st.text("mask_weared_incorrect")
                    display_image = Image.open('show/Crop_Image.jpeg')
                    return st.image(display_image,width=300)
    
                elif out == 1:
                    st.header("Your Image like this")

                    st.text("with_mask")
                    display_image = Image.open('show/Crop_Image.jpeg')
                    return st.image(display_image,width=300)
    
    
                else:
                    st.header("Your Image like this")

                    st.text('without_mask')
                    display_image = Image.open('show/Crop_Image.jpeg')
                    return st.image(display_image,width=300)




        uploaded = 'uploads'
        x = str(x)
        img_path = os.path.join(uploaded,x)
        img_path = str(img_path)
        extract_features(img_path,model)