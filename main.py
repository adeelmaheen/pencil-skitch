from io import StringIO
import pandas as pd
import streamlit as st
import cv2
import numpy as np
from PIL import Image


def dodgeV2(x,y):
    return cv2.divide(x, 255 - y, scale=256)

def pencil_sketch(input_image):
    img_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    img_invert = cv2.bitwise_not(img_gray)
    img_smooth = cv2.GaussianBlur(img_invert, (21, 21), sigmaX=0, sigmaY=0)
    final_img = dodgeV2(img_gray, img_smooth)
    return final_img


st.title("Pencil Sketcher App")
st.write("This Web App is to help you convert your images to pencil sketches.")
file_image = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if file_image is None:
    st.write("Please upload an image file.")

else:
    input_img = Image.open(file_image)
    final_sketch = pencil_sketch(np.array(input_img))
    st.write("Image uploaded successfully!")
    st.image(input_img, use_container_width=True)
    st.write("Processing the image...")
    st.image(final_sketch, caption="Pencil Sketch", use_container_width=True)
    
    
    
    