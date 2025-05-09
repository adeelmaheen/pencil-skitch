import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

class PencilSketcher:
    def __init__(self, image: Image.Image):
        self.original_image = image
        self.processed_image = None

    def dodgeV2(self, x, y):
        return cv2.divide(x, 255 - y, scale=256)

    def convert_to_sketch(self):
        img_array = np.array(self.original_image)
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        img_invert = cv2.bitwise_not(img_gray)
        img_smooth = cv2.GaussianBlur(img_invert, (21, 21), sigmaX=0, sigmaY=0)
        self.processed_image = self.dodgeV2(img_gray, img_smooth)
        return self.processed_image

    def get_sketch_bytes(self):
        if self.processed_image is None:
            raise ValueError("Processed image not found. Please run convert_to_sketch() first.")
        # Convert the NumPy array to a PIL Image
        pil_image = Image.fromarray(self.processed_image)
        # Save the PIL Image to a BytesIO object
        buf = BytesIO()
        pil_image.save(buf, format="PNG")
        byte_im = buf.getvalue()
        return byte_im

def main():
    st.title("Pencil Sketcher App")
    st.write("This Web App helps you convert your images to pencil sketches.")

    file_image = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if file_image is None:
        st.write("Please upload an image file.")
    else:
        input_img = Image.open(file_image)
        sketcher = PencilSketcher(input_img)
        final_sketch = sketcher.convert_to_sketch()

        st.write("Image uploaded successfully!")
        st.image(input_img, use_container_width=True)
        st.write("Processing the image...")
        st.image(final_sketch, caption="Pencil Sketch", use_container_width=True)

        # Prepare the sketch image for download
        sketch_bytes = sketcher.get_sketch_bytes()
        st.download_button(
            label="Download Sketch",
            data=sketch_bytes,
            file_name="pencil_sketch.png",
            mime="image/png"
        )

if __name__ == "__main__":
    main()
