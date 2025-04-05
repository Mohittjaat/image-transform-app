import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def apply_transformation(image, transformation):
    if transformation == 'Grayscale':
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif transformation == 'Resize':
        return cv2.resize(image, (image.shape[1]//2, image.shape[0]//2))
    elif transformation == 'Blur':
        return cv2.GaussianBlur(image, (15, 15), 0)
    elif transformation == 'Sharpen':
        kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel)
    elif transformation == 'Fourier Transform':
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift))
        return magnitude_spectrum
    return image

def main():
    st.title("Image Transformation App")
    st.write("Upload an image and choose a transformation")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert to OpenCV format
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        transformations = ['Grayscale', 'Resize', 'Blur', 'Sharpen', 'Fourier Transform']
        transformation = st.selectbox("Select a transformation:", transformations)

        if st.button("Apply Transformation"):
            transformed_image = apply_transformation(image, transformation)
            
            if transformation == 'Fourier Transform':
                fig, ax = plt.subplots()
                ax.imshow(transformed_image, cmap='gray')
                ax.set_title('Fourier Transform')
                st.pyplot(fig)
            else:
                st.image(transformed_image, caption=f"Transformed Image - {transformation}", use_column_width=True, channels="BGR" if transformation != 'Grayscale' else "GRAY")

if __name__ == "__main__":
    main()
