import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def apply_edge_detection(image, method, threshold1=100, threshold2=200):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if method == "Sobel":
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        return np.sqrt(sobelx**2 + sobely**2).astype(np.uint8)
    elif method == "Canny":
        return cv2.Canny(gray, threshold1, threshold2)
    return gray

def apply_cartoon_effect(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(image, 9, 250, 250)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

def apply_transformation(image, transformation, canny_thresh1=100, canny_thresh2=200, blur_ksize=15):
    if transformation == 'Grayscale':
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif transformation == 'Resize':
        return cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))
    elif transformation == 'Blur':
        k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        return cv2.GaussianBlur(image, (k, k), 0)
    elif transformation == 'Sharpen':
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel)
    elif transformation == 'Fourier Transform':
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        return magnitude_spectrum
    elif transformation == 'Frequency Domain':
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
        magnitude = 20 * np.log(magnitude + 1)
        return magnitude
    elif transformation == 'Edge Detection - Sobel':
        return apply_edge_detection(image, "Sobel")
    elif transformation == 'Edge Detection - Canny':
        return apply_edge_detection(image, "Canny", canny_thresh1, canny_thresh2)
    elif transformation == 'Histogram Equalization':
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.equalizeHist(gray)
    elif transformation == 'Unsharp Masking':
        blur = cv2.GaussianBlur(image, (9, 9), 10.0)
        return cv2.addWeighted(image, 1.5, blur, -0.5, 0)
    elif transformation == 'Sketchify':
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        inv = 255 - gray
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        sketch = cv2.divide(gray, 255 - blur, scale=256)
        return sketch
    elif transformation == 'Cartoon Effect':
        return apply_cartoon_effect(image)
    return image

def main():
    st.set_page_config(page_title="Image Transformer", layout="wide")
    st.title("🖼️ Image Transformation App")
    st.write("Upload an image and apply transformations. The results will appear side by side.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        image_np = np.array(image)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        transformations = [
            'Grayscale', 'Resize', 'Blur', 'Sharpen',
            'Fourier Transform', 'Frequency Domain',
            'Edge Detection - Sobel', 'Edge Detection - Canny',
            'Histogram Equalization', 'Unsharp Masking',
            'Sketchify', 'Cartoon Effect'
        ]

        transformation = st.selectbox("Select a transformation:", transformations)

        # Extra controls if needed
        blur_ksize = st.slider("Blur Kernel Size", 3, 51, 15, step=2) if transformation == 'Blur' else 15
        canny_thresh1 = st.slider("Canny Threshold 1", 0, 300, 100) if transformation == 'Edge Detection - Canny' else 100
        canny_thresh2 = st.slider("Canny Threshold 2", 0, 300, 200) if transformation == 'Edge Detection - Canny' else 200

        if st.button("Apply Transformation"):
            result = apply_transformation(
                image_np.copy(), transformation,
                canny_thresh1, canny_thresh2, blur_ksize
            )

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Original Image")
                st.image(image, use_column_width=True)

            with col2:
                st.subheader(f"Transformed - {transformation}")
                if transformation in ['Fourier Transform', 'Frequency Domain']:
                    fig, ax = plt.subplots()
                    ax.imshow(result, cmap='gray')
                    ax.set_title(transformation)
                    ax.axis('off')
                    st.pyplot(fig)
                elif len(result.shape) == 2:
                    st.image(result, use_column_width=True, channels="GRAY")
                else:
                    st.image(result, use_column_width=True, channels="BGR")

if __name__ == "__main__":
    main()
