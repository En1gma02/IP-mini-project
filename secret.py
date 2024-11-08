import cv2
import numpy as np
import streamlit as st
from PIL import Image
import io

# Function to convert text to binary format
def text_to_bin(text):
    return ''.join([format(ord(char), '08b') for char in text])

# Function to convert binary data to text
def bin_to_text(binary_data):
    all_bytes = [binary_data[i:i+8] for i in range(0, len(binary_data), 8)]
    return ''.join([chr(int(byte, 2)) for byte in all_bytes])

# Function to hide text in the image
def encode_text_in_image(image, text):
    max_bytes = image.size // 8
    if len(text) > max_bytes:
        raise ValueError("Text is too long to encode in the image.")
    
    binary_text = text_to_bin(text) + '1111111111111110'  # Delimiter to indicate end
    data_index = 0
    binary_len = len(binary_text)
    
    for row in image:
        for pixel in row:
            for i in range(3):  # Loop over each color channel (B, G, R)
                if data_index < binary_len:
                    pixel[i] = int(format(pixel[i], '08b')[:-1] + binary_text[data_index], 2)
                    data_index += 1

    return image

# Function to extract hidden text from the image
def decode_text_from_image(image):
    binary_data = ""
    
    for row in image:
        for pixel in row:
            for i in range(3):  # Loop over each color channel (B, G, R)
                binary_data += format(pixel[i], '08b')[-1]  # Extract LSB

    delimiter = '1111111111111110'  # Delimiter
    decoded_text = bin_to_text(binary_data.split(delimiter)[0])
    return decoded_text

# Morphological operations functions
def apply_morphological_operation(image_np, operation):
    kernel = np.ones((5, 5), np.uint8)  # Structuring element
    if operation == "Erosion":
        processed_image = cv2.erode(image_np, kernel, iterations=1)
    elif operation == "Dilation":
        processed_image = cv2.dilate(image_np, kernel, iterations=1)
    elif operation == "Opening":
        processed_image = cv2.morphologyEx(image_np, cv2.MORPH_OPEN, kernel)
    elif operation == "Closing":
        processed_image = cv2.morphologyEx(image_np, cv2.MORPH_CLOSE, kernel)
    else:
        processed_image = image_np  # Return the original image if no valid operation
    return processed_image

# Sidebar and Navigation
st.sidebar.title("🔐 Image Steganography & Morphological Operations App")
st.sidebar.markdown("Hide and reveal secret messages in images or apply morphological operations.")
selected_page = st.sidebar.selectbox("Choose a mode", ["Encode Message", "Decode Message", "Morphological Operations"])

# Apply dark/light theme based on selection
st.sidebar.markdown("### Appearance Settings")
theme = st.sidebar.selectbox("Select Theme", ["Light", "Dark"])

if theme == "Dark":
    st.markdown("""
        <style>
            .stApp { background-color: #2e2e2e; color: #f5f5f5; }
            h1, h2, h3, .stButton > button { color: #f5f5f5 !important; }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
            .stApp { background-color: #ffffff; color: #2c3e50; }
            h1, h2, h3, .stButton > button { color: #2c3e50 !important; }
        </style>
    """, unsafe_allow_html=True)

# Main Header
st.markdown('<h1 style="text-align: center; color: #6C63FF;">Image Processing & Steganography App</h1>', unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# Page-specific functionality
if selected_page == "Encode Message":
    st.subheader("Encode a Secret Message into an Image")
    
    uploaded_image = st.file_uploader("Upload an Image to Encode", type=["png", "jpg", "jpeg"])
    message = st.text_input("Enter the secret message to hide")
    
    if uploaded_image and message:
        image = Image.open(uploaded_image)
        image_np = np.array(image)
        
        if st.button("🔒 Encode Message"):
            with st.spinner("Encoding message into image..."):
                try:
                    encoded_image = encode_text_in_image(image_np.copy(), message)
                    st.success("Message encoded successfully!")

                    st.image(encoded_image, caption="Image with Hidden Message")
                    
                    # Provide download link for the encoded image
                    encoded_image_pil = Image.fromarray(encoded_image)
                    buf = io.BytesIO()
                    encoded_image_pil.save(buf, format="PNG")
                    byte_im = buf.getvalue()
                    st.download_button("💾 Download Encoded Image", byte_im, "encoded_image.png", "image/png")
                    
                except ValueError as e:
                    st.error(f"Error: {e}")

elif selected_page == "Decode Message":
    st.subheader("Decode a Secret Message from an Image")
    
    uploaded_image = st.file_uploader("Upload an Encoded Image to Decode", type=["png", "jpg", "jpeg"])

    if uploaded_image:
        image = Image.open(uploaded_image)
        image_np = np.array(image)
        
        if st.button("🔍 Decode Message"):
            with st.spinner("Decoding message from image..."):
                try:
                    hidden_message = decode_text_from_image(image_np)
                    st.success("Message decoded successfully!")
                    st.write("🔓 Hidden Message:", hidden_message)
                except Exception as e:
                    st.error(f"Error decoding message: {e}")

elif selected_page == "Morphological Operations":
    st.subheader("Apply Morphological Operations to an Image")

    uploaded_image = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
    operation = st.selectbox("Choose an Operation", ["Erosion", "Dilation", "Opening", "Closing"])
    
    if uploaded_image and operation:
        image = Image.open(uploaded_image).convert("RGB")
        image_np = np.array(image)
        
        if st.button("Apply Operation"):
            with st.spinner(f"Applying {operation} to image..."):
                processed_image = apply_morphological_operation(image_np, operation)
                st.success(f"{operation} applied successfully!")

                st.image(processed_image, caption=f"Image after {operation}")
                
                # Provide download link for the processed image
                processed_image_pil = Image.fromarray(processed_image)
                buf = io.BytesIO()
                processed_image_pil.save(buf, format="PNG")
                byte_im = buf.getvalue()
                st.download_button(f"💾 Download {operation} Image", byte_im, f"{operation.lower()}_image.png", "image/png")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
