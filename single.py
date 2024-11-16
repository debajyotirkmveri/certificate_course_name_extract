import os
import re
import fitz  # PyMuPDF
import streamlit as st
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv
from google.api_core.exceptions import ResourceExhausted

load_dotenv()  # Load environment variables from .env file

# Configure the Gemini model with the API key from the environment variables
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# Function to sanitize the file name by removing invalid characters
def sanitize_filename(filename):
    # Replace invalid characters with an underscore
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

# Function to save the uploaded file to a specific folder and rename it
def save_and_rename_file(uploaded_file, course_name):
    # Ensure the directory exists
    folder_path = "certificates"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Sanitize the course name to remove invalid characters
    sanitized_course_name = sanitize_filename(course_name)

    # Get the original file extension
    file_extension = uploaded_file.name.split('.')[-1]
    
    # Create the new file name
    new_file_name = f"{sanitized_course_name}.{file_extension}"

    # Save the file in the folder with the new name
    file_path = os.path.join(folder_path, new_file_name)
    
    # Write the file to the disk
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    return file_path

# Function to load the Gemini model and get responses
def get_gemini_response(input_text, image_parts, prompt):
    model = genai.GenerativeModel('gemini-1.5-flash')  # Updated model
    try:
        response = model.generate_content([input_text, image_parts[0], prompt])
        if response.candidates:
            return response.candidates[0].content
        else:
            return ""
    except ResourceExhausted as e:
        st.warning("Quota exceeded for the Gemini API. Please try again later.")
        return ""

# Function to prepare image for Gemini model
def input_image_setup(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image_parts = [
            {
                "mime_type": uploaded_file.type,  # Get the mime type of the uploaded file
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")

# Function to convert PDF to image and adjust brightness
def pdf_to_image(uploaded_file, page_number, brightness_factor=1.5):
    # Save the uploaded PDF file to a temporary location
    temp_pdf_path = "temp_uploaded_file.pdf"
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    doc = fitz.open(temp_pdf_path)
    page = doc.load_page(page_number - 1)
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # Close the document before deleting the temporary file
    doc.close()
    os.remove(temp_pdf_path)

    return img

# Function to extract substrings from text (between curly braces)
def extract_substrings(text):
    pattern = re.compile(r'\{(.*?)\}', re.DOTALL)
    matches = pattern.findall(text)
    stripped_matches = [match.strip() for match in matches]
    return stripped_matches

# Function to extract header text from the response
def extracted_header(text, file_name):
    # Remove the 'text: "' and trailing '"' from the input
    text = re.split(r'text: "', text)[1]
    text = re.split(r'"', text)[0]
    return text

# Initialize Streamlit app
st.set_page_config(page_title="Gemini Application")
st.header("Gemini Application")

# File uploader
uploaded_file = st.file_uploader("Choose an image or PDF...", type=["jpg", "jpeg", "png", "pdf"])

image = None
page_number = 0

if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        temp_pdf_path = "temp_uploaded_file.pdf"
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        doc = fitz.open(temp_pdf_path)
        num_pages = doc.page_count
        doc.close()
        os.remove(temp_pdf_path)
        st.write("Total number of pages in the given pdf is:", num_pages)
        page_number = st.number_input("Select page number", min_value=1, max_value=num_pages, step=1)
        image = pdf_to_image(uploaded_file, page_number)
        st.image(image, caption=f"Uploaded PDF Page {page_number} as Image.", use_container_width=True)
    else:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_container_width=True)

# Set the fixed input prompt
extraction_prompt = """You are an expert in understanding image content. 
You will receive input images of course certificates and answer questions based on them.
To find the answer to 'Course name', you need to return the course header name from the certificate.
"""
input_text = "give me the Course name"

submit = st.button("Process")

# If ask button is clicked
if submit and uploaded_file is not None:
    image_data = input_image_setup(uploaded_file)
    response_text = get_gemini_response(extraction_prompt, image_data, input_text)
    
    # Check if the response text is empty
    if not response_text:
        st.warning("No valid response from the model. Please check the input data and try again.")
    else:
        st.subheader("The Response is")
        st.write(response_text)

        # Assuming response_text is already a string with the data
        text = response_text  # Use response_text directly if it's a string

        # Ensure that 'text' is a string
        if not isinstance(text, str):
            text = str(text)

        st.subheader("Extracted Text:")
        st.write(text)

        # Extract substrings
        substrings = extract_substrings(text)
        st.subheader("Extracted substrings:")
        st.write(substrings)

        # Extract header text
        course_name = extracted_header(substrings[0], uploaded_file.name)
        st.subheader("Extracted header:")
        st.write(course_name)

        # Save and rename the file
        saved_file_path = save_and_rename_file(uploaded_file, course_name)
        st.subheader(f"File saved at: {saved_file_path}")
