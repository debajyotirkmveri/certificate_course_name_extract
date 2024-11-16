# Import the necessary libraries
import os
import re
import io
import numpy as np
import fitz  # PyMuPDF
from fuzzywuzzy import process, fuzz
import pandas as pd
from PIL import Image
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
#import oracledb
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime
from sqlalchemy import inspect
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from google.api_core.exceptions import ResourceExhausted
import base64


load_dotenv()  # Take environment variables from .env.

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


# Function to load Gemini model and get responses
def get_gemini_response(prompt, image_parts, input_text):
    model = genai.GenerativeModel('gemini-1.5-flash')  # Updated model
    try:
        response = model.generate_content([prompt, image_parts[0], input_text])
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
    
# Function to convert PDF to image
def pdf_to_image(uploaded_file):
    # Save the uploaded PDF file to a temporary location
    temp_pdf_path = "temp_uploaded_file.pdf"
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    doc = fitz.open(temp_pdf_path)
    page = doc.load_page(0)  # Always use the first page
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # Close the document before deleting the temporary file
    doc.close()
    os.remove(temp_pdf_path)

    return img

# Function to display PDF with a caption
@st.cache_data
def displayPDF(file, caption=""):
    base64_pdf = base64.b64encode(file.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)
    if caption:
        st.caption(caption)  # Display the caption below the PDF

#extract substring from the gemini response whatever we need 
def extract_substrings(text):
    # Define the pattern to match substrings starting from '{' and ending at '}'
    pattern = re.compile(r'\{(.*?)\}', re.DOTALL)

    # Find all substrings matching the pattern
    matches = pattern.findall(text)

    # Strip any leading/trailing whitespace from the substrings
    stripped_matches = [match.strip() for match in matches]

    return stripped_matches

# List to store file names with mismatched header and data lengths
mismatch_files = []

def extracted_header(text, file_name):
    # Remove the 'text: "' and trailing '"' from the input
    text = re.split(r'text: "', text)[1]  # Split at 'text: "'
    text = re.split(r'"', text)[0]  # Split at the closing quote to remove it

    return text
    
# Initialize Streamlit app
st.set_page_config(page_title="Certificate  Rename Demo",page_icon="📃")

st.header("Certificate  Processing Solution")

# 

# This code allows users to upload multiple image or PDF files
uploaded_files = st.file_uploader("Choose images or PDFs...", type=["jpg", "jpeg", "png", "pdf"], accept_multiple_files=True)

st.write(f"Number of uploaded files: {len(uploaded_files)}")
extraction_prompt = """You are an expert in understanding image content. 
You will receive input images of course certificates and answer questions based on them.
To find the answer to 'Course name', you need to return the course header name from the certificate.
"""

# Track the state of each file view toggle
if 'file_view_state' not in st.session_state:
    st.session_state['file_view_state'] = {}

# Initialize session state for combined_df
if 'combined_df' not in st.session_state:
    st.session_state['combined_df'] = pd.DataFrame()



# Process each file individually for viewing
for i, uploaded_file in enumerate(uploaded_files):
    file_name = uploaded_file.name
    if file_name not in st.session_state['file_view_state']:
        st.session_state['file_view_state'][file_name] = False

    cols = st.columns([6, 1])
    cols[0].write(file_name)
    if cols[1].button(f"Display", key=f"display_{i}"):
        st.session_state['file_view_state'][file_name] = True

    if st.session_state['file_view_state'][file_name]:
        if uploaded_file.type == "application/pdf":
            # image = pdf_to_image(uploaded_file)
            # st.image(image, caption=f"Uploaded PDF Page as Image for {file_name}.", use_column_width=True)            
            # Display the full PDF
            displayPDF(uploaded_file, caption=f"Uploaded PDF Page for {file_name}")
            # st.image(image, caption=f"Uploaded PDF Page as Image for {file_name}.", use_column_width=True)
        else:
            image = Image.open(uploaded_file)
            st.image(image, caption=f"Uploaded Image for {file_name}.", use_column_width=True)

        #if st.button(f"Close the display image {file_name}", key=f"close_{i}"):
        if st.button(f"Close this display image", key=f"close_{i}"):
            st.session_state['file_view_state'][file_name] = False
            # st.rerun()
            st.experimental_rerun()

input_text = "give me the Course name"

submit_process = st.button("Process")

# Initialize an empty DataFrame
database_df = pd.DataFrame()
# Initialize a list to track files with no valid response
invalid_files = []

if submit_process and uploaded_files:
# if st.button("Process"):
    combined_df = pd.DataFrame()

    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name  # Get the file name


        image_data = input_image_setup(uploaded_file)
        response_text = get_gemini_response(extraction_prompt, image_data, input_text)

        if not response_text:
            st.warning(f"No valid response from the model for {file_name}. Please check the input data.")
            invalid_files.append(file_name)  # Track the invalid file name
        else:
            print(f"The invoice extraction from [{file_name}] is given below")
            if not isinstance(response_text, str):
                response_text = str(response_text)

            substrings = extract_substrings(response_text)

            st.write(f"The Header extraction from [{file_name}] is given below")
            header_name = extracted_header(substrings[0],file_name)
            print(header_name)
            # st.subheader("Extracted header:")
            st.write(header_name)


            # Save and rename the file
            saved_file_path = save_and_rename_file(uploaded_file, header_name)
            st.success(f"File saved at: {saved_file_path}")
            st.write("---*20")

    