import streamlit as st
import os
from PIL import Image
import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from local_utils import load_data_from_folders, preprocess_image, save_uploaded_file, CulturalClassifier, TextClassifier, load_text_content
import json
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import base64
from swecha_api import get_swecha_search_results, upload_to_swecha, swecha_client

# Set page configuration
st.set_page_config(
    page_title="తెలుగు సాంస్కృతిక వారసత్వం",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Function to set background image
def set_background_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    
    bg_css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded_string}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(bg_css, unsafe_allow_html=True)

# --- CSS Styling ---
st.markdown("""
<style>
    /* Hide all sidebar elements completely, including the expander button */
    [data-testid="stSidebar"],
    [data-testid="stSidebarNav"],
    [data-testid="collapsedControl"],
    [data-testid="stSidebar"][aria-expanded="true"],
    [data-testid="stSidebar"][aria-expanded="false"],
    [role="complementary"],
    [data-testid="stHeader"] .css-1dp5vir {
        display: none !important;
        visibility: hidden !important;
        position: absolute !important;
        transform: translateX(-100%) !important;
    }

    /* NEW: Image grid styling for search results */
    .image-results-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
        gap: 20px;
        margin-bottom: 20px;
    }

    /* NEW: Image card styling */
    .image-card {
        background: white;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }

    .image-card:hover {
        transform: translateY(-5px);
    }

    .image-card img {
        width: 100%;
        height: 250px;
        object-fit: cover;
    }

    .image-caption {
        padding: 12px;
        text-align: center;
        font-weight: 600;
        color: #2c3e50;
    }

    /* NEW: Centered content for text and video results */
    .centered-content {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        width: 100%;
    }
    /* Remove header padding and margin for full page coverage */
    .stApp > header {
        display: none !important;
    }

    /* Main container adjustments */
    .main {
        padding: 0 !important;
        margin: 0 !important;
        width: 100% !important;
        min-height: 100vh !important;
        background: transparent !important;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: flex-start;
        overflow: auto !important;
    }

    /* UPDATED: Main header styling - brown color */
    .main-header {
        font-size: 3.5rem !important;
        color: #5D4037 !important;
        text-align: center;
        padding: 1.5rem 1rem;
        margin-top: 4rem;
        margin-bottom: 1.5rem;
        font-weight: 800;
        text-shadow: 2px 2px 8px rgba(255,255,255,0.7);
        white-space: nowrap;
        overflow: hidden;
        line-height: 1.2;
        letter-spacing: 1px;
        width: 100%;
        background-color: transparent !important;
        border-radius: 10px;
    }

    /* UPDATED: Page header styling for other pages - transparent background */
    .page-header {
        font-size: 2.5rem !important;
        color: #5D4037 !important;
        text-align: center;
        padding: 1rem;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
        font-weight: 700;
        text-shadow: 2px 2px 6px rgba(255,255,255,0.7);
        width: 100%;
        background-color: transparent !important;
        border-radius: 10px;
    }

    /* UPDATED: Sub header styling - transparent background */
    .sub-header {
        font-size: 1.8rem !important;
        color: #5D4037 !important;
        text-align: center;
        padding: 0.5rem 1rem;
        margin-bottom: 1.5rem;
        font-weight: 600;
        width: 100%;
        background-color: transparent !important;
        border-radius: 10px;
        text-shadow: 1px 1px 4px rgba(255,255,255,0.7);
    }

    /* Category grid styling - all buttons in one line */
    .category-grid {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin-bottom: 1.5rem;
        flex-wrap: nowrap;
        width: 100%;
        max-width: 900px;
    }

    /* Upload button container - centered below the 4 options */
    .upload-button-container {
        display: flex;
        justify-content: center;
        margin: 1rem 0;
        width: 100%;
    }
    
    /* UPDATED: Search bar styling - smaller and centered */
    .search-container {
        display: flex !important;;
        flex-direction: column !important;;
        align-items: center !important;;
        justify-content: center !important;;
        width: 60% !important;;
        max-width: 800px !important;;
        margin: 0 auto 0.5rem auto !important;;
        background-color: transparent !important;
        padding: 0 !important;
        box-shadow: none !important;
    }
        
    .search-container > div {
        display: flex !important;
        justify-content: center !important;
        width: 100% !important;
    }

    .search-container .stTextInput > div,
    .search-container .stSelectbox > div {
        width: 60% !important;
        margin: 0 auto !important;
        display: flex !important;
        justify-content: center !important;
    }
    div[data-testid="stTextInput"] label {
        font-size: 1.3rem !important;
        font-weight: 700 !important;
        color: #5D4037 !important;
        margin-bottom: 8px !important;
        display: flex !important;
        text-align: center !important;
        width: 100% !important;
        align-items: center !important;
        justify-content: center !important;
    }
    
    .search-container div[data-testid="stTextInput"] label,
    .search-container div[data-testid="stSelectbox"] label {
        font-size: 1.3rem !important;
        font-weight: 700 !important;
        color: #5D4037 !important;
        margin-bottom: 8px !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        width: 100% !important;
        text-align: center !important;
    }

    /* Content display styling - FIXED: Remove white background */
    .content-display {
        background-color: transparent !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin: 0 auto 20px auto !important;
        width: 100%;
        max-width: 1000px;
    }

    /* UPDATED: Results container styling - transparent background */
    .results-container {
        padding: 20px;
        background-color: rgba(255, 255, 255, 0.85);
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        width: 100%;
        max-width: 1200px;
        margin: 0 auto 20px auto;
        max-height: 65vh;
        overflow-y: auto;
        position: relative;
        z-index: 1;
    }
            
    div[data-testid="stTextInput"] label,
    div[data-testid="stSelectbox"] label {
        font-size: 1.3rem !important;
        font-weight: 700 !important;
        color: #5D4037 !important;
        margin-bottom: 8px !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        width: 100% !important;
        text-align: center !important;
    }

    .search-container .stTextInput,
    .search-container .stSelectbox {
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        width: 100% !important;
    }


    /* Ensure all content inside stays within the container */
    .results-container > div {
        max-width: 100%;
        overflow: hidden;
    }

    /* Fix for image display in results */
    .results-container .stImage {
        max-width: 100%;
    }

    /* Fix for video display in results */
    .results-container .stVideo {
        max-width: 100%;
    }

    /* Fix for text content in results */
    .results-container .text-content {
        max-width: 100%;
        overflow-wrap: break-word;
    }

    /* Custom scrollbars for content areas only */
    .results-container::-webkit-scrollbar {
        width: 8px;
    }

    .results-container::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.05);
        border-radius: 10px;
    }

    .results-container::-webkit-scrollbar-thumb {
        background: rgba(0, 0, 0, 0.2);
        border-radius: 10px;
    }

    .results-container::-webkit-scrollbar-thumb:hover {
        background: rgba(0, 0, 0, 0.3);
    }

    /* Hide any other streamlit elements */
    .stAlert,
    .stSpinner,
    .stException {
        display: none !important;
    }

    /* UPDATED: General button styling to ensure dark brown background and white text */
    .stButton > button,
    .category-button,
    .content-type-button,
    .back-button-style,
    .upload-button,
    .search-button,
    .st-eb,
    .st-ec,
    .st-ed,
    .stRadio > div > label,
    div[role="radiogroup"] label,
    button {
        background: linear-gradient(145deg, #5D4037 0%, #3e2723 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        border: none !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2) !important;
        transition: all 0.3s ease !important;
    }

    /* Button hover and active states */
    .stButton > button:hover,
    .stButton > button:active,
    .category-button:hover,
    .content-type-button:hover,
    .back-button-style:hover,
    .upload-button:hover,
    .search-button:hover,
    .st-eb[aria-checked="true"],
    .st-ec[aria-checked="true"],
    .st-ed[aria-checked="true"],
    .stRadio > div > label:hover,
    .stRadio > div > label[data-testid="stRadio"] {
        background: linear-gradient(145deg, #7C6656 0%, #A0836C 100%) !important;
        transform: translateY(-2px);
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
        color: white !important;
    }

    .stSelectbox [data-testid="stMarkdownContainer"] p {
        color: #5D4037 !important;
    }

    /* Dropdown menu background */
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #f5e8d3 !important;
        border-radius: 8px !important;
        border: 1px solid #5D4037 !important;
    }

    div[data-baseweb="popover"] {
        border: none !important;
        border-radius: 6px !important;
        overflow: hidden;
    }
    /* Dropdown options background */
    div[data-baseweb="popover"] div {
        background-color: #f5e8d3 !important;
        color: #5D4037 !important;
        border: none !important;
        box-shadow: none !important;
    }
    div[data-baseweb="popover"] div > div {
        border: none !important;
        box-shadow: none !important;
    }

    /* Dropdown hover effect */
    div[data-baseweb="popover"] div:hover {
        background-color: #e0d0b9 !important;
        border: none !important;  /* Ensure no border on hover */
    }

    /* Ensure text is visible in dropdown */
    .stSelectbox option {
        background-color: #f5e8d3 !important;
        color: #5D4037 !important;
    }
    div[data-baseweb="select"] > div {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: #4B2E05 !important; 
        border-radius: 6px !important;
        padding: 0.25rem 0.5rem !important;
        font-size: 1rem;
        width: 100%;
    }

    /* Style the actual select field */
    div[data-baseweb="select"] > div > div {
        border: none !important;
        background-color: transparent !important;
        box-shadow: none !important;
    }

    /* Remove focus outline and shadow */
    div[data-baseweb="select"] > div > div:focus {
        outline: none !important;
        box-shadow: 0 0 3px 1px rgba(139, 69, 19, 0.5) !important;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="select"] > div > div,
    div[data-baseweb="select"] > div > div > div {
        border: none !important;
        background-color: transparent !important;
        box-shadow: none !important;
    }
    /* Make the actual dropdown box smaller */
    .search-container .stSelectbox select {
        height: 32px !important;     /* smaller height */
        font-size: 0.9rem !important;
        padding: 4px 10px !important;
        width: 100% !important;     /* fixed width */
        border-radius: 6px !important;
        border: none !important;
    }
            
    /* Specific button dimensions */
    .category-button {
        width: 140px !important;
        height: 70px !important;
        font-size: 1rem !important;
    }

    .content-type-button {
        width: 120px !important;
        height: 60px !important;
        font-size: 1rem !important;
    }

    /* UPDATED: Small button styling for back and search buttons */
    .small-button {
        width: 200px !important;
        height: 40px !important;
        font-size: 1rem !important;
    }

    /* UPDATED: Container for small buttons to be centered */
    .small-button-container {
        display: flex;
        justify-content: center;
        margin: 10px 0;
        width: 100%;
    }

    /* UPDATED: Back and Search button styles */
    .back-button-style, .search-button, .upload-button {
        width: 200px !important;
        height: 40px !important;
        font-size: 1rem !important;
        padding: 0 !important;
    }

    /* Column adjustments to ensure buttons stay in one line */
    .stColumn {
        flex: 1;
        min-width: 0;
    }

    /* Ensure content doesn't get cut off */
    .element-container {
        max-width: 100% !important;
    }

    /* Fix for any Streamlit containers that might cause overflow */
    .st-emotion-cache-1y4p8pa {
        width: 100% !important;
        max-width: 100% !important;
        padding: 0 1rem !important;
    }

    /* Custom container for the button grid */
    .button-grid-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        width: 100%;
        margin-bottom: 1rem;
    }

    /* Hide the hamburger menu completely */
    .css-1vq4p4l {
        display: none !important;
    }

    /* Remove any potential scrollbars from the main page */
    .stAppViewContainer {
        overflow: auto !important;
    }

    /* Image grid styling */
    .image-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
        gap: 20px;
        margin-bottom: 20px;
    }

    /* Text content styling */
    .text-content {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        border-left: 4px solid #3498db;
    }

    /* Video container styling */
    .video-container {
        margin-bottom: 20px;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }

    /* UPDATED: Centered container for back button - smaller */
    .centered-button {
        display: flex;
        justify-content: center;
        margin-top: 15px;
        margin-bottom: 10px;
        background-color: transparent !important;
        padding: 5px 0 !important;
    }

    /* UPDATED: Radio button styling - dark brown with white text */
    .stRadio > div {
        flex-direction: row;
        display: flex;
        gap: 10px !important;
        justify-content: center;
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    .stRadio > div > label {
        background: linear-gradient(145deg, #5D4037 0%, #3e2723 100%) !important;
        color: white !important;
        padding: 8px 16px !important;
        border-radius: 10px;
        margin: 0 2px !important;
        cursor: pointer;
        transition: all 0.3s ease;
        border: none !important;
        height: auto !important;
        min-height: unset !important;
        font-weight: 600;
    }

    .stRadio > div > label:hover {
        background: linear-gradient(145deg, #3e2723 0%, #5D4037 100%) !important;
        transform: translateY(-2px);
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
        color: white !important;
    }

    .stRadio > div > label[data-testid="stRadio"] {
        background: linear-gradient(145deg, #3e2723 0%, #5D4037 100%) !important;
        color: white !important;
        font-weight: bold;
    }

    /* Remove white background from radio button container */
    div[data-testid="stRadio"] {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    /* Fix for the radio button selection box */
    div[role="radiogroup"] {
        background-color: transparent !important;
        border: none !important;
        padding: 0 !important;
        margin: 0 !important;
        gap: 5px !important;
    }

    /* Custom styling for file uploader */
    .stFileUploader > div {
        background-color: rgba(255, 255, 255, 0.7) !important;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
    }

    /* Custom styling for select boxes */
    .stSelectbox > div {
        background-color: #D2B48C !important;  /* light brown background */
        border: 1px solid #8B4513 !important;  /* brown border */
        box-shadow: 0 0 3px 1px rgba(139, 69, 19, 0.3) !important; /* subtle brown glow */
        border-radius: 6px !important;  /* rounded corners */
        padding: 8px 12px !important;  /* padding around input */
        max-width: 400px;
        margin: 0 auto !important;  /* center horizontally */
        display: block !important;
    }

    /* Fix for the content display area to remove extra white space */
    div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stRadio"]) {
        background-color: transparent !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    /* Remove extra padding from radio button container */
    div.element-container:has(> div[data-testid="stRadio"]) {
        background-color: transparent !important;
        padding: 0 !important;
        margin: 0 0 5px 0 !important;
    }

    /* Light background for radio options */
    .st-eb, .st-ec, .st-ed {
        background: linear-gradient(145deg, #5D4037 0%, #3e2723 100%) !important;
        border-radius: 10px !important;
        padding: 5px 15px !important;
        margin: 2px !important;
        border: none !important;
        color: white !important;
        font-weight: 600 !important;
    }

    /* Selected radio option */
    .st-eb[aria-checked="true"], .st-ec[aria-checked="true"], .st-ed[aria-checked="true"] {
        background: linear-gradient(145deg, #3e2723 0%, #5D4037 100%) !important;
        font-weight: bold !important;
        color: white !important;
    }

    /* Remove the white box that appears below content */
    div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div[data-testid="stVerticalBlock"] > div[data-testid="element-container"]) {
        background-color: transparent !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    /* Fix for results container to prevent white box issue */
    .stMarkdown:has(> .results-container) {
        background-color: transparent !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    /* Ensure results container has proper z-index */
    .results-container {
        z-index: 10;
        position: relative;
    }

    /* Remove any white background from the main container */
    .block-container {
        background-color: transparent !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    /* Fix for spacing after category buttons */
    .stButton button {
        margin: 2px !important;
        padding: 8px 12px !important;
    }

    /* Fix for spacing in content type selection */
    .stRadio > div {
        margin-bottom: 5px !important;
    }

    /* Remove extra padding from content display */
    .content-display .element-container {
        padding: 0 !important;
        margin: 0 !important;
    }

    /* Fix for spacing in search container */
    .search-container .stTextInput > div {
        background: white !important;
        padding: 8px 15px !important;
        margin: 0 auto !important;
        width: 60 !important;
        max-width: 500px; 
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

            
            
    /* Fix for vertical spacing in the content selection page */
    div[data-testid="stVerticalBlock"] {
        gap: 0.5rem !important;
    }

    /* Adjust the expander to fit within the container */
    .results-container .streamlit-expanderHeader {
        font-size: 1.2rem;
        color: #5D4037;
    }

    /* Adjust the column layout for images */
    .results-container .stColumn {
        flex: 0 0 48%;
        margin: 1%;
    }
            
    /* UPDATED: Warning message text color */
    .stWarning {
        color: #5D4037 !important;
    }

    /* UPDATED: Success message text color */
    .stSuccess {
        color: #5D4037 !important;
    }

    /* UPDATED: Error message text color */
    .stError {
        color: #5D4037 !important;
    }

    /* UPDATED: Info message text color */
    .stInfo {
        color: #5D4037 !important;
    }
    
    /* Image styling */
    .stImage > img {    
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-bottom: 15px;
    }

    .stImage > div {
        text-align: center;
        font-weight: 600;
        color: #2c3e50;
        margin-top: 5px;
    }

    /* Fix for the white space below radio buttons */
    .content-display > div:has(> div[data-testid="stRadio"]) {
        background-color: transparent !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    /* Target the specific container that holds the radio buttons */
    div[data-testid="stVerticalBlock"] > div > div[data-testid="stVerticalBlock"] > div[data-testid="element-container"]:has(div[data-testid="stRadio"]) {
        background-color: transparent !important;
        padding: 0 !important;
        margin: 0 !important;
        border: none !important;
        box-shadow: none !important;
    }

    /* Ensure the radio button group has no extra spacing */
    div[role="radiogroup"] {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    /* Remove any extra spacing from the radio button labels */
    .stRadio > div > label {
        margin: 0 2px !important;
    }

    /* Ensure the content display area has proper spacing */
    .content-display > div {
        margin-bottom: 5px !important;
    }

    /* UPDATED: Search input label styling - bigger and bolder */
    .stTextInput label {
        font-size: 1.3rem !important;
        font-weight: 700 !important;
        color: #5D4037 !important;
        margin-bottom: 8px !important;
        display: block;
        text-align: center;
        width: 100%;
    }

    /* UPDATED: Search input field styling */
    .stTextInput input {
        font-size: 1rem !important;
        padding: 8px 12px !important;
        border-radius: 8px !important;
        border: 2px solid #5D4037 !important;
        color: #5D4037 !important;
        background-color: transparent !important;
        width: 60% !important;
        margin: 0 auto !important;
        display: block !important;
        height: 40px !important;
    }
            
    .stSelectbox select {
        border: none !important;
        background-color: transparent !important;
        box-shadow: none !important;
    }
            /* Remove focus styles that might show borders */
    div[data-baseweb="select"] > div:focus,
    div[data-baseweb="select"] > div > div:focus {
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
    }

    /* UPDATED: Button container for search and back buttons */
    .button-row {
        display: flex;
        justify-content: center;
        gap: 15px;
        margin: 10px 0;
        width: 100%;
    }

    /* UPDATED: Small buttons for search and back */
    .small-button {
        width: 200px !important;
        height: 40px !important;
        font-size: 1rem !important;
        padding: 0 !important;
    }
        /* Reduce the white background size of the search bar */
    .search-container .stTextInput > div {
        background: transparent !important;  /* remove white */
        padding: 0 !important;               /* remove padding */
        margin: 0 auto !important;
        width: fit-content !important;       /* shrink to fit */
    }
            
     /* NEW: Brown placeholder for dropdown */
    .stSelectbox [data-testid="stMarkdownContainer"] p {
        color: #5D4037 !important;
    }

    div[data-baseweb="input"] {
        background-color: #D2B48C !important;  /* light brown background */
        border: 2px solid #8B4513 !important;  /* brown border */
        box-shadow: 0 0 5px 1px rgba(139, 69, 19, 0.4) !important; /* subtle brown glow */
        border-radius: 6px !important;  /* rounded corners */
        padding: 8px 12px !important;  /* padding around input */
        max-width: 400px;
        margin: 0 auto !important;  /* center horizontally */
        display: block !important;
    }
    /* Remove background, border, shadow from the base-input wrapper */
    div[data-baseweb="base-input"] {
        background-color: #D2B48C !important;  /* light brown background */
        border: 2px solid #8B4513 !important;  /* brown border */
        border-radius: 6px !important;
        padding: 0 !important;  /* no padding here */
        box-shadow: none !important;
    }

    /* Style the actual input field */
    div[data-baseweb="base-input"] > input {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color:  #4B2E05 !important;  /* white text */
        padding: 0.25rem 0.5rem !important;
        font-size: 1rem;
        width: 100%;
    }
    /* Remove focus outline and shadow */
    div[data-baseweb="base-input"] > input:focus {
        outline: none !important;
        box-shadow: 0 0 5px 2px rgba(139, 69, 19, 0.7) !important;
    }
            
    /* Make the actual input box smaller */
    .search-container .stTextInput input {
        height: 32px !important;     /* smaller height */
        font-size: 0.9rem !important;
        padding: 4px 10px !important;
        width: 100% !important;     /* fixed width instead of 60% */
        border-radius: 6px !important;
        border: 1px solid #5D4037 !important;
    }
    .search-container .stTextInput label {
        font-size: 1.3rem !important;
        font-weight: 700 !important;
        color: #5D4037 !important;
        margin-bottom: 8px !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        width: 100% !important;
        text-align: center !important;
    }
        /* NEW: Center all content in results */
    .results-container .stImage,
    .results-container .stVideo,
    .results-container .stMarkdown {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        margin: 0 auto !important;
    }

    /* Center image content specifically */
    .results-container .stImage img {
        display: block !important;
        margin: 0 auto !important;
    }

    /* Center video content specifically */
    .results-container .stVideo video {
        display: block !important;
        margin: 0 auto !important;
    }

    /* Center text content specifically */
    .results-container .stMarkdown {
        text-align: center !important;
    }

    /* Ensure columns are centered */
    .results-container .stColumn {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
    }
            
    
    /* NEW: Fix for radio button container to remove white box */
    div[data-testid="stVerticalBlock"] > div > div[data-testid="stVerticalBlock"] > div[data-testid="element-container"]:has(div[data-testid="stRadio"]) {
        background-color: transparent !important;
        padding: 0 !important;
        margin: 0 !important;
        border: none !important;
        box-shadow: none !important;
    }

    /* Ensure the radio button group has no background */
    div[role="radiogroup"] {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
                /* Force all button text to white */
    button, button span, .stRadio > div > label, .stRadio > div > label span {
        color: white !important;
    }
    
    /* Also target any divs or spans inside buttons or labels */
    button * {
        color: white !important;
    }
    .stRadio > div > label * {
        color: white !important;
    }


    /* Corrected: All buttons and labels inside buttons to have white text */
    button {
        color: white !important;
    }
        /* Add this rule to specifically target button text and ensure it's white */
    .stButton > button span,
    .category-button span,
    .content-type-button span,
    .back-button-style span,
    .upload-button span,
    .search-button span,
    .st-eb span,
    .st-ec span,
    .st-ed span,
    .stRadio > div > label span,
    div[role="radiogroup"] label span,
    button span {
        color: white !important;
    }

    /* Riddle Box Styling */
    .riddle-box {
        width: 92%;
        margin: 14px auto 18px auto;
        padding: 36px 28px;
        border: 3px solid #e6b800;
        background: #fff3b0;
        border-radius: 18px;
        font-size: 30px;
        line-height: 1.6;
        text-align: center;
        box-shadow: 4px 4px 14px rgba(0,0,0,0.12);
        color: #5D4037;
    }

    /* Answer box */
    .answer-box {
        width: 78%;
        margin: 8px auto 18px auto;
        padding: 16px 18px;
        border: 2px dashed #2f6b2f;
        background: #eef7ee;
        border-radius: 12px;
        font-size: 24px;
        text-align: center;
        font-weight: 700;
        color: #143d14;
    }  
</style>
""", unsafe_allow_html=True)

# --- Initialize session state ---
if 'current_category' not in st.session_state:
    st.session_state.current_category = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'cultural_data' not in st.session_state:
    st.session_state.cultural_data = None
if 'show_upload' not in st.session_state:
    st.session_state.show_upload = False
if 'content_page' not in st.session_state:
    st.session_state.content_page = False
if 'selected_content_type' not in st.session_state:
    st.session_state.selected_content_type = None
if 'search_query' not in st.session_state:
    st.session_state.search_query = ""
if 'show_results' not in st.session_state:
    st.session_state.show_results = False
if 'text_model' not in st.session_state:
    st.session_state.text_model = None
if 'text_vectorizer' not in st.session_state:
    st.session_state.text_vectorizer = None
if 'show_riddles' not in st.session_state:
    st.session_state.show_riddles = False
if 'riddle_index' not in st.session_state:
    st.session_state.riddle_index = 0
if 'show_riddle_answer' not in st.session_state:
    st.session_state.show_riddle_answer = False
if 'use_swecha_api' not in st.session_state:
    st.session_state.use_swecha_api = True
if 'show_api_status' not in st.session_state:
    st.session_state.show_api_status = False

# --- Riddles Data ---
riddles = [
    {"question": "ఎప్పుడూ మాటాడదు, చూపినప్పుడే అర్థమవుతుంది.", "answer": "అద్దం"},
    {"question": "చెవులు లేవు కానీ వింటుంది; నోరు లేనప్పటికీ మాట్లాడుతుంది.", "answer": "ఫోన్"},
    {"question": "నడవదు కానీ ఊరంతా చూపిస్తుంది.", "answer": "పటం (మ్యాప్)"},
    {"question": "తల ఉంది, తోక ఉంది; శరీరం లేదు.", "answer": "నాణెం"},
    {"question": "కళ్లున్నా కనబడదు; రెక్కలుండి ఎగరదు.", "answer": "సూది"},
    {"question": "నడవదు, తిరుగుతుంది; కాలాన్ని చూపిస్తుంది.", "answer": "గడియారం సూది"},
    {"question": "ఎంత తీస్తే అంత పెరుగుతుంది.", "answer": "రంధ్రం"},
    {"question": "నీటిలో పుడుతుంది, నీటిలోనే చస్తుంది.", "answer": "అల (తరంగం)"},
    {"question": "గుండె లేదు కానీ కొట్టుకుంటుంది.", "answer": "గడియారం"},
    {"question": "నువ్వు తీస్తే నాకు పెరుగుతుంది.", "answer": "అప్పు / గుంత"},
    {"question": "ఎప్పుడూ వస్తూనే ఉంటుంది, ఆగదు.", "answer": "సమయం"},
    {"question": "రెక్కలు లేవు, ఎగురుతుంది; కళ్ళు లేవు, ఏడుస్తుంది.", "answer": "మేఘం"},
    {"question": "తినకపోతే చస్తుంది, తింటే బతుకుతుంది.", "answer": "అగ్ని"},
    {"question": "మూసినప్పుడు మౌనం; తెరిచినప్పుడు పాటలు.", "answer": "రేడియో"},
    {"question": "నన్ను కొడితేనే నేను పాడుతాను.", "answer": "డ్రమ్/డోలు"},
    {"question": "పుట్టుకతో నలుపు, కాలంతో తెలుపవుతుంది.", "answer": "బొగ్గు—బూడిద"},
    {"question": "మాట్లాడదు కానీ అర్థమవుతుంది.", "answer": "కళ/కళ్ళు"},
    {"question": "ఇల్లు లేదు కానీ అందరినీ కప్పేస్తుంది.", "answer": "ఆకాశం"},
    {"question": "ఉండదు కానీ అనుభవిస్తాం.", "answer": "గాలి"},
    {"question": "నిద్రపోదు, అయినా కాపాడుతుంది.", "answer": "కళ్ళు"},
    {"question": "పదాలు చెప్తుంది కానీ నోరు లేదు.", "answer": "పుస్తకం"},
    {"question": "వస్తుంది కానీ తలుపు తట్టదు.", "answer": "గాలి"},
    {"question": "రాత్రివేళ పుడుతుంది, పగలు చచ్చిపోతుంది.", "answer": "నక్షత్రాలు/చీకటి"},
    {"question": "పెద్దది కానీ బరువు లేదు.", "answer": "నీడ"},
    {"question": "అదే ఇంట్లో పుడుతుంది, అదే ఇంట్లో చస్తుంది.", "answer": "కొవ్వొత్తి"},
    {"question": "పరుగులు తీస్తుంది కానీ కాళ్లు లేవు.", "answer": "నీరు"},
    {"question": "ఎంత తాగినా దప్పిక తీర్చదు.", "answer": "సముద్రం"},
    {"question": "ఆడితే మ్రోగుతుంది, నోరు లేదు.", "answer": "డోలు"},
    {"question": "రాత్రి వస్తుంది, పగలు పోతుంది.", "answer": "చీకటి"},
    {"question": "వెళ్లదు కానీ ఊరంతా తిరుగుతుంది.", "answer": "వార్త"},
    {"question": "ఉన్నది చిన్నది, చూపేది పెద్దది.", "answer": "కళ్లజోడు"},
    {"question": "నడవదు కానీ పాదముంటుంది.", "answer": "మంచం పాదం"},
    {"question": "ఉదయం పుడుతుంది, రాత్రి చస్తుంది.", "answer": "సూర్యకాంతి"},
    {"question": "నపడదు కానీ దెబ్బకొడుతుంది.", "answer": "గాలి"},
    {"question": "మూసి ఉంచితే రహస్యం; తెరిస్తే జ్ఞానం.", "answer": "పుస్తకం"},
    {"question": "దాన్ని కోసినా రక్తం రాదు.", "answer": "చెట్టు"},
    {"question": "ఎంత ఇస్తే అంత పెరుగుతుంది.", "answer": "ప్రేమ"},
    {"question": "దాన్ని పంచితే తగ్గదు, పెరుగుతుంది.", "answer": "జ్ఞానం"},
    {"question": "చుట్టూ ఉంటుంది కానీ కనబడదు.", "answer": "గాలి"},
    {"question": "ఎవరి ఇంటికైనా ఉంటుంది.", "answer": "తలుపు"},
    {"question": "వంగితే పెరుగుతుంది.", "answer": "జ్ఞానం/వినయం"},
    {"question": "పెరిగేది వయసు, తగ్గదు.", "answer": "వయస్సు"},
    {"question": "దాన్ని పట్టుకోలేరు కానీ అనుభవిస్తాం.", "answer": "సమయం"},
    {"question": "పుడితే తల, చనిపోతే తోక.", "answer": "దీపం"},
    {"question": "తినకపోయినా కరిగిపోతుంది.", "answer": "కొవ్వొత్తి"},
    {"question": "ఒకసారి వస్తే వెనక్కి పోదు.", "answer": "కాలం"},
    {"question": "మ్రోగుతుంటుంది కానీ నోరు లేదు.", "answer": "గంట"},
    {"question": "దాన్ని విరిస్తే వాసన వస్తుంది.", "answer": "మల్లె పువ్వు"},
    {"question": "దాన్ని తాగితే ప్రमాదం ఉండొచ్చు కానీ లోపల ప్రాణం.", "answer": "నీరు"},
    {"question": "నడవదు కానీ వాహనం కదిలిస్తుంది.", "answer": "చక్రం"},
    {"question": "చాలా తిన్నా ఎప్పుడూ ఆకలే ఉంటుంది.", "answer": "అగ్ని"},
    {"question": "ఎవరు ఇచ్చినా తీసుకుంటుంది.", "answer": "భూమి (ఉపమా)"},
    {"question": "దాన్ని కడిగితే మరింత మురికి అవుతుంది.", "answer": "నీరు (మురికి నీరు)"},
    {"question": "ఉన్నా కనబడదు; లేకుంటే జీవం ఉండదు.", "answer": "గాలి"},
    {"question": "మంచం మీద పడుకుంటుంది, కానీ నిద్రపోదు.", "answer": "తలగడ/పత్రం"},
    {"question": "కన్నీళ్లు వస్తాయి కానీ కళ్ళు లేవు.", "answer": "మేఘాలు/వర్షం"},
    {"question": "ఆకాశం ప్రతిరోజూ రంగు మార్చుకుంటుంది.", "answer": "ఉదయం/సాయంత్ర ఆకాశం"},
    {"question": "అదే చోటు నుంచే ప్రపంచం చూపిస్తుంది.", "answer": "టీవీ"},
    {"question": "నడవదు కానీ ప్రయాణం చేస్తుంది.", "answer": "బోటు"},
    {"question": "వర్షం రాకపోయినా కురుస్తుంది.", "answer": "కన్నీళ్లు"},
    {"question": "తినకపోయినా తింటుంది.", "answer": "తుపాకీ గుళ్లు (మాటల ఆట)"},
    {"question": "మంట లేకపోయినా కాలుతుంది.", "answer": "కారం/మనసు"},
    {"question": "తనకి ఎముకలు లేవు కానీ నిలబెడుతుంది.", "answer": "చెట్టు కాండం"},
    {"question": "ఒక నది లాంటిది కానీ నీరు లేదు.", "answer": "రోడ్డు"},
    {"question": "పాలు ఇస్తుంది కానీ జంతువు కాదు.", "answer": "కొబ్బరికాయ"},
    {"question": "ఎప్పుడూ వెనకే ఉంటుంది కానీ తాకలేం.", "answer": "నీడ"},
    {"question": "ముక్కు ఉంది కాని వాసన చూడదు.", "answer": "రైలు"},
    {"question": "ఒక ఇల్లు ఉంది, తలుపులే లేవు.", "answer": "గుడ్డు"},
    {"question": "ఎగరదు కానీ పైకే పోతుంది.", "answer": "పొగ"},
    {"question": "వాన వస్తే పుడుతుంది, వాన పోయితే పోతుంది.", "answer": "మట్టి వాసన"},
    {"question": "చిన్నదే కానీ పెద్దది చూపిస్తుంది.", "answer": "దర్పణం/లెన్స్"},
    {"question": "నడవదు కానీ దూరం చూపిస్తుంది.", "answer": "మ్యాప్"},
    {"question": "ఎప్పుడూ పెరుగుతుంది, తగ్గదు.", "answer": "వయస్సు"},
    {"question": "చెట్టు కాదు కానీ నీడ ఇస్తుంది.", "answer": "ఇల్లు/షెడ్"},
    {"question": "కర్రలాంటిది కానీ వెలుతురు ఇస్తుంది.", "answer": "మెవ్వత్తి"},
    {"question": "పరుగెడుతుంది కానీ ఊపిరి పీల్చదు.", "answer": "రైలు"},
    {"question": "ఒకటి పుడితే మరొకటి చస్తుంది.", "answer": "పగలు - రాత్రి"},
    {"question": "వెలుతురు లేకపోతేనే ఉంటుంది.", "answer": "నీడ"},
    {"question": "గుండ్రంగా తిరుగుతుంది కానీ ఎక్కడికీ పోదు.", "answer": "గడియారం సూది"},
    {"question": "నోరు లేనిది కానీ మాటలాడుతుంది.", "answer": "రేడియో/टేప్"},
    {"question": "మాట్లాడదు కానీ తిరిగి చెబుతుంది.", "answer": "ప్రతిధ్వని"},
    {"question": "తల ఉంది కానీ జుట్టు లేదు.", "answer": "గుడ్డు/బంగాళాదుంప"},
    {"question": "నడవదు కానీ పాడుతుంది.", "answer": "టేప్ రికార్డర్"},
    {"question": "పలుకులు చెప్తుంది కానీ జీవం లేదు.", "answer": "పుస్తకం"},
    {"question": "అన్నం తినదు కానీ వండిస్తుంది.", "answer": "స్టౌవ్"},
    {"question": "తినలేం కానీ తినిపిస్తుంది.", "answer": "డబ్బు"},
    {"question": "రెక్కలు ఉన్నాయి కానీ ఎగరదు.", "answer": "తలుపు"},
    {"question": "ఆగదు కానీ నడుస్తూనే ఉంటుంది.", "answer": "కాలం"},
    {"question": "వస్తుంది కానీ దుస్తులు వేసుకోదు.", "answer": "గాలి"},
    {"question": "ఇంట్లో ఉన్నా బయట కనిపిస్తుంది.", "answer": "కిటికీ"},
    {"question": "చిన్నదే కానీ బలముంది.", "answer": "సూది"},
    {"question": "పుట్టింది నల్లగా, చనిపోయింది తెల్లగా.", "answer": "బొగ్గు - బూడిద"},
    {"question": "నీరు తాగుతుంది కానీ దాహం తీరదు.", "answer": "భూమి"},
    {"question": "చనిపోయినా नిలిచి ఉంటుంది.", "answer": "చెట్టు"},
    {"question": "ఎంత కొట్టినా ఏడవదు.", "answer": "డ్రమ్"},
    {"question": "నది దాటుతుంది, తడవదు.", "answer": "వంతెన నీడ"},
    {"question": "నన్ను పగలగొడతారు, త్రాగుతారు.", "answer": "కొబ్బరి"},
    {"question": "గుడ్డతో కప్పితేనే చూస్తారు.", "answer": "సినిమా తెర"},
    {"question": "పలకలు ఉన్నా పడుకోదు.", "answer": "పుస్తకం"},
    {"question": "కొడితేనే పాట పాడుతుంది.", "answer": "మృదంగం/డప్పు"},
    {"question": "దానికి నీళ్లు ఇస్తే చిన్నదవుతుంది.", "answer": "ఉప్పు/చక్కెర"},
    {"question": "చిన్న మొన పెద్ద పని.", "answer": "పెన్/పెన్సిల్"},
    {"question": "పోయేకొద్దీ పొలుస్తుంది.", "answer": "పెన్సిల్"},
    {"question": "కప్పుకుంటే చప్పుడు చేస్తుంది.", "answer": "కాగితం"},
    {"question": "నీళ్లకింద పుడుతుంది; నేలపై చనిపోతుంది.", "answer": "బుడగ"},
    {"question": "ఎక్కడికి వెళ్లినా పక్కనే ఉంటుంది.", "answer": "నీడ"},
    {"question": "నువ్వు పాట పాడితే ఇది నాట్యం చేస్తుంది.", "answer": "నీడ"},
    {"question": "పొట్టలో పాలు, వెలుపల దుస్తులు.", "answer": "కొబ్బరికాయ"},
    {"question": "వాన పడితేనే వస్తుంది, తడవదు.", "answer": "ఇంద్రధనుస్సు"},
    {"question": "మామిడి చెట్టుపై కాకి కూస్తే ఏం పడుతుంది?", "answer": "కాకి నీడ"},
    {"question": "ఎక్కించాకే దిగుతుంది.", "answer": "లిఫ్ట్"},
    {"question": "తలంటీ లేకపోయినా ధరించేవారు.", "answer": "ఉంగరం"},
    {"question": "ఎండలో చచ్చిపోతుంది; నీటిలో బతుకుతుంది.", "answer": "చేప"},
    {"question": "ఎప్పుడూ పడుకుని మాట్లాడుతుంది.", "answer": "పుస్తకం"},
    {"question": "ఎక్కడ పెట్టినా సరిపోయే ఇల్లు.", "answer": "తాబేలు కవచం"},
    {"question": "తల లేదు; అయినా కిరీటం ఉంటుంది.", "answer": "అనాసపండు"},
    {"question": "వాడితే కుదుస్తుంది; కుదిస్తే వాడుతుంది.", "answer": "బట్ట"},
    {"question": "గాలి లేక కదలదు; గాలి ఎక్కువైతే ఎగురుతుంది.", "answer": "గాలిపటం"},
    {"question": "ఒక్క అడుగుతో ప్రపంచం చుట్టేస్తుంది.", "answer": "వార్త/ఇంటర్నెట్"},
]

# --- Data Loading and Model Functions ---
@st.cache_resource
def load_cultural_data():
    return load_data_from_folders('data')

@st.cache_resource
def load_image_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CulturalClassifier(num_classes=4)
    if os.path.exists('models/cultural_classifier.pth'):
        model.load_state_dict(torch.load('models/cultural_classifier.pth', map_location=device))
        model.eval()
        return model
    else:
        return None

@st.cache_resource
def load_text_assets():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = 'models/text_classifier.pth'
    vectorizer_path = 'models/text_vectorizer.pkl'

    text_model = None
    text_vectorizer = None

    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        try:
            # Load vectorizer first to get input dimension
            text_vectorizer = joblib.load(vectorizer_path)
            input_dim = len(text_vectorizer.vocabulary_)

            # Load model
            text_model = TextClassifier(input_dim, hidden_dim=128, num_classes=4)
            text_model.load_state_dict(torch.load(model_path, map_location=device))
            text_model.eval()

        except Exception as e:
            st.error(f"టెక్స్ట్ మోడల్ లేదా వెక్టరైజర్‌ను లోడ్ చేయడంలో లోపం: {e}")
            text_model = None
            text_vectorizer = None

    return text_model, text_vectorizer

# Function to classify content
def classify_content(image):
    model = load_image_model()
    if model is None:
        return "మోడల్ అందుబాటులో లేదు"

    # Preprocess image
    processed_image = preprocess_image(image)

    # Convert to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_tensor = transform(image).unsqueeze(0)

    # Predict
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        outputs = model(image_tensor.to(device))
        _, predicted = torch.max(outputs, 1)

    # Map prediction to category
    category_map = {0: 'స్మారకాలు', 1: 'సంస్కృతి', 2: 'సంప్రదాయాలు', 3: 'జానపద కథలు'}
    return category_map[predicted.item()]

# --- Search Functions ---
def get_search_results(category, content_type, search_query):
    """
    This function interfaces with both local data and Swecha API to get relevant results
    based on the search query.
    """
    results = []
    
    # First, try to get results from local data
    if st.session_state.cultural_data and category in st.session_state.cultural_data:
        category_data = st.session_state.cultural_data[category]
        
        if content_type in category_data:
            # Simple text match for direct results
            local_results = [
                item for item in category_data[content_type]
                if search_query.lower() in item['name'].lower() or
                   (content_type == 'texts' and search_query.lower() in item.get('content', '').lower())
            ]
            results.extend(local_results)
            
            # If no results found with simple matching and a query exists, try semantic search
            if not local_results and search_query:
                semantic_results = get_semantic_search_results(category, content_type, search_query)
                results.extend(semantic_results)
    
    # If we have a search query and API is enabled, also try to get results from Swecha API
    if search_query and search_query.strip() and st.session_state.use_swecha_api:
        try:
            # Check if Swecha API is accessible
            if swecha_client.health_check():
                swecha_results = get_swecha_search_results(
                    query=search_query,
                    category=category,
                    content_type=content_type,
                    limit=10
                )
                
                # Add source indicator to Swecha results
                for result in swecha_results:
                    result['source'] = 'swecha_api'
                
                results.extend(swecha_results)
                
                # Show info about API results
                if swecha_results:
                    st.info(f"Found {len(swecha_results)} additional results from Swecha API")
            else:
                st.warning("Swecha API is currently unavailable. Showing only local results.")
        except Exception as e:
            st.warning(f"Could not fetch results from Swecha API: {str(e)}")
    
    return results

def get_semantic_search_results(category, content_type, search_query):
    """
    This function performs a semantic search for text content.
    """
    if content_type == 'texts':
        if st.session_state.text_model is None or st.session_state.text_vectorizer is None:
            return []

        text_data = st.session_state.cultural_data[category].get('texts', [])
        if not text_data:
            return []

        # Extract all text content and file data
        all_text_content = [item['content'] for item in text_data]

        # Transform the search query into a vector
        try:
            query_vector = st.session_state.text_vectorizer.transform([search_query])
        except ValueError as e:
            return []

        # Get vectors for all documents
        document_vectors = st.session_state.text_vectorizer.transform(all_text_content)

        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, document_vectors).flatten()

        # Get the indices of the most similar documents (top 5)
        top_indices = similarities.argsort()[-5:][::-1]

        # Filter for results with a similarity score > 0 (or a small threshold)
        top_results = [
            text_data[i]
            for i in top_indices
            if similarities[i] > 0
        ]

        return top_results
    else:
        return []

# --- Display Functions ---
def display_content_selection_page():
    # Map English categories to Telugu
    category_names = {
        "monuments": "స్మారకాలు",
        "culture": "సంస్కృతి",
        "traditions": "సంప్రదాయాలు",
        "folktales": "జానపద కథలు"
    }

    # Set background image based on category
    bg_images = {
        "monuments": r"C:\Users\DELL\OneDrive\Desktop\streamlit app\background\monuments_image.jpg",
        "culture": r"C:\Users\DELL\OneDrive\Desktop\streamlit app\background\culture_image.jpg",
        "traditions": r"C:\Users\DELL\OneDrive\Desktop\streamlit app\background\traditions_image.jpg",
        "folktales": r"C:\Users\DELL\OneDrive\Desktop\streamlit app\background\folktales_image.jpg"
    }
    
    if st.session_state.current_category in bg_images:
        set_background_image(bg_images[st.session_state.current_category])

    # Page header
    st.markdown(f'<div class="page-header">{category_names[st.session_state.current_category]}</div>', unsafe_allow_html=True)

    # Content type selection
    st.markdown('<div class="content-display">', unsafe_allow_html=True)

    # Use radio buttons for content type selection
    st.markdown('<h3 style="text-align: center; color: #5D4037; margin-bottom: 10px; text-shadow: 1px 1px 4px rgba(0,0,0,0.7);">కంటెంట్ రకాన్ని ఎంచుకోండి</h3>', unsafe_allow_html=True)
    st.session_state.selected_content_type = st.radio(
        "",
        ["చిత్రాలు", "వీడియోలు", "టెక్స్ట్ ఫైల్స్"],
        key="content_type_radio",
        horizontal=True,
        label_visibility="collapsed"
    )
    # Get available content names for dropdown if culture or folktales
    content_options = [""]
    if st.session_state.current_category in ["culture", "folktales"] and st.session_state.cultural_data:
        content_type_en = "images" if st.session_state.selected_content_type == "చిత్రాలు" else \
                        "videos" if st.session_state.selected_content_type == "వీడియోలు" else "texts"
        
        if st.session_state.current_category in st.session_state.cultural_data:
            category_data = st.session_state.cultural_data[st.session_state.current_category]
            if content_type_en in category_data:
                # Remove file extensions from the names for display
                content_options.extend([os.path.splitext(item['name'])[0] for item in category_data[content_type_en]])

    # Search bar with dropdown for culture and folktales
    st.markdown('<div class="search-container">', unsafe_allow_html=True)
    
    if st.session_state.current_category in ["culture", "folktales"]:
        # Use dropdown for culture and folktales
        search_query = st.selectbox(
            "వెతకండి:",
            options=content_options,
            key="search_dropdown",
            format_func=lambda x: os.path.splitext(x)[0] if x else "మీరు కోరుకునేది ఇక్కడ ఎంచుకోండి ...",
        index=0
        )
    else:
        # Use text input for other categories
        search_query = st.text_input(
            "వెతకండి:",
            value=st.session_state.search_query,
            placeholder="మీరు కోరుకునేది ఇక్కడ టైప్ చేయండి...",
            key="search_input"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)

    # UPDATED: Button row with smaller buttons
    st.markdown('<div class="button-row">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("శోధన", use_container_width=True, key="search_button"):
            st.session_state.search_query = search_query
            st.session_state.show_results = True
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # UPDATED: Back button - smaller
    st.markdown('<div class="centered-button">', unsafe_allow_html=True)
    if st.button("← వెనక్కి", key="back_button", use_container_width=True):
        st.session_state.content_page = False
        st.session_state.selected_content_type = None
        st.session_state.search_query = ""
        st.session_state.show_results = False
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

def display_search_results_page():
    # Map Telugu to English for internal logic
    telugu_to_english = {
        "చిత్రాలు": "images",
        "వీడియోలు": "videos",
        "టెక్స్ట్ ఫైల్స్": "texts"
    }

    # Map English categories to Telugu
    category_names = {
        "monuments": "స్మారకాలు",
        "culture": "సంస్కృతి",
        "traditions": "సంప్రదాయాలు",
        "folktales": "జానపద కథలు"
    }

    content_type_names = {
        "images": "చిత్రాలు",
        "videos": "వీడియోలు",
        "texts": "టెక్స్ట్ ఫైల్స్"
    }

    # Set background image based on category
    bg_images = {
        "monuments": r"C:\Users\DELL\OneDrive\Desktop\streamlit app\background\monuments_image.jpg",
        "culture": r"C:\Users\DELL\OneDrive\Desktop\streamlit app\background\culture_image.jpg",
        "traditions": r"C:\Users\DELL\OneDrive\Desktop\streamlit app\background\traditions_image.jpg",
        "folktales": r"C:\Users\DELL\OneDrive\Desktop\streamlit app\background\folktales_image.jpg"
    }
    
    if st.session_state.current_category in bg_images:
        set_background_image(bg_images[st.session_state.current_category])

    # Retrieve results
    content_type_en = telugu_to_english[st.session_state.selected_content_type]
    search_results = get_search_results(st.session_state.current_category, content_type_en, st.session_state.search_query)

    # Page header
    st.markdown(f'<div class="page-header">{category_names[st.session_state.current_category]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">{content_type_names[content_type_en]} - శోధన: "{st.session_state.search_query}"</div>', unsafe_allow_html=True)

    # Display results
    if search_results:
        # Display images in a centered layout
        if content_type_en == 'images':
            for img_data in search_results:
                try:
                    # Create a centered container for each image
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.image(
                            img_data['path'],
                            use_column_width=True
                        )
                        # Show source indicator
                        if img_data.get('source') == 'swecha_api':
                            st.info("🌐 Swecha API నుండి")
                        else:
                            st.info("💾 లోకల్ డేటా నుండి")
                except Exception as e:
                    st.error(f"చిత్రాన్ని ప్రదర్శించడంలో లోపం: {e}")

        # Display texts
        elif content_type_en == 'texts':
            for text_data in search_results:
                # Center text content
                col1, col2, col3 = st.columns([1, 3, 1])
                with col2:
                    st.write(text_data['content'])
                    # Show source indicator
                    if text_data.get('source') == 'swecha_api':
                        st.info("🌐 Swecha API నుండి")
                    else:
                        st.info("💾 లోకల్ డేటా నుండి")

        # Display videos
        elif content_type_en == 'videos':
            for video_data in search_results:
                # Center video content
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.video(video_data['path'])
                    # Show source indicator
                    if video_data.get('source') == 'swecha_api':
                        st.info("🌐 Swecha API నుండి")
                    else:
                        st.info("💾 లోకల్ డేటా నుండి")
    else:
        st.warning(f"'{st.session_state.search_query}' కోసం {category_names[st.session_state.current_category]}లో {content_type_names[content_type_en]} కనుగొనబడలేదు.")

    # Single back button at the bottom
    st.markdown('<div class="centered-button">', unsafe_allow_html=True)
    if st.button("← వెనక్కి", key="back_from_results", use_container_width=True):
        st.session_state.show_results = False
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

def display_riddles_page():
    # Set background for riddles page
    set_background_image(r"C:\Users\DELL\OneDrive\Desktop\streamlit app\background\background_image.jpg")
    
    # Page header
    st.markdown('<div class="page-header">తెలుగు పొడుపు కథలు</div>', unsafe_allow_html=True)
    
    # Get current riddle
    current = riddles[st.session_state.riddle_index]
    
    # Display riddle
    st.markdown(f'<div class="riddle-box">{current["question"]}</div>', unsafe_allow_html=True)
    
    # Display answer if revealed
    if st.session_state.show_riddle_answer:
        st.markdown(f'<div class="answer-box">సమాధానం: {current["answer"]}</div>', unsafe_allow_html=True)
    
    # Navigation buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("మునుపటి ప్రశ్న", key="prev_btn", use_container_width=True):
            if st.session_state.riddle_index > 0:
                st.session_state.riddle_index -= 1
            st.session_state.show_riddle_answer = False
            st.rerun()
    with col2:
        if st.button("సమాధానం చూపించు", key="ans_btn", use_container_width=True):
            st.session_state.show_riddle_answer = True
            st.rerun()
    with col3:
        if st.button("తరువాతి ప్రశ్న", key="next_btn", use_container_width=True):
            if st.session_state.riddle_index < len(riddles) - 1:
                st.session_state.riddle_index += 1
            st.session_state.show_riddle_answer = False
            st.rerun()
    
    # Back button
    st.markdown('<div class="centered-button">', unsafe_allow_html=True)
    if st.button("← వెనక్కి", key="back_from_riddles", use_container_width=True):
        st.session_state.show_riddles = False
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

def display_api_status():
    """Display Swecha API status and configuration"""
    # Set background for API status page
    set_background_image(r"C:\Users\DELL\OneDrive\Desktop\streamlit app\background\background_image.jpg")
    
    # Page header
    st.markdown('<div class="page-header">Swecha API స్థితి</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="content-display">', unsafe_allow_html=True)
    
    # API Status
    st.markdown('<h3 style="text-align: center; color: #5D4037; margin-bottom: 20px;">API స్థితి</h3>', unsafe_allow_html=True)
    
    # Check API health
    with st.spinner("API స్థితిని తనిఖీ చేస్తున్నాము..."):
        api_healthy = swecha_client.health_check()
    
    if api_healthy:
        st.success("✅ Swecha API అందుబాటులో ఉంది")
        
        # Get API statistics
        try:
            stats = swecha_client.get_statistics()
            if stats:
                st.info("📊 API గణాంకాలు:")
                st.json(stats)
        except:
            st.info("📊 API గణాంకాలు అందుబాటులో లేవు")
    else:
        st.error("❌ Swecha API అందుబాటులో లేదు")
        st.warning("దయచేసి మీ ఇంటర్నెట్ కనెక్షన్ మరియు API URL ని తనిఖీ చేయండి")
    
    # API Configuration
    st.markdown('<h3 style="text-align: center; color: #5D4037; margin-bottom: 20px;">API కాన్ఫిగరేషన్</h3>', unsafe_allow_html=True)
    
    # API URL
    st.text_input("API URL", value=swecha_client.base_url, disabled=True)
    
    # Toggle API usage
    st.session_state.use_swecha_api = st.checkbox(
        "Swecha API ని ఉపయోగించండి",
        value=st.session_state.use_swecha_api,
        help="ఈ ఎంపికను ఆఫ్ చేస్తే, మీరు లోకల్ డేటాను మాత్రమే ఉపయోగిస్తారు"
    )
    
    # Test API endpoints
    st.markdown('<h3 style="text-align: center; color: #5D4037; margin-bottom: 20px;">API టెస్టింగ్</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 శోధన టెస్ట్", use_container_width=True):
            test_results = get_swecha_search_results("telugu", limit=5)
            if test_results:
                st.success(f"శోధన టెస్ట్ విజయవంతం! {len(test_results)} ఫలితాలు కనుగొనబడ్డాయి")
                st.json(test_results[:2])  # Show first 2 results
            else:
                st.warning("శోధన టెస్ట్ విఫలం")
    
    with col2:
        if st.button("📤 అప్‌లోడ్ టెస్ట్", use_container_width=True):
            # Create a test file
            test_file_path = "test_upload.txt"
            with open(test_file_path, "w", encoding="utf-8") as f:
                f.write("ఇది ఒక పరీక్ష ఫైల్")
            
            try:
                success = upload_to_swecha(
                    test_file_path, 
                    "culture", 
                    "texts", 
                    {"test": True, "description": "Test upload"}
                )
                if success:
                    st.success("అప్‌లోడ్ టెస్ట్ విజయవంతం!")
                else:
                    st.warning("అప్‌లోడ్ టెస్ట్ విఫలం")
                
                # Clean up test file
                if os.path.exists(test_file_path):
                    os.remove(test_file_path)
            except Exception as e:
                st.error(f"అప్‌లోడ్ టెస్ట్ లోపం: {str(e)}")
    
    # Back button
    st.markdown('<div class="centered-button">', unsafe_allow_html=True)
    if st.button("← వెనక్కి", key="back_from_api_status", use_container_width=True):
        st.session_state.show_api_status = False
        st.rerun()
    st.markdown('</div>')
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_upload_section():
    # Set background for upload page
    set_background_image(r"C:\Users\DELL\OneDrive\Desktop\streamlit app\background\background_image.jpg")
    
    # Page header
    st.markdown('<div class="page-header">ఫైల్ అప్‌లోడ్ చేయండి</div>', unsafe_allow_html=True)
    
    # Category selection
    st.markdown('<div class="content-display">', unsafe_allow_html=True)
    
    # Use radio buttons for category selection
    st.markdown('<h3 style="text-align: center; color: #5D4037; margin-bottom: 10px; text-shadow: 1px 1px 4px rgba(0,0,0,0.7);">వర్గాన్ని ఎంచుకోండి</h3>', unsafe_allow_html=True)
    upload_category = st.radio(
        "",
        ["స్మారకాలు", "సంస్కృతి", "సంప్రదాయాలు", "జానపద కథలు"],
        key="upload_category",
        horizontal=True,
        label_visibility="collapsed"
    )
    
    # Map Telugu category names to English for internal logic
    category_map = {
        "స్మారకాలు": "monuments",
        "సంస్కృతి": "culture",
        "సంప్రదాయాలు": "traditions",
        "జానపద కథలు": "folktales"
    }
    
    # Content type selection based on category
    st.markdown('<h3 style="text-align: center; color: #5D4037; margin-bottom: 10px; text-shadow: 1px 1px 4px rgba(0,0,0,0.7);">కంటెంట్ రకాన్ని ఎంచుకోండి</h3>', unsafe_allow_html=True)
    upload_content_type = st.radio(
        "",
        ["చిత్రాలు", "వీడియోలు", "టెక్స్ట్ ఫైల్స్"],
        key="upload_content_type",
        horizontal=True,
        label_visibility="collapsed"
    )
    
    # File uploader based on selected content type
    file_types = {
        "చిత్రాలు": ['jpg', 'jpeg', 'png'],
        "వీడియోలు": ['mp4', 'mov', 'avi'],
        "టెక్స్ట్ ఫైల్స్": ['txt']
    }
    
    # Center the uploader text
    st.markdown("""
    <style>
    .upload-label {
        text-align: center;
        display: block;
        font-size: 1.3rem;
        font-weight: 700;
        color: #5D4037;
        margin-bottom: 8px;
    }
    .upload-info {
        text-align: center;
        display: block;
        font-size: 1.1rem;
        color: #5D4037;
        margin: 10px 0;
        padding: 10px;
        background-color: rgba(255, 255, 255, 0.7);
        border-radius: 8px;
    }
    .centered-content {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        width: 100%;
    }
    .centered-upload {
        display: flex;
        justify-content: center;
        width: 100%;
    }
    .success-message {
        text-align: center;
        color: green;
        font-weight: bold;
        margin: 15px 0;
        padding: 10px;
        background-color: rgba(0, 255, 0, 0.1);
        border-radius: 8px;
        border: 1px solid green;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="upload-label">ఫైల్‌ను ఎంచుకోండి</p>', unsafe_allow_html=True)
    
    # Create a container for the file uploader to center it
    st.markdown('<div class="centered-upload">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "",
        type=file_types[upload_content_type],
        help=f"{upload_content_type} అప్‌లోడ్ చేయండి",
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display preview of uploaded content in the center (only if file is selected)
    if uploaded_file is not None:
        st.markdown('<div class="centered-content">', unsafe_allow_html=True)
        
        if upload_content_type == "చిత్రాలు":
            try:
                image = Image.open(uploaded_file)
                st.image(image, width=300, caption="అప్‌లోడ్ చేయబడిన చిత్రం")
            except Exception as e:
                st.error(f"చిత్రాన్ని ప్రాసెస్ చేయడంలో లోపం: {e}")
        
        elif upload_content_type == "వీడియోలు":
            st.video(uploaded_file, format='video/mp4')
            st.markdown('<p style="text-align: center;">అప్‌లోడ్ చేయబడిన వీడియో</p>', unsafe_allow_html=True)
            
        elif upload_content_type == "టెక్స్ట్ ఫైల్స్":
            try:
                content = uploaded_file.getvalue().decode('utf-8')
                # Show first 200 characters as preview
                preview = content[:200] + "..." if len(content) > 200 else content
                st.text_area("ఫైల్ ప్రివ్యూ", value=preview, height=150, disabled=True)
            except Exception as e:
                st.error(f"టెక్స్ట్ ఫైల్‌ను చదవడంలో లోపం: {e}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Upload button
        st.markdown('<div class="centered-button">', unsafe_allow_html=True)
        if st.button("అప్‌లోడ్ చేయండి", key="upload_confirm_button", use_container_width=True):
            # Create directory if it doesn't exist
            category_en = category_map[upload_category]
            content_type_en = "images" if upload_content_type == "చిత్రాలు" else \
                            "videos" if upload_content_type == "వీడియోలు" else "texts"
            
            os.makedirs(f"data/{category_en}/{content_type_en}", exist_ok=True)
            
            # Save file with timestamp to avoid name conflicts
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_extension = uploaded_file.name.split('.')[-1]
            file_name = f"{timestamp}.{file_extension}"
            file_path = os.path.join(f"data/{category_en}/{content_type_en}", file_name)
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Also try to upload to Swecha API if enabled
            if st.session_state.use_swecha_api:
                try:
                    if swecha_client.health_check():
                        # Prepare metadata for Swecha API
                        metadata = {
                            "title": uploaded_file.name,
                            "description": f"Uploaded content for {upload_category} - {upload_content_type}",
                            "language": "te",  # Telugu
                            "tags": [upload_category, upload_content_type, "telugu", "cultural_heritage"]
                        }
                        
                        # Upload to Swecha API
                        swecha_success = upload_to_swecha(
                            file_path=file_path,
                            category=category_en,
                            content_type=content_type_en,
                            metadata=metadata
                        )
                        
                        if swecha_success:
                            st.success("ఫైల్ విజయవంతంగా అప్‌లోడ్ చేయబడింది (లోకల్ మరియు Swecha API)")
                        else:
                            st.warning("ఫైల్ లోకల్‌గా అప్‌లోడ్ చేయబడింది, కానీ Swecha API కి అప్‌లోడ్ చేయలేకపోయాము")
                    else:
                        st.warning("Swecha API అందుబాటులో లేదు. ఫైల్ లోకల్‌గా మాత్రమే అప్‌లోడ్ చేయబడింది.")
                except Exception as e:
                    st.warning(f"ఫైల్ లోకల్‌గా అప్‌లోడ్ చేయబడింది, కానీ Swecha API కి అప్‌లోడ్ చేయలేకపోయాము: {str(e)}")
            else:
                st.info("Swecha API నిష్క్రియం చేయబడింది. ఫైల్ లోకల్‌గా మాత్రమే అప్‌లోడ్ చేయబడింది.")
            
            # Show success message
            st.markdown('<div class="success-message">ఫైల్ విజయవంతంగా అప్‌లోడ్ చేయబడింది!</div>', unsafe_allow_html=True)
            
            # Add a back button after successful upload
            st.markdown('<div class="centered-button">', unsafe_allow_html=True)
            if st.button("← వెనక్కి", key="back_after_upload", use_container_width=True):
                st.session_state.show_upload = False
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Don't rerun immediately - let the user see the success message and back button
        else:
            # Regular back button when no upload has been done yet
            st.markdown('<div class="centered-button">', unsafe_allow_html=True)
            if st.button("← వెనక్కి", key="back_from_upload", use_container_width=True):
                st.session_state.show_upload = False
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Regular back button when no file is selected
        st.markdown('<div class="centered-button">', unsafe_allow_html=True)
        if st.button("← వెనక్కి", key="back_from_upload", use_container_width=True):
            st.session_state.show_upload = False
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

def display_main_page():
    # Set main background image
    set_background_image(r"C:\Users\DELL\OneDrive\Desktop\streamlit app\background\background_image.jpg")
    
    # Main header
    st.markdown('<h1 class="main-header">తెలుగు సాంస్కృతిక వారసత్వం</h1>', unsafe_allow_html=True)

    # Category buttons in a grid
    st.markdown('<div class="button-grid-container">', unsafe_allow_html=True)
    st.markdown('<div class="category-grid">', unsafe_allow_html=True)

    # Create 4 columns for the 4 categories
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("స్మారకాలు", key="monuments_button", use_container_width=True):
            st.session_state.current_category = "monuments"
            st.session_state.content_page = True
            st.rerun()

    with col2:
        if st.button("సంస్కృతి", key="culture_button", use_container_width=True):
            st.session_state.current_category = "culture"
            st.session_state.content_page = True
            st.rerun()

    with col3:
        if st.button("సంప్రదాయాలు", key="traditions_button", use_container_width=True):
            st.session_state.current_category = "traditions"
            st.session_state.content_page = True
            st.rerun()

    with col4:
        if st.button("జానపద కథలు", key="folktales_button", use_container_width=True):
            st.session_state.current_category = "folktales"
            st.session_state.content_page = True
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)  # Close category-grid

    # Upload, Riddles, and API Status buttons below the category buttons
    st.markdown('<div class="upload-button-container">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("ఫైల్ అప్‌లోడ్ చేయండి", key="upload_button", use_container_width=True):
            st.session_state.show_upload = True
            st.rerun()
    with col2:
        if st.button("పొడుపు కథలు", key="riddles_button", use_container_width=True):
            st.session_state.show_riddles = True
            st.rerun()
    with col3:
        if st.button("🔌 API స్థితి", key="api_status_button", use_container_width=True):
            st.session_state.show_api_status = True
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)  # Close upload-button-container

    st.markdown('</div>', unsafe_allow_html=True)  # Close button-grid-container

# --- Main App Logic ---
def main():
    # Load data and models
    if not st.session_state.data_loaded:
        with st.spinner("డేటా మరియు మోడల్స్ లోడ్ అవుతున్నాయి..."):
            st.session_state.cultural_data = load_cultural_data()
            st.session_state.text_model, st.session_state.text_vectorizer = load_text_assets()
            st.session_state.data_loaded = True

    # Display appropriate page based on session state
    if st.session_state.show_upload:
        display_upload_section()
    elif st.session_state.show_riddles:
        display_riddles_page()
    elif st.session_state.show_api_status:
        display_api_status()
    elif st.session_state.content_page:
        if st.session_state.show_results:
            display_search_results_page()
        else:
            display_content_selection_page()
    else:
        display_main_page()

if __name__ == "__main__":
    main()