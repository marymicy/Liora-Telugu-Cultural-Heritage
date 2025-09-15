తెలుగు సాంసృతిక వారసత్వం
A digital platform dedicated to preserving, exploring, and celebrating the rich cultural heritage of Telugu-speaking communities. This app combines local data with the Swecha Corpus API to create a vast, multilingual archive of Telugu monuments, traditions, folktales, and cultural narratives.


Project Overview
This project aims to build a comprehensive digital archive and interactive AI tool named Liora, which allows users—especially rural and semi-urban communities—to contribute, access, and engage with cultural material in Telugu. It fosters cultural pride and supports the preservation of oral and written traditions through an accessible, low-bandwidth Streamlit application.


Features


Multilingual User Interface primarily in Telugu, with support for interaction in local languages.

Content Upload: Images, videos, and texts across four categories — monuments, culture, traditions, and folktales.

Smart Search: Combines local storage and Swecha Corpus API powered search with semantic and keyword matching.

AI Classification: Automated classification of cultural content using deep learning models.

Interactive Riddles and Games in Telugu to engage users and promote cultural learning.

Swecha API Integration for extended dataset access, upload, and seamless data synchronization.

Robust Error Handling with fallback to offline mode and clear feedback messages in Telugu.



Live link
The application is hosted at:
https://huggingface.co/spaces/Sudheshnaa/telugu-cultural-heritage

Demo Video of the working website
https://drive.google.com/file/d/1jPVb9vv10VhiKKS6Lt0yCz8Q1kzRNifY/view?usp=sharing

Images of the working website
https://drive.google.com/drive/folders/1oZK-JHCd782FKb3l_mkUVgOyGQZngUMV?usp=sharing

Getting Started

Prerequisites

Python 3.8+
Access to the Git repository (or local files)
Proper folder structure with required data and model files.


Installation

Clone the repository:

git clone 
cd 
text

Install dependencies:

pip install -r requirements.txt
text


Place trained model files (cultural_classifier.pth, text_classifier.pth, text_vectorizer.pkl) under the models/ directory.


Fill the data/ folder with cultural assets categorized properly.


Configure API by editing swecha_config.py if needed.


Run the Streamlit application:


streamlit run app.py
text


Directory Structure


app.py - Main Streamlit app.

local_utils.py - Helpers for data loading, preprocessing, and ML models.

swecha_api.py - API client for Swecha integration.

swecha_config.py - API configuration settings.

models/ - Pretrained ML models and vectorizers.

data/ - Organized cultural data (images, texts, videos).

background/ - Background images for UI.

requirements.txt - Python dependencies.

data/
monuments/
culture/
traditions/
folktales/
models/
cultural_classifier.pth
text_classifier.pth
text_vectorizer.pkl
background/
monuments_image.jpg
culture_image.jpg
traditions_image.jpg
folktales_image.jpg
app.py
local_utils.py
swecha_api.py
swecha_config.py
README.md
CONTRIBUTING.md
CHANGELOG.md
LICENSE
requirements.txt


Usage

Select cultural category and upload images, videos, or text files.
Search content using keywords or semantic queries.
Enjoy interactive riddles in Telugu.
Access Swecha API functionality to broaden dataset reach.
Monitor and toggle API status in the dedicated status page.



Contribution
Contributions are welcome! Please follow community guidelines for submitting bug fixes, new features, or data improvements. Refer to the CONTRIBUTING.md for detailed instructions.


License
This project is licensed.LICENSE file for details.


Acknowledgements
Special thanks to the Swecha Corpus team and viswam.ai for API infrastructure and assistance. Together, we strive to preserve and promote Telugu cultural heritage through technology.


References

Swecha API Documentation
Project report and cultural data provided within the repository
Detailed AI and cultural content strategy document



Contact
For questions and collaboration, reach out to the team via project GitHub or designated community channels.
