# తెలుగు సాంసృతిక వారసత్వం (Liora)

A digital platform dedicated to preserving, exploring, and celebrating the rich cultural heritage of Telugu-speaking communities. This app combines local data with the Swecha Corpus API to create a vast, multilingual archive of Telugu monuments, traditions, folktales, and cultural narratives.

---

## Project Overview
Liora is a comprehensive digital archive and interactive AI tool that enables rural and semi-urban communities to contribute, access, and engage with Telugu cultural material. It fosters cultural pride and preserves oral and written traditions through an accessible, low-bandwidth Streamlit application.

---

## Features
- Multilingual User Interface in Telugu with support for local languages  
- Content Upload: Images, videos, and texts in categories (monuments, culture, traditions, folktales)  
- Smart Search with semantic and keyword matching using local data + Swecha Corpus API  
- AI Classification of cultural content using deep learning models  
- Interactive riddles and games in Telugu for cultural learning  
- Swecha API Integration for dataset expansion and data synchronization  
- Robust error handling with offline fallback and feedback messages in Telugu  

---

## Live Demo
- **Hosted Application**: [Click here](https://huggingface.co/spaces/Sudheshnaa/telugu-cultural-heritage)  
- **Demo Video**: [Google Drive Link](https://drive.google.com/file/d/1jPVb9vv10VhiKKS6Lt0yCz8Q1kzRNifY/view?usp=sharing)  
- **Screenshots**: [Project Images](https://drive.google.com/drive/folders/1oZK-JHCd782FKb3l_mkUVgOyGQZngUMV?usp=sharing)  

---

## Getting Started

### Prerequisites
- Python 3.8+  
- Git access or local files  
- Proper folder structure with data and model files  

### Installation
Clone the repository:
```bash
git clone <repository-url>
cd liora
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Place trained models under models/:

cultural_classifier.pth

text_classifier.pth

text_vectorizer.pkl

Fill the data/ folder with categorized cultural assets.
Configure the API in swecha_config.py if needed.

Run the application:

bash
Copy code
streamlit run app.py
Directory Structure
bash
Copy code
app.py               # Main Streamlit app
local_utils.py       # Helper functions
swecha_api.py        # Swecha API client
swecha_config.py     # API configuration
models/              # Pretrained ML models
data/                # Cultural datasets
background/          # Background images
requirements.txt     # Dependencies
README.md            # Project documentation
CONTRIBUTING.md      # Contribution guidelines
CHANGELOG.md         # Changelog
LICENSE              # License file
Usage
Upload cultural content (images, videos, texts)

Search by keywords or semantic queries

Play Telugu riddles for cultural engagement

Access Swecha API to expand dataset reach

Monitor API status via status page

Contribution
Contributions are welcome. Please follow the guidelines in CONTRIBUTING.md when submitting fixes, new features, or data improvements.

License
This project is licensed under the terms in the LICENSE file.

Acknowledgements
Swecha Corpus Team – API and dataset support

viswam.ai – API infrastructure and assistance

References
Swecha API Documentation

Project report and cultural data in the repository

AI and cultural content strategy document

Contact
For questions or collaborations, please reach out through the project GitHub or community channels.





