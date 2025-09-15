Changelog
All notable changes to this project will be documented in this file.


[Unreleased]

Planned improvements for AI response accuracy and expanded language support
UI enhancements for better mobile and low-bandwidth experience
Additional integration options for community contributions and metadata enrichment
Enhanced error reporting and troubleshooting tools for API connectivity
Performance optimizations for faster search and loading times
Expanded riddle and interactive content library



[1.0.0] - 2025-08-31

Initial Release

Launched Streamlit-based web application titled తెలుగు సాంసృతిక వారసత్వం

Core functionalities:

Upload and management of images, videos, and texts under categories: monuments, culture, traditions, and folktales
AI-driven classification models for images (CulturalClassifier) and texts (TextClassifier)
Semantic and keyword search across local data
Riddles and interactive quizzes in Telugu to engage users
User-friendly UI with Telugu language interface and English support


Integrated Swecha Corpus API:

Search extension through external API with source indicator tags
Dual upload system from app to local and external API storage
API status monitoring with toggling option between offline and hybrid modes


Robust handling of API failures with fallback to local-only mode
Clear session management and user prompt updates
Background theming based on selected cultural category
Data storage and metadata management for uploaded content
Dashboard views for search results, uploaded content, and riddles
Documentation and setup instructions for running the app locally



[0.9.0] - YYYY-MM-DD

Beta Release

Initial prototype with basic upload and display of cultural content
Developed initial classification models for images and text
Implemented basic keyword search for local data
Added rudimentary support for video content and frame extraction
Setup basic Streamlit UI with category selection
Included a set of Telugu riddles for engagement
Configured initial version of Swecha API client for health checks



[0.8.0] - 2025-08-31

Development Milestones

Designed local data storage schema and folder structures
Built preprocessing pipelines for images, text, and video
Developed training scripts and datasets for cultural classifiers
Established vectorization and semantic search for text content
Implemented cache mechanisms for model loading and data access
Enhanced error handling during file uploads and processing

