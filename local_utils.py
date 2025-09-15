import os
import json
import numpy as np
from PIL import Image
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import Dataset
from sklearn.metrics.pairwise import cosine_similarity
import joblib

def load_text_content(file_path):
    """Load text content from various file types including CSV"""
    try:
        if file_path.endswith('.csv'):
            # Read CSV file
            try:
                df = pd.read_csv(file_path)
                # Try to extract text from different column types
                text_content = ""
                for col in df.columns:
                    if df[col].dtype == 'object':  # String data
                        text_content += " " + " ".join(df[col].dropna().astype(str).tolist())
                return text_content.strip()
            except Exception as e:
                print(f"Error reading CSV {file_path}: {e}")
                return ""
        else:
            # Read text file
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except UnicodeDecodeError:
                # Try different encodings
                try:
                    with open(file_path, 'r', encoding='latin-1') as f:
                        return f.read()
                except:
                    return ""
    except Exception as e:
        print(f"Error loading text file {file_path}: {e}")
        return ""

# Data loading and preprocessing functions
def load_data_from_folders(base_path):
    """
    Load all data from the organized folder structure
    Returns: Dictionary with category-wise data
    """
    categories = ['monuments', 'culture', 'traditions', 'folktales']
    data_dict = {}
    
    for category in categories:
        category_path = os.path.join(base_path, category)
        data_dict[category] = {
            'images': [],
            'videos': [],
            'texts': []
        }
        
        # Load images
        image_path = os.path.join(category_path, 'images')
        if os.path.exists(image_path):
            for img_file in os.listdir(image_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        img_path = os.path.join(image_path, img_file)
                        data_dict[category]['images'].append({
                            'path': img_path,
                            'name': img_file
                        })
                    except Exception as e:
                        print(f"Error loading image {img_file}: {e}")
        
        # Load texts (including CSV files)
        text_path = os.path.join(category_path, 'texts')
        if os.path.exists(text_path):
            for text_file in os.listdir(text_path):
                if text_file.lower().endswith(('.txt', '.csv')):
                    try:
                        file_path = os.path.join(text_path, text_file)
                        content = load_text_content(file_path)
                        
                        data_dict[category]['texts'].append({
                            'path': file_path,
                            'content': content,
                            'name': text_file
                        })
                    except Exception as e:
                        print(f"Error loading text {text_file}: {e}")
        
        # Load videos (store paths and extract metadata)
        video_path = os.path.join(category_path, 'videos')
        if os.path.exists(video_path):
            for video_file in os.listdir(video_path):
                if video_file.lower().endswith(('.mp4', '.avi', '.mov')):
                    video_path_full = os.path.join(video_path, video_file)
                    
                    # Extract video metadata
                    cap = cv2.VideoCapture(video_path_full)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = frame_count / fps if fps > 0 else 0
                    cap.release()
                    
                    data_dict[category]['videos'].append({
                        'path': video_path_full,
                        'name': video_file,
                        'fps': fps,
                        'frame_count': frame_count,
                        'duration': duration
                    })
    
    return data_dict

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model input"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image
    image = image.resize(target_size)
    
    # Convert to numpy array and normalize
    img_array = np.array(image) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def extract_frames_from_video(video_path, num_frames=5):
    """Extract frames from video for processing"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        return frames
    
    # Calculate frame indices to extract
    frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
    
    cap.release()
    return frames

# Simple CNN model for image classification
class CulturalClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(CulturalClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Text classification model
class TextClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(TextClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Video classification model (using extracted frames)
class VideoClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(VideoClassifier, self).__init__()
        # Use the same architecture as image classifier
        self.frame_classifier = CulturalClassifier(num_classes)
    
    def forward(self, x):
        # x is a batch of videos, each represented by multiple frames
        batch_size, num_frames, C, H, W = x.size()
        x = x.view(batch_size * num_frames, C, H, W)
        outputs = self.frame_classifier(x)
        outputs = outputs.view(batch_size, num_frames, -1)
        # Average the predictions across frames
        return outputs.mean(dim=1)
    
class TextDataset(Dataset):
    def __init__(self, data_dict, vectorizer=None, min_text_length=10):
        self.texts = []
        self.labels = []
        self.valid_indices = []  # Keep track of valid text samples
        
        # Map categories to numerical labels
        category_map = {'monuments': 0, 'culture': 1, 'traditions': 2, 'folktales': 3}
        
        # Prepare text data
        for category, content in data_dict.items():
            for text_data in content['texts']:
                text_content = load_text_content(text_data['path'])
                
                # Only include texts that have sufficient content
                if text_content and len(text_content.strip()) >= min_text_length:
                    self.texts.append(text_content)
                    self.labels.append(category_map[category])
                    self.valid_indices.append(True)
                else:
                    print(f"Skipping short/empty text: {text_data['name']}")
                    self.valid_indices.append(False)
        
        print(f"Found {len(self.texts)} valid text samples out of {len(self.valid_indices)} total files")
        
        # Check if we have texts from all categories
        unique_labels = set(self.labels)
        print(f"Text samples per category: { {k: self.labels.count(k) for k in unique_labels} }")
        
        # Vectorize texts if we have valid samples
        if len(self.texts) > 0:
            if vectorizer is None:
                self.vectorizer = TfidfVectorizer(
                    max_features=1000,
                    min_df=2,  # Ignore terms that appear in less than 2 documents
                    max_df=0.8  # Ignore terms that appear in more than 80% of documents
                )
                self.features = self.vectorizer.fit_transform(self.texts).toarray()
            else:
                self.vectorizer = vectorizer
                self.features = self.vectorizer.transform(self.texts).toarray()
        else:
            self.features = np.array([])
            self.vectorizer = None
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), self.labels[idx]
    
def save_uploaded_file(uploaded_file, category, file_type):
    """
    Save uploaded file to the appropriate folder structure with error handling
    """
    try:
        # Validate inputs
        if uploaded_file is None:
            return None
            
        valid_categories = ['monuments', 'culture', 'traditions', 'folktales']
        valid_types = ['image', 'text', 'video']
        
        if category not in valid_categories:
            print(f"Invalid category: {category}")
            return None
            
        if file_type not in valid_types:
            print(f"Invalid file type: {file_type}")
            return None
        
        # Create directory structure if it doesn't exist
        base_dir = 'data'
        category_dir = os.path.join(base_dir, category, file_type + 's')
        os.makedirs(category_dir, exist_ok=True)
        
        # Generate file path
        file_path = os.path.join(category_dir, uploaded_file.name)
        
        # Check if file already exists and handle naming
        counter = 1
        original_name, extension = os.path.splitext(uploaded_file.name)
        while os.path.exists(file_path):
            new_name = f"{original_name}_{counter}{extension}"
            file_path = os.path.join(category_dir, new_name)
            counter += 1
        
        # Save the file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        print(f"File saved successfully: {file_path}")
        return file_path
        
    except Exception as e:
        print(f"Error saving file {uploaded_file.name}: {e}")
        return None