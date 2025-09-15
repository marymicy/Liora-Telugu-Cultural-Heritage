import requests
import json
import os
from typing import Dict, List, Optional, Any
import streamlit as st
from datetime import datetime
import base64
from swecha_config import (
    SWECHA_API_BASE_URL, API_ENDPOINTS, API_TIMEOUT, 
    CONTENT_CATEGORIES, CONTENT_TYPES, DEFAULT_METADATA,
    ERROR_MESSAGES, SUCCESS_MESSAGES, API_AUTH_TOKEN, 
    API_AUTH_HEADER, API_AUTH_TYPE
)

class SwechaAPIClient:
    """
    Client for interacting with the Swecha Corpus API
    """
    
    def __init__(self, base_url: str = SWECHA_API_BASE_URL, auth_token: str = None):
        self.base_url = base_url
        self.auth_token = auth_token or API_AUTH_TOKEN
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'TeluguCulturalHeritage/1.0'
        })
        
        # Add authentication header if token is provided
        if self.auth_token:
            if API_AUTH_TYPE.lower() == "bearer":
                self.session.headers.update({
                    API_AUTH_HEADER: f"Bearer {self.auth_token}"
                })
            else:
                self.session.headers.update({
                    API_AUTH_HEADER: f"Token {self.auth_token}"
                })
            print(f"🔐 Authentication configured with {API_AUTH_TYPE} token")
        else:
            print("⚠️  No authentication token provided - some endpoints may require authentication")
        
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Optional[Dict]:
        """Make HTTP request to the API"""
        try:
            url = f"{self.base_url}{endpoint}"
            response = self.session.request(method, url, **kwargs)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 201:
                return {"success": True, "message": "Resource created successfully"}
            elif response.status_code == 404:
                st.warning(f"Resource not found: {endpoint}")
                return None
            elif response.status_code == 401:
                st.error("Authentication required. Please check your credentials.")
                return None
            elif response.status_code == 403:
                st.error("Access forbidden. You don't have permission to access this resource.")
                return None
            else:
                st.error(f"API request failed with status {response.status_code}: {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            st.error(f"Network error: {str(e)}")
            return None
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON response: {str(e)}")
            return None
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            return None
    
    def search_content(self, query: str, category: str = None, content_type: str = None, 
                      limit: int = 20) -> List[Dict]:
        """
        Search for content using the Swecha API
        
        Args:
            query: Search query string
            category: Optional category filter (monuments, culture, traditions, folktales)
            content_type: Optional content type filter (images, videos, texts)
            limit: Maximum number of results to return
            
        Returns:
            List of search results
        """
        endpoint = "/content/search"
        
        # Prepare search parameters
        search_params = {
            "query": query,
            "limit": limit
        }
        
        if category:
            search_params["category"] = category
            
        if content_type:
            search_params["content_type"] = content_type
        
        # Try different search endpoints based on available API structure
        results = []
        
        # Method 1: Direct content search
        response = self._make_request("POST", endpoint, json=search_params)
        if response and "results" in response:
            results.extend(response["results"])
        
        # Method 2: Files search if content search doesn't work
        if not results:
            files_response = self._make_request("POST", "/files/search", json=search_params)
            if files_response and "files" in files_response:
                results.extend(files_response["files"])
        
        # Method 3: General search endpoint
        if not results:
            general_response = self._make_request("GET", f"/search?q={query}&limit={limit}")
            if general_response and "results" in general_response:
                results.extend(general_response["results"])
        
        return results[:limit]
    
    def get_content_by_id(self, content_id: str) -> Optional[Dict]:
        """Get specific content by ID"""
        return self._make_request("GET", f"/content/{content_id}")
    
    def get_content_list(self, category: str = None, content_type: str = None, 
                        page: int = 1, limit: int = 20) -> List[Dict]:
        """Get list of available content"""
        endpoint = "/content"
        params = {"page": page, "limit": limit}
        
        if category:
            params["category"] = category
        if content_type:
            params["content_type"] = content_type
            
        response = self._make_request("GET", endpoint, params=params)
        
        if response and "content" in response:
            return response["content"]
        elif response and "results" in response:
            return response["results"]
        else:
            return []
    
    def upload_content(self, file_path: str, category: str, content_type: str, 
                      metadata: Dict = None) -> Optional[Dict]:
        """
        Upload content to the Swecha API
        
        Args:
            file_path: Path to the file to upload
            category: Content category
            content_type: Type of content (images, videos, texts)
            metadata: Additional metadata for the content
            
        Returns:
            Upload response or None if failed
        """
        if not os.path.exists(file_path):
            st.error(f"File not found: {file_path}")
            return None
        
        try:
            # Prepare upload data
            upload_data = {
                "category": category,
                "content_type": content_type,
                "uploaded_at": datetime.now().isoformat()
            }
            
            if metadata:
                upload_data.update(metadata)
            
            # Try content upload endpoint first
            endpoint = "/content/upload"
            
            # Prepare multipart form data
            with open(file_path, 'rb') as f:
                files = {'file': (os.path.basename(file_path), f, 'application/octet-stream')}
                data = {'metadata': json.dumps(upload_data)}
                
                response = self._make_request("POST", endpoint, files=files, data=data)
                
                if response:
                    return response
            
            # Fallback to files upload endpoint
            endpoint = "/files/upload"
            with open(file_path, 'rb') as f:
                files = {'file': (os.path.basename(file_path), f, 'application/octet-stream')}
                data = {'metadata': json.dumps(upload_data)}
                
                response = self._make_request("POST", endpoint, files=files, data=data)
                
                if response:
                    return response
            
            return None
            
        except Exception as e:
            st.error(f"Upload failed: {str(e)}")
            return None
    
    def get_categories(self) -> List[Dict]:
        """Get available content categories"""
        response = self._make_request("GET", "/categories")
        if response and "categories" in response:
            return response["categories"]
        else:
            # Return default categories if API doesn't provide them
            return [
                {"id": "monuments", "name": "స్మారకాలు", "description": "Historical monuments and landmarks"},
                {"id": "culture", "name": "సంస్కృతి", "description": "Cultural heritage and traditions"},
                {"id": "traditions", "name": "సంప్రదాయాలు", "description": "Traditional practices and customs"},
                {"id": "folktales", "name": "జానపద కథలు", "description": "Folk tales and stories"}
            ]
    
    def get_content_types(self) -> List[Dict]:
        """Get available content types"""
        response = self._make_request("GET", "/content-types")
        if response and "content_types" in response:
            return response["content_types"]
        else:
            # Return default content types if API doesn't provide them
            return [
                {"id": "images", "name": "చిత్రాలు", "description": "Images and photographs"},
                {"id": "videos", "name": "వీడియోలు", "description": "Video content"},
                {"id": "texts", "name": "టెక్స్ట్ ఫైల్స్", "description": "Text documents and files"}
            ]
    
    def get_statistics(self) -> Optional[Dict]:
        """Get API statistics and usage information"""
        return self._make_request("GET", "/stats")
    
    def health_check(self) -> bool:
        """Check if the API is accessible"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False

# Global API client instance
swecha_client = SwechaAPIClient()

def get_swecha_search_results(query: str, category: str = None, content_type: str = None, 
                            limit: int = 20) -> List[Dict]:
    """
    Convenience function to get search results from Swecha API
    
    Args:
        query: Search query
        category: Optional category filter
        content_type: Optional content type filter
        limit: Maximum number of results
        
    Returns:
        List of search results
    """
    try:
        results = swecha_client.search_content(query, category, content_type, limit)
        
        # Transform results to match your app's expected format
        transformed_results = []
        for result in results:
            transformed_result = {
                'name': result.get('title', result.get('name', 'Unknown')),
                'path': result.get('file_path', result.get('url', '')),
                'content': result.get('content', result.get('description', '')),
                'category': result.get('category', ''),
                'content_type': result.get('content_type', ''),
                'uploaded_at': result.get('uploaded_at', ''),
                'source': 'swecha_api'
            }
            transformed_results.append(transformed_result)
        
        return transformed_results
        
    except Exception as e:
        st.error(f"Error fetching results from Swecha API: {str(e)}")
        return []

def upload_to_swecha(file_path: str, category: str, content_type: str, 
                     metadata: Dict = None) -> bool:
    """
    Convenience function to upload content to Swecha API
    
    Args:
        file_path: Path to file to upload
        category: Content category
        content_type: Content type
        metadata: Additional metadata
        
    Returns:
        True if upload successful, False otherwise
    """
    try:
        result = swecha_client.upload_content(file_path, category, content_type, metadata)
        return result is not None
    except Exception as e:
        st.error(f"Error uploading to Swecha API: {str(e)}")
        return False
