import requests
from bs4 import BeautifulSoup
import pandas as pd
from typing import List, Dict
import time

class WebScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def extract_urls_from_excel(self, excel_path: str) -> List[str]:
        """Extract URLs from Excel file"""
        urls = []
        
        try:
            # Read Excel file
            df = pd.read_excel(excel_path)
            
            # Look for URL columns (common names)
            url_columns = ['url', 'link', 'website', 'URL', 'Link', 'Website']
            
            for col in df.columns:
                if col in url_columns or 'url' in col.lower() or 'link' in col.lower():
                    urls.extend(df[col].dropna().astype(str).tolist())
                    break
            
            # If no URL column found, check all columns for URLs
            if not urls:
                for col in df.columns:
                    potential_urls = df[col].dropna().astype(str)
                    for url in potential_urls:
                        if url.startswith(('http://', 'https://')):
                            urls.append(url)
            
        except Exception as e:
            print(f"Error reading Excel file: {str(e)}")
        
        return list(set(urls))  # Remove duplicates
    
    def scrape_website(self, url: str) -> str:
        """Scrape content from a single website"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text content
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
            return ""
    
    def process_urls(self, urls: List[str]) -> List[Dict]:
        """Process multiple URLs and return documents"""
        documents = []
        
        for url in urls:
            print(f"Scraping: {url}")
            content = self.scrape_website(url)
            
            if content.strip():
                documents.append({
                    'content': content,
                    'source': url,
                    'type': 'website'
                })
            
            # Be respectful with requests
            time.sleep(1)
        
        return documents
