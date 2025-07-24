from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.pdf_extractor import PDFExtractor
from utils.web_scraper import WebScraper
import config

class DataProcessor:
    def __init__(self):
        self.pdf_extractor = PDFExtractor(config.EXTRACTED_DIR)
        self.web_scraper = WebScraper()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
        )
    
    def process_zip_file(self, zip_path: str) -> List[Dict]:
        """Process ZIP file containing PDFs"""
        # Extract PDFs from ZIP
        pdf_files = self.pdf_extractor.extract_zip(zip_path)
        
        # Process PDFs
        documents = self.pdf_extractor.process_pdfs(pdf_files)
        
        return documents
    
    def process_excel_file(self, excel_path: str) -> List[Dict]:
        """Process Excel file containing URLs"""
        # Extract URLs from Excel
        urls = self.web_scraper.extract_urls_from_excel(excel_path)
        
        # Scrape websites
        documents = self.web_scraper.process_urls(urls)
        
        return documents
    
    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """Split documents into smaller chunks"""
        chunked_docs = []
        
        for doc in documents:
            chunks = self.text_splitter.split_text(doc['content'])
            
            for i, chunk in enumerate(chunks):
                chunked_docs.append({
                    'content': chunk,
                    'source': doc['source'],
                    'type': doc['type'],
                    'chunk_id': i
                })
        
        return chunked_docs
