import zipfile
import os
from pathlib import Path
import PyPDF2
from typing import List, Dict

class PDFExtractor:
    def __init__(self, extract_dir: str):
        self.extract_dir = Path(extract_dir)
        self.extract_dir.mkdir(exist_ok=True)
    
    def extract_zip(self, zip_path: str) -> List[str]:
        """Extract PDF files from zip archive"""
        pdf_files = []
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file_info in zip_ref.infolist():
                if file_info.filename.lower().endswith('.pdf'):
                    # Extract PDF file
                    zip_ref.extract(file_info, self.extract_dir)
                    pdf_files.append(str(self.extract_dir / file_info.filename))
        
        return pdf_files
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from PDF file"""
        text = ""
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
                    
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {str(e)}")
            
        return text
    
    def process_pdfs(self, pdf_files: List[str]) -> List[Dict]:
        """Process multiple PDF files and return documents"""
        documents = []
        
        for pdf_file in pdf_files:
            text = self.extract_text_from_pdf(pdf_file)
            if text.strip():
                documents.append({
                    'content': text,
                    'source': os.path.basename(pdf_file),
                    'type': 'pdf'
                })
        
        return documents
