import PyPDF2
from docx import Document
import io
from typing import Optional

class DocumentProcessor:
    def extract_text(self, file_path: str) -> str:
        """Extract text from a document (PDF or DOCX)."""
        if file_path.lower().endswith('.pdf'):
            return self._extract_text_from_pdf(file_path)
        elif file_path.lower().endswith(('.doc', '.docx')):
            return self._extract_text_from_docx(file_path)
        else:
            raise ValueError("Unsupported file format. Please upload a PDF or DOCX file.")
    
    def _extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from a PDF file."""
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = []
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text.append(page_text)
                return '\n'.join(text)
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    def _extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from a DOCX file."""
        try:
            doc = Document(file_path)
            return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        except Exception as e:
            raise Exception(f"Error extracting text from DOCX: {str(e)}")
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text for better processing."""
        # Remove extra whitespace and newlines
        import re
        text = re.sub(r'\s+', ' ', text).strip()
        return text

# Example usage:
if __name__ == "__main__":
    processor = DocumentProcessor()
    # Test with a sample file
    # text = processor.extract_text("path/to/your/resume.pdf")
    # print(processor.clean_text(text))
