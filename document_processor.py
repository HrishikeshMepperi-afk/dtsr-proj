import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config import Config

class DocumentProcessor:
    """Handles extracting text from documents and chunking for vector storage."""
    
    @staticmethod
    def process_pdf(uploaded_file):
        """
        Reads a Streamlit UploadedFile (PDF), extracts text, and returns a list 
        of Langchain Document objects.
        """
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
                
        # Split the text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len
        )
        
        chunks = text_splitter.split_text(text)
        
        # Convert chunks into Langchain Document objects
        documents = [Document(page_content=chunk, metadata={"source": uploaded_file.name}) for chunk in chunks]
        return documents
