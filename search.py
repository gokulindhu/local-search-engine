import os
import ollama
import json
import sys
from docx import Document
from PyPDF2 import PdfReader
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Set base directory for files
BASE_DIR = Path(__file__).resolve().parent
FILE_DIRECTORY = BASE_DIR / "files"

def extract_text_from_file(file_path):
    """Extract text from different file formats."""
    text = ""
    try:
        if file_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
        elif file_path.endswith(".docx"):
            doc = Document(file_path)
            text = "\n".join(para.text for para in doc.paragraphs)
        elif file_path.endswith(".pdf"):
            reader = PdfReader(file_path)
            text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return text.strip()  # Ensure empty text is properly handled

def process_file(file_path, query):
    """Extract text from a file and check for relevant content using Ollama."""
    content = extract_text_from_file(file_path)
    if not content:
        return None  # Skip empty files

    # Prevent sending excessive text to Ollama
    max_chars = 4000  # Adjust based on model limitations
    trimmed_content = content[:max_chars] if len(content) > max_chars else content

    try:
        # Use Ollama to check relevance
        response = ollama.chat(
            model="mistral",
            messages=[{"role": "user", "content": f"Find relevant content for: {query} in the following text:\n{trimmed_content}"}]
        )
        answer = response["message"]["content"].strip()
        if answer:
            return {"file": os.path.basename(file_path), "content": answer}
    except Exception as e:
        print(f"Error processing {file_path} with Ollama: {e}")
    
    return None

def search_in_files(query):
    """Search relevant content in files using parallel processing."""
    results = []
    files = [os.path.join(FILE_DIRECTORY, f) for f in os.listdir(FILE_DIRECTORY) if f.endswith((".txt", ".docx", ".pdf"))]

    with ThreadPoolExecutor(max_workers=5) as executor:  # Adjust workers as needed
        futures = [executor.submit(process_file, file, query) for file in files]
        for future in futures:
            result = future.result()
            if result:
                results.append(result)

    return json.dumps(results, indent=2)

# Example usage
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Query is required"}))
        sys.exit(1)

    query = sys.argv[1]
    response = search_in_files(query)
    print(response)
