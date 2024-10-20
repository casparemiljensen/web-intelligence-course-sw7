import json

from docx import Document
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "./../"))

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def read_word_file(file_path):
    doc = Document(file_path)
    return ' '.join(paragraph.text for paragraph in doc.paragraphs)

def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def load_document(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    _, file_extension = os.path.splitext(file_path)

    if file_extension.lower() == '.txt':
        return read_text_file(file_path)
    elif file_extension.lower() == '.docx':
        return read_word_file(file_path)
    elif file_extension.lower() == '.json':
        return read_json_file(file_path)
    else:
        raise ValueError("Unsupported file type. Please provide a .txt or .docx file.")


def is_dir_empty(dir_path):
    directory = os.path.join(PROJECT_ROOT, dir_path)
    """Check if a directory is empty."""
    return len(os.listdir(directory))


def load_documents_from_directory(base_directory):
    """
    Load and return all documents from JSON files in all subdirectories of the specified base directory.
    - base_directory: The directory containing subdirectories with the crawled JSON files.
    """
    documents = []

    # Iterate over all subdirectories inside the base directory
    for subdir in os.listdir(base_directory):
        subdir_path = os.path.join(base_directory, subdir)

        if os.path.isdir(subdir_path):  # Ensure it is a directory
            # Iterate over all files in the subdirectory
            for filename in os.listdir(subdir_path):
                if filename.endswith(".json"):  # Ensure we're only processing JSON files
                    filepath = os.path.join(subdir_path, filename)
                    with open(filepath, 'r', encoding='utf-8') as file:
                        # Load the JSON data
                        data = json.load(file)
                        # Assuming the document text is stored under the "text" key in the JSON
                        documents.append((filename, data.get('text', '')))  # Handle cases where 'text' might not exist

    return documents