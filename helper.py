from docx import Document
import os


def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def read_word_file(file_path):
    doc = Document(file_path)
    return ' '.join(paragraph.text for paragraph in doc.paragraphs)


def load_document(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    _, file_extension = os.path.splitext(file_path)

    if file_extension.lower() == '.txt':
        return read_text_file(file_path)
    elif file_extension.lower() == '.docx':
        return read_word_file(file_path)
    else:
        raise ValueError("Unsupported file type. Please provide a .txt or .docx file.")
