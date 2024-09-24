import hashlib
from docx import Document
import os


def tokenize_text(text):
    # Tokenizes the input text into words.
    return text.lower().split()


def generate_n_grams(tokens, n):
    # Generates n-grams (n-shingles) from a list of word tokens.
    n_grams = set()  # Using a set to ensure unique shingles
    for i in range(len(tokens) - n + 1):  # Iterate over tokens to create n-grams
        # The loop iterates over the list of tokens, but only up to len(tokens) - n + 1 to avoid generating incomplete n-grams.
        # Why len(tokens) - n + 1?: The index i refers to the starting position of an n-gram. It stops at len(tokens) - n + 1 because if we go beyond this, we won't have enough tokens left to form an n-gram of length n.
        # Create the n-gram by joining 'n' consecutive tokens
        n_gram = " ".join(tokens[i:i + n])
        # This line slices the tokens list starting at index i and taking n tokens (tokens[i:i + n]), then joins them into a single string separated by spaces.
        # First i is the starting index, second i + n is ending index(excluding)
        n_grams.add(n_gram)
    return n_grams


def n_gram_shingler(text, n):
    # Complete function to generate n-grams from a text.
    if (n == 0):
        raise ValueError("Please provide a value of n greater than 0.")

    if not isinstance(n, int):
        raise TypeError("n must be an integer.")

    if (len(text) == 0):
        raise ValueError("Please provide a non-empty text.")

    # First we Tokenize the text into words
    tokens = tokenize_text(text)

    if (len(tokens) < n):
        return ValueError("Less tokens than n-grams. Please provide a text with more words.")

    # Then we generate and return the n-grams
    return generate_n_grams(tokens, n)


def jaccard_similarity(set1, set2):
    # Calculates the Jaccard Similarity of two sets.
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    similarity = intersection / union
    return similarity


def get_64_bit_integer_min_hash(hash_func, data):
    """
    Compute the hash of the input data using the given hash function,
    truncate to the first 16 hexadecimal characters, and convert to a 64-bit integer.
    Hash each shingle with (64bit) hashing function:
    Store the min hash
    """
    hashes = set()
    for shingle in data:
        hash_value = hash_func(shingle.encode('utf-8')).hexdigest()
        # Make it 64 bit by truncating
        hashes.add(int(hash_value[:16], 16))

    return min(hashes)


def hash_64bit_5_permutations(data):
    hash_functions = {
        'SHA-1': hashlib.sha1,
        'SHA-224': hashlib.sha224,
        'SHA-256': hashlib.sha256,
        'SHA-512': hashlib.sha512,
        'MD5': hashlib.md5
    }

    min_hashes = set()

    # Compute and print 64-bit integer hashes
    for name, func in hash_functions.items():
        hash_value = get_64_bit_integer_min_hash(func, data)
        # Add the minhash from each hashing function
        min_hashes.add(hash_value)
        # print(f"{name} 64-bit integer hash: {hash_value}")
    return min_hashes


def calculate_similarity(document1, document2, n):
    shingles1 = n_gram_shingler(document1, n)
    shingles2 = n_gram_shingler(document2, n)

    if (len(shingles1) == 0 or len(shingles2) == 0):
        print("An error occured: One of the documents does not contain any tokens. Please provide a valid input.")
        exit(1)

    # input shingles/output a set of 5 minHashes
    min_hash_shingles1 = hash_64bit_5_permutations(shingles1)
    min_hash_shingles2 = hash_64bit_5_permutations(shingles2)
    print(f"N-gram shingling: {n}")
    print(f"MinHashes for Document 1: {min_hash_shingles1}")
    print(f"MinHashes for Document 2: {min_hash_shingles2}")

    jac_sim = jaccard_similarity(min_hash_shingles1, min_hash_shingles2)

    if (jac_sim >= 0.9):
        print("--------------------")
        print(f"The documents are similar - Jaccard Similarity: {jac_sim}")
    else:
        print("--------------------")
        print(f"The documents are not similar - Jaccard Similarity: {jac_sim}")


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


file_path1 = 'documents/document1.txt'
file_path2 = 'documents/document2.docx'

document1 = load_document(file_path1)
document2 = load_document(file_path2)
n = 8  # Change this to any n for n-grams
calculate_similarity(document1, document2, n)