import hashlib
import random

from docx import Document
import os

import helper


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


# def hash_and_permute_shingles(shingles, num_perm, seed):
#     hash_function = hashlib.sha256  # Using MD5 as an example
#     min_hashes = []
#     random.seed(seed)
#     # print(random.getstate()[1][0])  # Take the minimum hash value
#     # Step 1: Hash all shingles first
#     hashed_shingles = [hash_function(shingle.encode('utf-8')).hexdigest()[:16] for shingle in shingles]
#     hashed_shingles = [int(hash_val, 16) for hash_val in hashed_shingles]  # Convert to integer
#
#     # Step 2: Permutate and extract min index
#     for _ in range(num_perm):
#         shuffled = list(hashed_shingles)
#         random.shuffle(shuffled)
#         min_hash = shuffled[0]
#         min_hashes.append(min_hash)
#     return set(min_hashes)

def hash_and_permute_shingles(shingles1, shingles2, num_permutations):
    hash_function = hashlib.md5  # Using MD5 as an example
    min_hashes = []

    # Step 1: Hash all shingles first
    hashed_shingles1 = [hash_function(shingle1.encode('utf-8')).hexdigest()[:16] for shingle1 in shingles1]
    hashed_shingles1 = [int(hash_val, 16) for hash_val in hashed_shingles1]  # Convert to integer

    hashed_shingles2 = [hash_function(shingle2.encode('utf-8')).hexdigest()[:16] for shingle2 in shingles2]
    hashed_shingles2 = [int(hash_val, 16) for hash_val in hashed_shingles2]  # Convert to integer

    #print(hashed_shingles)

    random.seed(0)

    # Step 2: Permutate and extract min index
    for _ in range(num_permutations):
        # Shuffle the hashed shingles
        shuffled1 = list(hashed_shingles1)
        shuffled2 = list(hashed_shingles2)
        st = random.getstate()
        random.shuffle(shuffled1)
        random.setstate(st)
        random.shuffle(shuffled2)
        # Take the minimum hash value
        #min_hash = shuffled[0]
        min_hashed_shingles1.add(shuffled1[0])
        min_hashed_shingles2.add(shuffled2[0])
        #min_hashes.append(min_hash)
    #print(min_hashes)
    #return min_hashes


def calculate_similarity(doc1, doc2, n, num_perm, seed):
    try:
        shingles1 = n_gram_shingler(doc1, n)
        shingles2 = n_gram_shingler(doc2, n)

        if len(shingles1) == 0 or len(shingles2) == 0:
            raise ValueError("One of the documents does not contain any valid shingles.")

        # Generate min-hashes
        # min_hashed_shingles1 = hash_and_permute_shingles(shingles1, num_perm, seed)
        # min_hashed_shingles2 = hash_and_permute_shingles(shingles2, num_perm, seed)

        hash_and_permute_shingles(shingles1, shingles2, num_perm)

        if len(min_hashed_shingles1) == 0 or len(min_hashed_shingles2) == 0:
            return

        print(f"N-gram shingling: {n}")
        intersection = min_hashed_shingles1.intersection(min_hashed_shingles2)
        union = len(min_hashed_shingles1.union(min_hashed_shingles2))

        print(f"Probability: {len(intersection) / union * 100}")

        max_entries = len(max(min_hashed_shingles1, min_hashed_shingles2))

        print(f"Number of shingles1: {len(shingles1)}")
        print(f"Number of shingles2: {len(shingles2)}")
        print(f"Number of matches: {len(intersection)}")
        print(f"Matches is: {intersection}")
        print(f"Similarity: {len(intersection) / max_entries * 100:.2f}%")

    except ValueError as ve:
        print(f"Error: {ve}")


def count_same_entries(list1, list2):
    min_hashes_1 = set(list1)
    min_hashes_2 = set(list2)
    same_entries_count = len(min_hashes_1.intersection(min_hashes_2))

    return same_entries_count


# def read_text_file(file_path):
#     with open(file_path, 'r', encoding='utf-8') as file:
#         return file.read()
#
#
# def read_word_file(file_path):
#     doc = Document(file_path)
#     return ' '.join(paragraph.text for paragraph in doc.paragraphs)
#
#
# def load_document(file_path):
#     if not os.path.isfile(file_path):
#         raise FileNotFoundError(f"The file {file_path} does not exist.")
#
#     _, file_extension = os.path.splitext(file_path)
#
#     if file_extension.lower() == '.txt':
#         return read_text_file(file_path)
#     elif file_extension.lower() == '.docx':
#         return read_word_file(file_path)
#     else:
#         raise ValueError("Unsupported file type. Please provide a .txt or .docx file.")


file_path1 = 'documents/document1.txt'
file_path2 = 'documents/document1_940.txt'

document1 = helper.load_document(file_path1)
document2 = helper.load_document(file_path2)
n = 8  # Change this to any n for n-grams
num_permutations = 2500
seed_value = 0

min_hashed_shingles1 = set()
min_hashed_shingles2 = set()

calculate_similarity(document1, document2, n, num_permutations, seed_value)
