import re
from gensim.models import Word2Vec
from gensim.test.utils import datapath
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd


def visualize_embeddings(model, top_n=100):
    words = list(model.wv.index_to_key[:top_n])
    vectors = [model.wv[word] for word in words]
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(vectors)

    # Plot
    plt.figure(figsize=(12, 8))
    plt.scatter(reduced[:, 0], reduced[:, 1])
    for i, word in enumerate(words):
        plt.annotate(word, (reduced[i, 0], reduced[i, 1]), fontsize=9)
    plt.title("Word Embeddings Visualization")
    plt.show()


# Load the CMU Book Summary Dataset
# Assuming 'book_summaries.txt' is downloaded and located in the working directory
# def load_data(file_path):
#     sentences = []
#     with open(file_path, 'r', encoding='utf-8') as file:
#         # for line in file:
#         for i, line in enumerate(file):
#             if i >= 10000:
#                 break
#             # Tokenize and preprocess each line
#             words = word_tokenize(line.lower())
#             words = [re.sub(r'\W+', '', word) for word in words]  # Remove special characters
#             words = [word for word in words if word.isalpha()]  # Keep alphabetic words only
#             words = [word for word in words if word not in stopwords.words('english')]  # Remove stopwords
#             if words:
#                 sentences.append(words)
#     return sentences


def load_data(file_path, max_lines=2000, batch_size=5000):
    stop_words = set(stopwords.words('english'))  # Precompute stopwords set
    sentences = []

    with open(file_path, 'r', encoding='utf-8') as file:
        batch = []
        for i, line in enumerate(file):
            if i >= max_lines:
                break
            batch.append(line)

            # Process in batches
            if len(batch) >= batch_size:
                print(f'New batch started {i}')
                sentences.extend(process_batch(batch, stop_words))
                batch = []

        # Process remaining lines
        if batch:
            sentences.extend(process_batch(batch, stop_words))

    return sentences


def process_batch(batch, stop_words):
    processed = []
    for line in batch:
        words = word_tokenize(line.lower())
        words = [re.sub(r'\W+', '', word) for word in words if word.isalpha()]
        words = [word for word in words if word not in stop_words]
        if words:
            processed.append(words)
    return processed


# File path to the CMU Book Summary dataset
file_path = './booksummaries.txt'
sentences = load_data(file_path)
print(f"Loaded {len(sentences)} sentences.")

# USING NEGATIVE SAMPLING: 10
# The value 10 means that for each positive training example, 10 "negative" samples (random words from the vocabulary) are drawn and trained against.

# Train SkipGram model
model_skipgram = Word2Vec(sentences, vector_size=100, window=3, sg=1, negative=10, epochs=3)

# Train CBOW model
model_cbow = Word2Vec(sentences, vector_size=100, window=3, sg=0, negative=10, epochs=3)

# This is SkipGram using Word2Vec


print("Training started")

# USING Hierarchial Softmax

# # Train SkipGram model
# print("Training skipgram")
# model_skipgram = Word2Vec(sentences, vector_size=100, window=3, sg=1, hs=1, negative=0, epochs=100, min_count=10)
#
# # Train CBOW model
# print("Training CBOW")
# model_cbow = Word2Vec(sentences, vector_size=100, window=3, sg=0, hs=1, negative=0, epochs=100, min_count=10)

# https://radimrehurek.com/gensim/models/word2vec.html

print("Calculating accuracy")

# Evaluate using the analogy dataset
accuracy_skipgram = model_skipgram.wv.evaluate_word_analogies(datapath('questions-words.txt'))
accuracy_cbow = model_cbow.wv.evaluate_word_analogies(datapath('questions-words.txt'))

# print(f"SkipGram Overall Accuracy: {accuracy_skipgram[0]}")
# print(f"CBOW Overall Accuracy: {accuracy_cbow[0]}")

# # Evaluate word similarity (requires a word similarity dataset, e.g., 'wordsim353.tsv')
# similarity_skipgram = model_skipgram.wv.evaluate_word_pairs(datapath('./wordsim353.tsv'))
# similarity_cbow = model_cbow.wv.evaluate_word_pairs(datapath('./wordsim353.tsv'))

# print(f"SkipGram Similarity: {similarity_skipgram[0]['spearman'][0]}")
# print(f"CBOW Similarity: {similarity_cbow[0]['spearman'][0]}")

# Summary of performance
results = {
    "Model": ["SkipGram", "CBOW"],
    "Analogy Accuracy": [accuracy_skipgram[0], accuracy_cbow[0]],
}

results_df = pd.DataFrame(results)
print(results_df)
