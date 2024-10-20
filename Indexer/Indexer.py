import json
import os

import helper
from nltk.stem import PorterStemmer

##### GLOBAL #######
TERM_SEQUENCE = []  # Initialize the sequence of terms
INVERTED_INDEX = {}  # Initialize the dictionary for postings
DOC_FREQUENCY = {}  # Initialize document frequency tracking

# To do strip and remove white spaces
# what about æøå - only handle english texts.
# alter eval_query to read from inverted_index.json and doc_frequency.json, now it reads empty inverted index, thats why it does not work.
# something wrong with the eval query and bool intersect, not getting dicts and freqs.

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "./../"))


def tokenization(doc):
    """
    Tokenizes a document into a list of words.
    """
    stream = doc.split(" ")

    remove = [",", ".", "-", "!", "?", ":", ";", "(", ")", "[", "]", "{", "}", "<", ">", "/", "\\", "|", "@", "#", "$",
              "%", "^", "&", "*", "_", "+", "=", "~", "`", "´", "\"", "'"]
    # WordNet for synonyms
    for i in range(len(stream)):
        stream[i] = stream[i].strip()
        stream[i] = stream[i].lower()

        for j in range(len(remove)):
            stream[i] = stream[i].replace(remove[j], "")

    return stream


def remove_stop_words(tokens):
    directory = os.path.join(PROJECT_ROOT, "indexer", "stopwords_en.txt")

    stop_words = helper.load_document(directory).split("\n")
    for word in tokens:
        for stopword in stop_words:
            if word == stopword:
                tokens.remove(word)
                # print(f"Word to remove {word}")
                break

    return tokens


def stemming(tokens):
    ps = PorterStemmer()
    # Apply stemmer to each token and return the list of stemmed words
    stemmed_tokens = [ps.stem(token) for token in tokens]
    return stemmed_tokens


def insert_into_inverted_index(terms, docID, index, doc_frequency):
    """
    Inserts terms along with their docID into the dictionary and postings structure.

    - terms: A list of terms (e.g., tokens) to be inserted.
    - docID: The identifier of the document.
    - dictionary: The dictionary where terms map to their postings lists.
    - doc_frequency: The dictionary where terms map to their document frequency.
    """
    for term in terms:
        if term not in index:
            index[term] = []  # Initialize postings list for new term

        # Avoid adding duplicate docIDs
        if not index[term] or index[term][-1] != docID:
            index[term].append(docID)

    # Update document frequency
    for term in terms:
        if term in index:
            doc_frequency[term] = len(index[term])  # Update frequency count


def create_dictionary_and_postings():
    """
    Constructs a dictionary and postings list from the TERM_SEQUENCE.

    TERM_SEQUENCE: A sorted list of (term, docID) tuples.

    Returns:
    - A dictionary where each term points to its posting list.
    - A dictionary where each term points to its document frequency.
    """
    dictionary = {}  # Dictionary to store term -> postings list
    doc_frequency = {}  # Dictionary to store term -> document frequency

    current_term = None
    current_postings = []

    for term, doc_id in TERM_SEQUENCE:
        if term != current_term:
            # If we have a new term, finalize the postings list of the previous term
            if current_term is not None:
                dictionary[current_term] = current_postings
                doc_frequency[current_term] = len(current_postings)  # Document frequency is the size of postings list

            # Reset for the new term
            current_term = term
            current_postings = [doc_id]  # Start the new postings list
        else:
            # Avoid adding duplicate doc_ids (since the list is sorted)
            if current_postings[-1] != doc_id:
                current_postings.append(doc_id)

    # Finalize the last term's postings list
    if current_term is not None:
        dictionary[current_term] = current_postings
        doc_frequency[current_term] = len(current_postings)

    return dictionary, doc_frequency


def insert_and_sort_term_sequence(terms, docID):
    """
    Inserts terms along with their docID into TERM_SEQUENCE and keeps it sorted.

    - terms: A list of terms (e.g., tokens) to be inserted.
    - docID: The identifier of the document.
    - TERM_SEQUENCE: The sequence (list) to which terms will be appended, as (term, docID) pairs.
    """
    for term in terms:
        TERM_SEQUENCE.append((term, docID))

    # Sort the sequence right after inserting new terms.
    # First by term, then by docID.
    TERM_SEQUENCE.sort(key=lambda pair: (pair[0], pair[1]))


def lingustic_normalization(doc):
    # print(f"Normalizing document...")
    tokens = tokenization(doc)
    tokens = remove_stop_words(tokens)
    tokens = stemming(tokens)
    # print(f"No. of Tokens: {len(tokens)}")
    # print("----------------------\n")
    return tokens


def boolean_intersect(term1, term2):
    if type(term1) is list and type(term2) is list:
        postings1 = term1
        postings2 = term2
    elif type(term1) is str and type(term2) is str:
        postings1 = INVERTED_INDEX.get(term1, [])
        postings2 = INVERTED_INDEX.get(term2, [])


    else:
        return []

    result = []
    i, j = 0, 0
    while i < len(postings1) and j < len(postings2):
        if postings1[i] == postings2[j]:
            result.append(postings1[i])
            i += 1
            j += 1
        elif postings1[i] < postings2[j]:
            i += 1
        else:
            j += 1
    return result


def eval_query(query):
    global INVERTED_INDEX
    query_tokens = lingustic_normalization(query)
    DOC_FREQUENCY = helper.load_document(os.path.join(PROJECT_ROOT, "lib", "index_data", "doc_frequency.json"))
    INVERTED_INDEX = helper.load_document(os.path.join(PROJECT_ROOT, "lib", "index_data", "inverted_index.json"))

    if len(query_tokens) == 0:
        return []

    # Check if there is any query token not in the doc_frequency
    for query_token in query_tokens:
        if query_token not in DOC_FREQUENCY:
            return []

    visited = len(query_tokens)
    total_result = []

    # Iterating over all query terms.
    # If only two terms, we return after one iteration.
    # Else we intersect the current result with new result of two terms.
    while visited > 0:
        for i in range(len(query_tokens)):
            term1 = query_tokens[i]
            term2 = query_tokens[i - 1]
            if term1 in INVERTED_INDEX and term2 in INVERTED_INDEX:
                partial_res = boolean_intersect(term1, term2)
                if len(total_result) > 0:
                    total_result = boolean_intersect(total_result, partial_res)
                else:
                    total_result = partial_res
                visited -= 1
            else:
                return []
    return list(total_result)  # Return the final intersected results


# Function to save a dictionary to a JSON file
def save_dict_to_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)


def run_indexer():
    docs = helper.load_documents_from_directory("lib/crawled_pages")
    docs_tokens = []

    for id, doc in docs:
        docs_tokens.append((id, lingustic_normalization(doc)))

    for id, doc_tokens in docs_tokens:
        insert_and_sort_term_sequence(doc_tokens, id)

    INVERTED_INDEX, DOC_FREQUENCY = create_dictionary_and_postings()

    # # Create the output directory if it doesn't exist
    output_dir = "lib/index_data"
    os.makedirs(output_dir, exist_ok=True)

    # Save inverted index and document frequency to separate files
    save_dict_to_json(INVERTED_INDEX, os.path.join(output_dir, 'inverted_index.json'))
    save_dict_to_json(DOC_FREQUENCY, os.path.join(output_dir, 'doc_frequency.json'))


if __name__ == "__main__":
    # Parse and evaluate the query

    doc1 = "aalborg engineer is'n SomThing banana. I want to know if it works. Hahah, nope JUST KIDDING. I'm not sure if it works. O. 2014"
    doc2 = "banana hello from the other side 2014"

    docs = [("doc1", doc1), ("doc2", doc2)]

    docs_tokens = []

    for id, doc in docs:
        docs_tokens.append((id, lingustic_normalization(doc)))

    for id, doc_tokens in docs_tokens:
        insert_and_sort_term_sequence(doc_tokens, id)

    INVERTED_INDEX, DOC_FREQUENCY = create_dictionary_and_postings()

    print("Dictionary and Postings:")
    for term, postings in INVERTED_INDEX.items():
        print(f"{term}: {postings}")

    print("\nDocument Frequencies:")
    for term, freq in DOC_FREQUENCY.items():
        print(f"{term}: {freq}")

    query = "banana AND aalborg AND engineer"
    print(f"Processing query({query})...")

    result = eval_query(query)
    print(f"Result: {result}")
