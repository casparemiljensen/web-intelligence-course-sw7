import helper
from nltk.stem import PorterStemmer

##### GLOBAL #######
TERM_SEQUENCE = []
DICTIONARY = {}  # Initialize the dictionary for postings
DOC_FREQUENCY = {}  # Initialize document frequency tracking


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
    stop_words = helper.load_document("stopwords_en.txt").split("\n")
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


def insert_into_dictionary(terms, docID, dictionary, doc_frequency):
    """
    Inserts terms along with their docID into the dictionary and postings structure.

    - terms: A list of terms (e.g., tokens) to be inserted.
    - docID: The identifier of the document.
    - dictionary: The dictionary where terms map to their postings lists.
    - doc_frequency: The dictionary where terms map to their document frequency.
    """
    for term in terms:
        if term not in dictionary:
            dictionary[term] = []  # Initialize postings list for new term

        # Avoid adding duplicate docIDs
        if not dictionary[term] or dictionary[term][-1] != docID:
            dictionary[term].append(docID)

    # Update document frequency
    for term in terms:
        if term in dictionary:
            doc_frequency[term] = len(dictionary[term])  # Update frequency count


def create_dictionary_and_postings(TERM_SEQUENCE):
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


def insert_and_sort_term_sequence(terms, docID, TERM_SEQUENCE):
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

    return TERM_SEQUENCE


def lingustic_normalization(doc):
    # print(f"Normalizing document...")
    tokens = tokenization(doc)
    tokens = remove_stop_words(tokens)
    tokens = stemming(tokens)
    # print(f"No. of Tokens: {len(tokens)}")
    # print("----------------------\n")
    return tokens


def boolean_intersect(term1, term2):
    postings1 = DICTIONARY.get(term1, [])
    postings2 = DICTIONARY.get(term2, [])

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


def eval_query(query_tokens):
    if len(query_tokens) == 0:
        return []

    visited = len(query_tokens)
    result = []  # Initialize result as a list to store final results

    # Start with the first token's postings
    result = set(DICTIONARY.get(query_tokens[0], []))

    for i in range(1, visited):
        term = query_tokens[i]
        if term in DICTIONARY:  # Check if the term exists in the dictionary
            result = boolean_intersect(query_tokens[i - 1], term)
        else:
            return []  # If any term is not found, return empty list

    return list(result)  # Return the final intersected results


if __name__ == "__main__":
    # Parse and evaluate the query

    doc1 = "aalborg engineer is'n SomThing banana. I want to know if it works. Hahah, nope JUST KIDDING. I'm not sure if it works. O. 2014"
    doc2 = "banana hello from the other side 2014"

    docs = [("doc1", doc1), ("doc2", doc2)]

    docs_tokens = []

    for id, doc in docs:
        docs_tokens.append((id, lingustic_normalization(doc)))

    for id, doc_tokens in docs_tokens:
        TERM_SEQUENCE = insert_and_sort_term_sequence(doc_tokens, id, TERM_SEQUENCE)
        # for i in TERM_SEQUENCE:
        # print(i)

    DICTIONARY, DOC_FREQUENCY = create_dictionary_and_postings(TERM_SEQUENCE)

    # print("Dictionary and Postings:")
    # for term, postings in DICTIONARY.items():
    #     print(f"{term}: {postings}")
    #
    # print("\nDocument Frequencies:")
    # for term, freq in DOC_FREQUENCY.items():
    #     print(f"{term}: {freq}")

    query = "banana AND aalborg AND engineer"
    print(f"Processing query({query})...")
    query_tokens = lingustic_normalization(query)
    result = eval_query(query_tokens)
    print(f"Result: {result}")
