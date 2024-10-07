import helper
from nltk.stem import PorterStemmer
import re

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
                print(f"Word to remove {word}")
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


def lingustic_normalization(id, doc):
    print(f"Normalizing document {id}...")
    tokens = tokenization(doc)
    tokens = remove_stop_words(tokens)
    tokens = stemming(tokens)
    print(f"No. of Tokens: {len(tokens)}")
    print("----------------------\n")
    return tokens


def boolean_intersect(term1, term2, dictionary):
    postings1 = dictionary.get(term1, [])
    postings2 = dictionary.get(term2, [])

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


def boolean_union(term1, term2, dictionary):
    """
    Performs a Boolean union operation on two terms.

    Args:
        term1: The first term.
        term2: The second term.
        dictionary: The inverted index.

    Returns:
        A sorted list of document IDs that contain either term1 or term2.
    """

    postings1 = dictionary.get(term1, [])
    postings2 = dictionary.get(term2, [])

    if not postings1 or not postings2:
        # If either term is not found in the dictionary, return an empty list
        return []

    return sorted([doc_id for doc_id in set(postings1 + postings2)])

def boolean_not(term, all_docs, dictionary):
    postings = dictionary.get(term, [])

    # Get the set of all documents and subtract the documents in the term's postings
    return sorted(list(set(all_docs) - set(postings)))


def parse_query(query):
    """
    Parses a Boolean query into tokens (terms and operators).

    Supports:
    - AND
    - OR
    - NOT
    - Parentheses for grouping
    """
    # Tokenize the query (terms, AND, OR, NOT, parentheses)
    tokens = re.findall(r'\(|\)|AND|OR|NOT|[a-zA-Z]+', query)
    return tokens


def eval_query(tokens, dictionary, all_docs):
    """
    Evaluates a Boolean query represented as a list of tokens.

    - tokens: List of query terms and operators.
    - dictionary: The inverted index (postings lists for terms).
    - all_docs: A list of all document IDs in the collection.

    Returns: A list of document IDs satisfying the query.
    """

    def eval_and(op1, op2):
        return boolean_intersect(op1, op2, dictionary)

    def eval_or(op1, op2):
        return boolean_union(op1, op2, dictionary)

    def eval_not(op1):
        return boolean_not(op1, all_docs, dictionary)

    # Operator precedence
    precedence = {'NOT': 3, 'AND': 2, 'OR': 1}

    # Stack for values (terms and intermediate results)
    values = []

    # Stack for operators
    operators = []

    def apply_operator():
        op = operators.pop()
        if op == 'NOT':
            val = values.pop()
            values.append(eval_not(val))  # The result of NOT should be a list of IDs
        else:
            right = values.pop()
            left = values.pop()
            if op == 'AND':
                values.append(eval_and(left, right))  # The result is a list of IDs
            elif op == 'OR':
                values.append(eval_or(left, right))  # The result is a list of IDs

    # Evaluate tokens
    for token in tokens:
        if token == '(':
            operators.append(token)
        elif token == ')':
            while operators and operators[-1] != '(':
                apply_operator()
            operators.pop()  # Pop '('
        elif token in precedence:
            while (operators and operators[-1] in precedence and
                   precedence[operators[-1]] >= precedence[token]):
                apply_operator()
            operators.append(token)
        else:
            # This is a term; we need to ensure it is treated as a string
            values.append(token)  # Push the term (as a string)

    # Apply remaining operators
    while operators:
        apply_operator()

    # Final result should be a list of document IDs
    return values[0] if values else []  # Return an empty list if no values are present


if __name__ == "__main__":
    # Parse and evaluate the query

    doc1 = "apple THIS is'n SomThing. I want to know if it works. Hahah, nope JUST KIDDING. I'm not sure if it works. O. 2014"
    doc2 = "banana hello from the other side 2014"

    docs = [("doc1", doc1), ("doc2", doc2)]

    docs_tokens = []

    for id, doc in docs:
        docs_tokens.append((id, lingustic_normalization(id, doc)))

    for id, doc_tokens in docs_tokens:
        TERM_SEQUENCE = insert_and_sort_term_sequence(doc_tokens, id, TERM_SEQUENCE)
        # for i in TERM_SEQUENCE:
        # print(i)

    DICTIONARY, DOC_FREQUENCY = create_dictionary_and_postings(TERM_SEQUENCE)

    print("Dictionary and Postings:")
    for term, postings in DICTIONARY.items():
        print(f"{term}: {postings}")

    print("\nDocument Frequencies:")
    for term, freq in DOC_FREQUENCY.items():
        print(f"{term}: {freq}")

    query = "apple AND (banana OR NOT orange)"
    tokens = parse_query(query)
    result = eval_query(tokens, DICTIONARY, docs)

    print(f"Documents that satisfy the query '{query}': {result}")
