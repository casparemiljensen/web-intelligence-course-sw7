from functools import reduce

import helper
from nltk.stem import PorterStemmer

##### GLOBAL #######
TERM_SEQUENCE = []


def tokenization(doc):
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

    # A copy of the list is needed to avoid the "RuntimeError: Set changed size during iteration
    # for word in tokens[:]:
    #     if word in stop_words:
    #         tokens.remove(word)

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


def create_therm_sequence(terms, dockID, TERM_SEQUENCE):
    for term in terms:
        TERM_SEQUENCE.append((term, dockID))

    return TERM_SEQUENCE

def sort_therm_sequence(TERM_SEQUENCE):
    TERM_SEQUENCE.sort(key=lambda pair: (pair[0], pair[1]))
    return TERM_SEQUENCE

def calculate_doc_frequency(TERM_SEQUENCE):
    placeholder = ""

    term_doc_mappings = []

    for tuple_term in TERM_SEQUENCE:
        term, doc_id = tuple_term[0][1]
        if term not in term_doc_mappings:
            term_doc_mappings.append(term)
    return None


def lingustic_normalization(id, doc):
    print(f"Normalizing document {id}...")
    tokens = tokenization(doc)
    tokens = remove_stop_words(tokens)
    tokens = stemming(tokens)
    print(f"No. of Tokens: {len(tokens)}")
    print("----------------------\n")
    return tokens


doc1 = "THIS is'n SomThing. I want to know if it works. Hahah, nope JUST KIDDING. I'm not sure if it works. O. 2014"
doc2 = "hello from the other side"

docs = [("doc1", doc1), ("doc2", doc2)]

docs_tokens = []

for id, doc in docs:
    docs_tokens.append((id, lingustic_normalization(id, doc)))

for id, doc_tokens in docs_tokens:
    TERM_SEQUENCE = create_therm_sequence(doc_tokens, id, TERM_SEQUENCE)
    # for i in TERM_SEQUENCE:
        # print(i)

TERM_SEQUENCE = sort_therm_sequence(TERM_SEQUENCE)
print(TERM_SEQUENCE)
calculate_doc_frequency(TERM_SEQUENCE)


