from gensim import corpora
from gensim import models
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string

documents = ["Human machine interface for lab abc computer applications",
             "A survey of user opinion of computer system response time",
             "The EPS user interface management system",
             "System and human system engineering testing of EPS",
             "Relation of user perceived response time to error measurement",
             "The generation of random binary unordered trees",
             "The intersection graph of paths in trees",
             "Graph minors IV Widths of trees and well quasi ordering",
             "Graph minors A survey"]

print('Building LDA model using following documents:')
for doc in documents:
    print('\t {}'.format(doc))

stop = set(stopwords.words('english')) #  common words
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

def clean(doc):
    stop_free = " ".join([word for word in doc.lower().split() if word not in stop])
    stop_punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    stop_punc_free_lemmatized = " ".join(lemma.lemmatize(word) for word in stop_punc_free.split())
    return stop_punc_free_lemmatized

cleaned_documents = [clean(doc).split() for doc in documents]  

num_topics = 2
top_words_count = 5

# Create the term dictionary of our courpus, where every unique term is assigned an index. 
dictionary = corpora.Dictionary(cleaned_documents)

# Convert list of documents into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in cleaned_documents]

# Train LDA model on the document term matrix.
ldamodel = models.ldamodel.LdaModel(corpus=doc_term_matrix, num_topics=num_topics, id2word = dictionary, passes=50)

topic_labels = []
for topic in range(num_topics):
    words_in_topic = ldamodel.show_topic(topic,topn=top_words_count)
    topic_label = words_in_topic[0][0] + '_' + words_in_topic[1][0] # top two words becomes the label
    topic_labels.append(topic_label)
    print('Topic #{}: {}: {}'.format(topic, topic_label, words_in_topic))


test_documents = ["The user interface management system",
                  "The intersection graph of paths in trees",
                  "A survey of graph minors"]
print('Testing model with following unseen documents:')

for doc in test_documents:
    print('\t {}'.format(doc))

cleaned_test_documents = [clean(doc).split() for doc in test_documents]  
test_doc_term_matrix = [dictionary.doc2bow(doc) for doc in cleaned_test_documents] # working with the orig dict

test_doc_topics = ldamodel[test_doc_term_matrix]

import numpy as np
for d, topic_mix in enumerate(test_doc_topics):
    print('-------------')
    print('Doc: {}'.format(test_documents[d]))
    for t in range(num_topics):
        print(topic_labels[t], topic_mix[t][1])
    print('Topic: {}'.format(topic_labels[np.argmax(list(zip(*topic_mix))[1])]))
