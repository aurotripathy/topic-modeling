import spacy
import os
import gensim
from gensim.models import LdaModel
from gensim.corpora import Dictionary
import pyLDAvis.gensim

num_topics = 10

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load('en')


my_stop_words = [u'say', u'\'s', u'Mr', u'be', u'said', u'says', u'saying']
for stopword in my_stop_words:
    lexeme = nlp.vocab[stopword]
    lexeme.is_stop = True


test_data_dir = '{}'.format(os.sep).join([gensim.__path__[0], 'test', 'test_data'])
lee_train_file = test_data_dir + os.sep + 'lee_background.cor'
text = open(lee_train_file).read()

processed_text = nlp(text)

processed_docs, doc = [], []
for word in processed_text:
    # if it's not a stop word or punctuation mark, add it to our article!
    if word.text != '\n' and not word.is_stop and not word.is_punct and not word.like_num:
        # we add the lematized version of the word
        doc.append(word.lemma_)
    # if it's a new line, it means we're onto our next document
    if word.text == '\n':
        processed_docs.append(doc)
        doc = []

phrases = gensim.models.Phrases(processed_docs)
bigram = gensim.models.phrases.Phraser(phrases)
bigram_docs = [bigram[line] for line in processed_docs]

dictionary = Dictionary(bigram_docs)
corpus = [dictionary.doc2bow(doc) for doc in bigram_docs]

ldamodel = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)
for topic_idx in range(num_topics):
    print('Topic #{}: {}'.format(topic_idx, ldamodel.print_topic(topic_idx, topn=5)))

pyLDAvis.enable_notebook()
pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
