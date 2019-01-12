
# coding: utf-8
"""
Author @NirajDevPandey
Purpose = Passage search for a given query using Doc2Vec algorithm. this will train your own 
text passages and return the most similar paragraph from the corpus. The more passages you 
have the better it works. I would recommend to use pre trained model if you have less data.
"""

from os import listdir
from os.path import isfile, exists, join
from os import walk
import gensim
from gensim.models.doc2vec import LabeledSentence
from gensim.models.doc2vec import TaggedLineDocument
import re
import random
import smart_open
import collections
from random import shuffle
import warnings
warnings.filterwarnings("ignore")


filename = ('data/data.txt')


def read_corpus(fname, tokens_only=False):
    with smart_open.smart_open(fname, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if tokens_only:
                yield gensim.utils.simple_preprocess(line, min_len=3)
            else:
                yield gensim.models.doc2vec.TaggedDocument(
                    gensim.utils.simple_preprocess(line, min_len=3), [i]
                )



train_corpus = list(read_corpus(filename))


"""
To know that how many paragraphs your text file has
"""

len(train_corpus)


model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=50)


model.build_vocab(train_corpus)


model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)



ranks = []
second_ranks = []
for doc_id in range(len(train_corpus)):
    inferred_vector = model.infer_vector(train_corpus[doc_id].words)
    sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
    rank = [docid for docid, sim in sims].index(doc_id)
    ranks.append(rank)
    
    second_ranks.append(sims[1])


def similarDoc(doc_id,sim_id):
    """" 
    Pick a random document from the corpus and infer a vector from the model
    then Compare and print the second-most-similar document
    
    """
    print('Train Document ({}): «{}»\n'.format(doc_id, ' '.join(train_corpus[doc_id].words)))
    print()
    print('Similar Document {}: «{}»\n'.format(sim_id, ' '.join(train_corpus[sim_id[0]].words)))




#if you want to see that which passages are similar in nature uncomment this
"""
doc_id = random.randint(0, len(train_corpus) - 1)
sim_id = second_ranks[doc_id]
similarDoc(doc_id,sim_id)
"""

while True:
    test_corpus = input("Enter your queries and press 'stop' when done >> ")
    inferred_vector = model.infer_vector(test_corpus)
    sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))

    for label, index in [('MOST', 0)]: #, ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
        print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))
    if test_corpus == "stop":
        break

