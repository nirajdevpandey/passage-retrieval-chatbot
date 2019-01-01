
# coding: utf-8
"""

Author __Niraj Dev Pandey__
Purpose __ TF_IDF Pssage retrieval bot__

"""

import re, collections, math, random

def paragraphs(file, separator=None):
    if not callable(separator):
        if separator != None: 
            print("TypeError separator must be callable")
        def separator(line): 
            return line == '\n'
    paragraph = []
    for line in file:
        if separator(line):
            if paragraph:
                yield ''.join(paragraph)
                paragraph = []
        else:
            paragraph.append(line)
    if paragraph :
        yield ''.join(paragraph)


response_file = "oscar-wilde-quotes.txt"
with open(response_file, 'rb') as f :
    data = f.read().decode('ISO-8859-1')
    responses = [line.strip() for line in paragraphs(data)]

responses[16:25]
data[0:200]

stop_word_file = "stop-word-list.txt"
with open(stop_word_file,"rb") as f :
    stop_words = set(line.strip() for line in f)
    
def content_words(text) :
    return [w.lower() for w in re.findall(r"\w(?:'?\w)*", text)
            if w.lower() not in stop_words]


def count_train(text, vocabulary) :
    words = content_words(text)
    vocabulary.update(words)
    return collections.Counter(words)

def count_test(text, vocabulary) :
    words = [w for w in content_words(text) if w in vocabulary]
    return collections.Counter(words)


def mk_idf(vocabulary, counts) :
    ndocs = float(len(counts))
    result = dict()
    for w in vocabulary :
        result[w] = math.log(ndocs / sum(1 if count[w] > 0 else 0 for count in counts))
    return result

def mk_tf_idf(count, idf) :
    total = sum(count[w] for w in count)
    result = dict((w, idf[w] * count[w] / total) for w in count)
    length = math.sqrt(sum(result[w] * result[w] for w in result))
    for w in result :
        result[w] = result[w] / length
    return result

vocabulary = set()
counts = [count_train(utt, vocabulary) for utt in responses]
idf_dict = mk_idf(vocabulary, counts)
scores = [mk_tf_idf(count, idf_dict) for count in counts]


def similarity(d1, d2) :
    return sum(d1[w] * d2[w] for w in d1 if w in d2)


def sort_responses(tf_idf_utt, scores, utts) :
    options = [(utts[i], similarity(tf_idf_utt, scores[i]))
               for i in range(len(utts))]
    return sorted(options,reverse=True)


def weighted_random_item(options) :
    total = sum(w for (u,w) in options)
    r = random.uniform(0,total)
    i = 0
    while r > 0 :
        u, w = options[i]
        if r < w or i == len(options) - 1 :
            return u
        else :
            r = r - w
            i = i + 1

def respond(text) :
    content = count_test(text, vocabulary)
    tf_idf_utt = mk_tf_idf(content, idf_dict)
    return(weighted_random_item(sort_responses(tf_idf_utt, scores, responses)))

if __name__ == '__main__':
    print("""
Hi Niraj

Talk to the me by typing in plain English and please Enter "quit" when done.'""")
    print('='*72)
    print()
    s = ""
    while s != "quit":
        
        s = input(">>")
        while s and s[-1] in "!.":
            s = s[:-1]        
        
        print(respond(s))

