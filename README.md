### passage-retrieval-chatbot

Input a text file separated with many paragraphs and ask a question to get relevant passage back based on TF-IDF wights 
***
#### Following is the details & workflow of the repository 
This chatbot work with a text file which has number of passages in it. Most of the work comes in preprocessing this collection to index its candidate utterances using the TFIDF model so we can easily find the utterance that's most similar to what the user has just said. The devision of the passage is based on the blank space between two paragraphs. 

```python
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
 ```

In a large text corpus, some words will be very present (e.g. `the`, `a`, `is` in English) hence carrying very little meaningful information about the actual contents of the document. If we were to feed the direct count data directly to a classifier those very frequent terms would shadow the frequencies of rarer yet more interesting terms.

In order to re-weight the count features into floating point values suitable for usage by a classifier it is very common to use the tf–idf transform.

![title](https://github.com/nirajdevpandey/passage-retrieval-chatbot/blob/master/data/images/1_8XpbsR4HdAHBXy5MgpIyug.png)

Tf means term-frequency while tf–idf means term-frequency times inverse document-frequency. 
Term-frequency refer to the times a particular word x appears in a document. Whereas, inverse document-frequency means that how many time the word x appears in entire corpus of document. 
***
`Note`: While the tf–idf normalization is often very useful, there might be cases where the binary occurrence markers might offer better features. This can be achieved by using the binary parameter of CountVectorizer. In particular, some estimators such as Bernoulli Naive Bayes explicitly model discrete boolean random variables. Also, very short texts are likely to have noisy tf–idf values while the binary occurrence info is more stable.

As usual the best way to adjust the feature extraction parameters is to use a cross-validated grid search, for instance by pipelining the [feature extractor with a classifier](https://scikit-learn.org/stable/auto_examples/model_selection/grid_search_text_feature_extraction.html#sphx-glr-auto-examples-model-selection-grid-search-text-feature-extraction-py)

> Improvement: There are many improvements to information retrieval models nowadays, including [Google's PageRank](http://en.wikipedia.org/wiki/PageRank) model to weight documents based on their importance and a variety of improved statistical models of document topics. 

```diff
- This method was not that helpful for me so I would suggest to use other algorithms instead. 
- Reason being the nature of data I was dealing with.
```
***
