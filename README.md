### passage-retrieval-chatbot

Input a text file separated with many paragraphs and ask a question to get relevant passage back based on TF-IDF wights 

#### Following s the details & workflow of the repository 
In a large text corpus, some words will be very present (e.g. `the`, `a`, `is` in English) hence carrying very little meaningful information about the actual contents of the document. If we were to feed the direct count data directly to a classifier those very frequent terms would shadow the frequencies of rarer yet more interesting terms.

In order to re-weight the count features into floating point values suitable for usage by a classifier it is very common to use the tf–idf transform.

Tf means term-frequency while tf–idf means term-frequency times inverse document-frequency. 
Term-frequency refer to the times a particular word x appears in a document. Whereas, inverse document-frequency means that how many time the word x appears in entire corpus of document. 

`Note`: While the tf–idf normalization is often very useful, there might be cases where the binary occurrence markers might offer better features. This can be achieved by using the binary parameter of CountVectorizer. In particular, some estimators such as Bernoulli Naive Bayes explicitly model discrete boolean random variables. Also, very short texts are likely to have noisy tf–idf values while the binary occurrence info is more stable.

As usual the best way to adjust the feature extraction parameters is to use a cross-validated grid search, for instance by pipelining the feature extractor with a classifier https://scikit-learn.org/stable/auto_examples/model_selection/grid_search_text_feature_extraction.html#sphx-glr-auto-examples-model-selection-grid-search-text-feature-extraction-py

