from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# Create a list to use for clustering - this is for topic modelling
doclist = []

# Create doclist for use in topic modelling
doclist.append(parsed)


# Cluster documents and demonstrate prediction
# K means clustering using the following parameters:
# - filter out english stopwords
# - word must appear in no more than 80% of documents
# - word must appear in no less than 20% of documents
# - token and stem words, allow up to 3 words together as a grouping
# TODO - calculate ideal k value
def clustering(documents):
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.8, min_df=0.2, use_idf=True,
                                 tokenizer=tokenize_and_stem, ngram_range=(1, 3))
    X = vectorizer.fit_transform(doclist)

    true_k = 5
    model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
    model.fit(X)

    print("Top terms per cluster:")
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    for i in range(true_k):
        print("Cluster %d:" % i),
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind]),
        print

    print("\n")
    print("Prediction")

    Y = vectorizer.transform(["this is a document about islamic state "
                              "and terrorists and bombs IS jihad terrorism isil"])
    prediction = model.predict(Y)
    print("A document with 'bad' terms would be in:")
    print(prediction)

    Y = vectorizer.transform(["completely innocent text just about kittens and puppies"])
    prediction = model.predict(Y)
    print("A document with 'good' terms would be in:")
    print(prediction)

