'''
Title: Find Top Topics (Linc Global Inc.)
Author: Jingyang Li (ljygeek@gmail.com)
'''

import pandas as pd
import glob
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models

def read_files(filepath):
    '''
    Read all the tsv files
    :param filepath: folder path
    :return: list of conversation words
    '''
    files = glob.glob(filepath + "/*.tsv")
    frame = pd.DataFrame()
    filelist = []
    count = 1

    # this line is only for showing how to replace next part with Dask
    # import dask.dataframe as dd
    # dialogue = dd.read_csv('./4test/*.tsv', header=None, encoding='UTF-8', delimiter='\t')

    for file in files:
        df = pd.read_csv(file, header=None, encoding='utf-8', delimiter='\t')
        filelist.append(df)
        # for debugging, too many read in files will fail because of "low memory"
        print ("File %s done!" % str(count))
        count += 1
    frame = pd.concat(filelist)
    # conversation words are selected
    words = frame[3].tolist()
    return (words)

def topic_modeling(words):
    '''
    Topic modeling using Latent Dichlet Allocation (LDA)
    :param words: list of converstaions
    :return: None
    '''
    tokenizer = RegexpTokenizer(r'\w+')
    stop_words = set(stopwords.words('english'))
    p_stemmer = PorterStemmer()
    texts = []
    for w in words:
        raw = w.lower()
        # tokenize
        tokens = tokenizer.tokenize(raw)
        # remove stop tokens
        stopped_tokens = [w for w in tokens if not w in stop_words]
        # find stemming words
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
        texts.append(stemmed_tokens)
    # create corpus dictionary
    dictionary = corpora.Dictionary(texts)
    # Converting corpus into Document Term Matrix using dictionary prepared above
    corpus = [dictionary.doc2bow(text) for text in texts]
    # apply LDA model
    ldamodel = models.ldamodel.LdaModel(corpus, num_topics=10, id2word=dictionary, passes=20)
    print(ldamodel.print_topics(num_topics=10, num_words=2))

if __name__ == "__main__":
    filepath = str(input("Please enter the folder directory: "))
    words = read_files(filepath)
    # download the "stopwords" file from nltk
    nltk.download('stopwords')
    topic_modeling(words)