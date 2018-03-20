'''
Title: Topic Detector (Linc Global Inc.)
Author: Jingyang Li (ljygeek@gmail.com)
'''

from gensim.models import LdaModel, TfidfModel, LsiModel
from gensim import corpora


def create_data(corpus_path):
    '''
    Create data by reading files and preparing document-term matrix
    :param corpus_path: file path
    :return: sentence_dict, dictionary, corpus, corpus_tfidf
    '''
    # initialize variables
    sentences = []
    sentence_dict = {}
    count = 0
    # read lines and create dictionary based on line and corresponding index
    for line in open(corpus_path):
        line = line.strip().split('\t')
        if len(line) == 2:
            sentence_dict[count] = line[1]
            count += 1
            sentences.append(line[1].split(' '))
        else:
            break
    # get corpus dictionary
    dictionary = corpora.Dictionary(sentences)
    # Converting corpus into Document Term Matrix using dictionary prepared above
    corpus = [dictionary.doc2bow(text) for text in sentences]
    # apply TF-IDF model on corpus
    tfidf = TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    return sentence_dict, dictionary, corpus, corpus_tfidf


def lda_model(sentence_dict, dictionary, corpus, corpus_tfidf, cluster_keyword_lda):
    '''
    Obtain topic distribution by using LDA model
    :param sentence_dict: sentence dictionary
    :param dictionary: corpus dictionary
    :param corpus: corpus in document term matrix
    :param corpus_tfidf: TF-IDF model of corpus
    :param cluster_keyword_lda: LDA method name
    :return: None
    '''
    # initialize LDA model
    lda = LdaModel(corpus=corpus_tfidf, id2word=dictionary, num_topics=10)
    f_keyword = open(cluster_keyword_lda, 'w+')
    for topic in lda.print_topics(num_topics=10, num_words=50):
        words = []
        for word in topic[1].split('+'):
            word = word.split('*')[1].replace(' ', '')
            words.append(word)
        f_keyword.write(str(topic[0]) + '\t' + ','.join(words) + '\n')


def lsi_model(sentence_dict, dictionary, corpus, corpus_tfidf, cluster_keyword_lsi):
    '''
    Obtain topic distribution by using LSI model
    :param sentence_dict: sentence dictionary
    :param dictionary: corpus dictionary
    :param corpus: corpus in document term matrix
    :param corpus_tfidf: TF-IDF model of corpus
    :param cluster_keyword_lsi: LSI method name
    :return: None
    '''
    lsi = LsiModel(corpus=corpus_tfidf, id2word=dictionary, num_topics=10)
    f_keyword = open(cluster_keyword_lsi, 'w+')
    for topic in lsi.print_topics(num_topics=10, num_words=50):
        words = []
        for word in topic[1].split('+'):
            word = word.split('*')[1].replace(' ', '')
            words.append(word)
        f_keyword.write(str(topic[0]) + '\t' + ','.join(words) + '\n')


if __name__ == "__main__":
    corpus_path = "./corpus_train_test.tsv"
    cluster_keyword_lda = './topics_lda.tsv'
    cluster_keyword_lsi = './topics_lsi.tsv'
    sentence_dict, dictionary, corpus, corpus_tfidf = create_data(corpus_path)
    lsi_model(sentence_dict, dictionary, corpus, corpus_tfidf, cluster_keyword_lsi)
    lda_model(sentence_dict, dictionary, corpus, corpus_tfidf, cluster_keyword_lda)