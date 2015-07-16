from lib import helpers

helpers.printCurrentTime("start ./train_tfid.py")

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# CONFIG

from pyhocon import ConfigFactory

config = ConfigFactory.parse_file('./application.conf')
amazon_dump_dir = config.get_string('amazon-dump.dir')
movie_reviews_file = config.get_string('amazon-dump.files.reviews.movies')
movie_meta_data_file = config.get_string('amazon-dump.files.meta-data.movies')

bow_cache_dir = config.get_string('amazon-dump.bow.cache-dir')
dictionary_cache_file = config.get_string('amazon-dump.bow.dictionary.cache-file')
corpus_cache_file = config.get_string('amazon-dump.bow.corpus.cache-file')
tfidf_cache_file = config.get_string('amazon-dump.bow.tfidf.cache-file')
lsi_cache_file = config.get_string('amazon-dump.bow.lsi.cache-file')
lsi_num_topics = config.get_int('amazon-dump.bow.lsi.num-topics')
lda_cache_file = config.get_string('amazon-dump.bow.lda.cache-file')
lda_num_topics = config.get_int('amazon-dump.bow.lda.num-topics')
hdp_cache_file = config.get_string('amazon-dump.bow.hdp.cache-file')


# BAG OF WORDS AS DICTIONARY

import os
from gensim import corpora

helpers.printCurrentTime("dictionary and corpus...")

dictionary = None
corpus = None

if os.path.isfile(bow_cache_dir+"/"+dictionary_cache_file) and os.path.isfile(bow_cache_dir+"/"+corpus_cache_file):
  dictionary = corpora.Dictionary.load(bow_cache_dir+"/"+dictionary_cache_file)
  corpus = corpora.MmCorpus(bow_cache_dir+"/"+corpus_cache_file)
else:

  bow = []
  
  import gzip
  
  def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
      yield eval(l)
  
  movie_reviews = parse(amazon_dump_dir+"/"+movie_reviews_file)
  
#  import nltk
#  from nltk.tokenize import RegexpTokenizer
#  from nltk.corpus import stopwords
  from gensim.utils import simple_preprocess
  
#  tokenizer = RegexpTokenizer(r'\w+')
  
  for review in movie_reviews:
  #for i in range(10):
  #  review = movie_reviews.next()
#    label = 'ASIN_'+review['asin']
      
    text = review['reviewText']
#    text = text.translate(None, ''.join(['\'']))                                        # it's => its, haven't => havent
#    words = tokenizer.tokenize(text)                                                    # take just words
#    words = [w.lower() for w in words]                                                  # lowercase all words!
#    filtered_words = [word for word in words if word not in stopwords.words('english')] # remove stopwords
    
    words = simple_preprocess(text)
    
    bow.append(words)
  
  dictionary = corpora.Dictionary(bow)
  dictionary.save(bow_cache_dir+"/"+dictionary_cache_file)
  
  corpus = [dictionary.doc2bow(text) for text in bow]
  corpora.MmCorpus.serialize(bow_cache_dir+"/"+corpus_cache_file, corpus)


helpers.printCurrentTime("dictionary and corpus... DONE!")




# TF-IDF

from gensim import models

helpers.printCurrentTime("tf-idf...")


tfidf = None

if os.path.isfile(bow_cache_dir+"/"+tfidf_cache_file):
  tfidf = models.TfidfModel.load(bow_cache_dir+"/"+tfidf_cache_file)
else:
  tfidf = models.TfidfModel(corpus)
  tfidf.save(bow_cache_dir+"/"+tfidf_cache_file)

corpus_tfidf = tfidf[corpus]


helpers.printCurrentTime("tf-idf... DONE!")




# LSI

helpers.printCurrentTime("lsi...")


lsi = None

if os.path.isfile(bow_cache_dir+"/"+lsi_cache_file):
  lsi = models.LsiModel.load(bow_cache_dir+"/"+lsi_cache_file)
else:
  lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=lsi_num_topics)
  lsi.save(bow_cache_dir+"/"+lsi_cache_file)

corpus_tfidf_lsi = lsi[corpus_tfidf]


helpers.printCurrentTime("lsi... DONE!")



# LDA

helpers.printCurrentTime("lda...")


from gensim.models import ldamodel

lda = None

if os.path.isfile(bow_cache_dir+"/"+lda_cache_file):
  lda = ldamodel.LdaModel.load(bow_cache_dir+"/"+lda_cache_file)
else:
  lda = ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=lda_num_topics)
  lda.save(bow_cache_dir+"/"+lda_cache_file)

corpus_lda = lda[corpus]


helpers.printCurrentTime("lda... DONE!")



# HDP

helpers.printCurrentTime("hdp...")


from gensim.models import hdpmodel

hdp = None

if os.path.isfile(bow_cache_dir+"/"+hdp_cache_file):
  hdp = hdpmodel.HdpModel.load(bow_cache_dir+"/"+hdp_cache_file)
else:
  hdp = hdpmodel.HdpModel(corpus, id2word=dictionary)
  hdp.save(bow_cache_dir+"/"+hdp_cache_file)

corpus_hdp = hdp[corpus]


helpers.printCurrentTime("hdp... DONE!")





helpers.printCurrentTime("end ./train_tfids.py")