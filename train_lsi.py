from lib import helpers

helpers.printCurrentTime("start ./train_lsi.py")





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










from train_dictionary_corpus import dictionary
from train_tfidf import corpus_tfidf

import os
from gensim import models

lsi = None

if os.path.isfile(bow_cache_dir+"/"+str(lsi_num_topics)+"_"+lsi_cache_file):
  lsi = models.LsiModel.load(bow_cache_dir+"/"+str(lsi_num_topics)+"_"+lsi_cache_file)
else:
  lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=lsi_num_topics)
  lsi.save(bow_cache_dir+"/"+str(lsi_num_topics)+"_"+lsi_cache_file)

corpus_tfidf_lsi = lsi[corpus_tfidf]






helpers.printCurrentTime("end ./train_lsi.py")