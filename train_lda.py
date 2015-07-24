from lib import helpers

helpers.printCurrentTime("start ./train_lda.py")





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










from train_dictionary_corpus import dictionary, corpus

import os
from gensim import models
from gensim.models import ldamodel

lda = None

if os.path.isfile(bow_cache_dir+"/"+lda_cache_file):
  lda = ldamodel.LdaModel.load(bow_cache_dir+"/"+lda_cache_file)
else:
  lda = ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=lda_num_topics)
  lda.save(bow_cache_dir+"/"+lda_cache_file)

corpus_lda = lda[corpus]






helpers.printCurrentTime("end ./train_lda.py")