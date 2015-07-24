from lib import helpers

helpers.printCurrentTime("start ./train_dictionary_corpus.py")





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











import os
from gensim import corpora


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
  
  from gensim.utils import simple_preprocess
  
  for review in movie_reviews:
    text = review['reviewText']
    words = simple_preprocess(text)
    
    bow.append(words)
  
  dictionary = corpora.Dictionary(bow)
  dictionary.save(bow_cache_dir+"/"+dictionary_cache_file)
  
  corpus = [dictionary.doc2bow(text) for text in bow]
  corpora.MmCorpus.serialize(bow_cache_dir+"/"+corpus_cache_file, corpus)





helpers.printCurrentTime("end ./train_dictionary_corpus.py")
