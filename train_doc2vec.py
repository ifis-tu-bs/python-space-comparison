import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


from pyhocon import ConfigFactory

config = ConfigFactory.parse_file('./application.conf')
amazon_dump_dir = config.get_string('amazon-dump.dir')
movie_reviews_file = config.get_string('amazon-dump.files.reviews.movies')
movie_meta_data_file = config.get_string('amazon-dump.files.meta-data.movies')

model_size = config.get_int('amazon-dump.doc2vec.model.training.size')
model_window = config.get_int('amazon-dump.doc2vec.model.training.window')
model_min_count = config.get_int('amazon-dump.doc2vec.model.training.min_count')
model_workers = config.get_int('amazon-dump.doc2vec.model.training.workers')
model_cache_dir = config.get_string('amazon-dump.doc2vec.model.cache_dir')


from gensim.models import doc2vec

import gzip
import nltk
import os

model_name = "doc2vec-size_"+str(model_size)+"-window_"+str(model_window)+"-min_count_"+str(model_min_count)+"-workers_"+str(model_workers)+".sav"
model_path = model_cache_dir+"/"+model_name


doc2vec_model = None

if os.path.isfile(model_path):
  doc2vec_model = doc2vec.Doc2Vec.load(model_path)
  print("Model "+model_name+" already trained!")
else:

  def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
      yield eval(l)
  
  
  movie_reviews = parse(amazon_dump_dir+"/"+movie_reviews_file)
  #movie_meta_data = parse(amazon_dump_dir+"/"+movie_meta_data_file)
  
  
  train_data = []
  #for i in range(10):
    #review = movie_reviews.next()
  for review in movie_reviews:
    asin = review['asin']
    label = 'ASIN_'+asin
    
    text = review['reviewText']
    words = nltk.word_tokenize(text)
    
    sent = doc2vec.LabeledSentence(words=words, labels=[label])
    
    train_data.append(sent)
  
  
  doc2vec_model = doc2vec.Doc2Vec(train_data, size=model_size, window=model_window, min_count=model_min_count, workers=model_workers)
  doc2vec_model.save(model_path)
  print("Finished training model "+model_name+"!")


# doc2vec_model