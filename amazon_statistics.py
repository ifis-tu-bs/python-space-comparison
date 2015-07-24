from lib import helpers

helpers.printCurrentTime("start ./amazon_statistics.py")





import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# CONFIG

from pyhocon import ConfigFactory

config = ConfigFactory.parse_file('./application.conf')
amazon_dump_dir = config.get_string('amazon-dump.dir')
movie_reviews_file = config.get_string('amazon-dump.files.reviews.movies')

text_min_len = config.get_int('amazon-dump.statistics.review.text.min-len')
statistic_measures_cache_dir = config.get_string('amazon-dump.statistics.measures.cache-dir')
reviews_count_file = config.get_string('amazon-dump.statistics.measures.reviews-count.cache-file')
number_of_reviews_by_asin_file = config.get_string('amazon-dump.statistics.measures.number-of-reviews-by-asin.cache-file')
number_of_reviews_by_person_file = config.get_string('amazon-dump.statistics.measures.number-of-reviews-by-person.cache-file')
bow_by_asin_file = config.get_string('amazon-dump.statistics.measures.bow-by-asin.cache-file')






import pickle
import os


reviews_count = { 'count': 0 }
number_of_reviews_by_asin = {}
number_of_reviews_by_person = {}
bow_by_asin = {}


file1 = statistic_measures_cache_dir+"/"+reviews_count_file
file2 = statistic_measures_cache_dir+"/"+number_of_reviews_by_asin_file
file3 = statistic_measures_cache_dir+"/"+number_of_reviews_by_person_file
file4 = statistic_measures_cache_dir+"/"+bow_by_asin_file

if os.path.isfile(file1) and os.path.isfile(file2) and os.path.isfile(file3) and os.path.isfile(file4):
  reviews_count = pickle.load(open(file1, "rb"))
  number_of_reviews_by_asin = pickle.load(open(file2, "rb"))
  number_of_reviews_by_person = pickle.load(open(file3, "rb"))
  bow_by_asin = pickle.load(open(file4, "rb"))
else:
  
  import gzip

  def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
      yield eval(l)

  movie_reviews = parse(amazon_dump_dir+"/"+movie_reviews_file)
  
  for review in movie_reviews:
    
    # only count reviews with a certain length
    text = review['reviewText']
    if len(text) > text_min_len:
    
      # count reviews
      reviews_count['count'] = reviews_count['count'] + 1
      
      # number of reviews for each asin
      asin = review['asin']
      if asin in number_of_reviews_by_asin:
        number_of_reviews_by_asin[asin] += 1
      else:
        number_of_reviews_by_asin[asin] = 1
      
      # number of reviews by person
      reviewer = review['reviewerID']
      if reviewer in number_of_reviews_by_person:
        number_of_reviews_by_person[reviewer] += 1
      else:
        number_of_reviews_by_person[reviewer] = 1
    
      # BOW for each review
      from gensim.utils import simple_preprocess
      
      words = simple_preprocess(text)
      bow_by_asin[asin] = words
    
    
  pickle.dump(reviews_count, open(file1, 'wb'))
  pickle.dump(number_of_reviews_by_asin, open(file2, 'wb'))
  pickle.dump(number_of_reviews_by_person, open(file3, 'wb'))
  pickle.dump(bow_by_asin, open(file4, 'wb'))
  
  

print("reviews_count")
print(reviews_count)

print("number_of_reviews_by_asin")
print(len(number_of_reviews_by_asin))

print("number_of_reviews_by_person")
print(len(number_of_reviews_by_person))

print("bow_by_asin")
print(len(bow_by_asin))






helpers.printCurrentTime("end ./amazon_statistics.py")
