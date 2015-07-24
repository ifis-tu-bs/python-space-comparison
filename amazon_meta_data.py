from lib import helpers

helpers.printCurrentTime("start ./amazon_meta_data.py")





import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# CONFIG

from pyhocon import ConfigFactory

config = ConfigFactory.parse_file('./application.conf')
amazon_dump_dir = config.get_string('amazon-dump.dir')
movie_reviews_file = config.get_string('amazon-dump.files.reviews.movies')
movie_meta_data_file = config.get_string('amazon-dump.files.meta-data.movies')

text_min_len = config.get_int('amazon-dump.statistics.review.text.min-len')
statistic_measures_cache_dir = config.get_string('amazon-dump.statistics.measures.cache-dir')
reviews_count_file = config.get_string('amazon-dump.statistics.measures.reviews-count.cache-file')
number_of_reviews_by_asin_file = config.get_string('amazon-dump.statistics.measures.number-of-reviews-by-asin.cache-file')
number_of_reviews_by_person_file = config.get_string('amazon-dump.statistics.measures.number-of-reviews-by-person.cache-file')
bow_by_asin_file = config.get_string('amazon-dump.statistics.measures.bow-by-asin.cache-file')







title_by_asin = {}


import gzip

def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield eval(l)

meta_data = parse(amazon_dump_dir+"/"+movie_meta_data_file)

for json in meta_data:
  try:
    asin = json['asin']
    title = json['title']
    title_by_asin[asin] = title
  except KeyError, e:
    asin
  




helpers.printCurrentTime("end ./amazon_meta_data.py")
