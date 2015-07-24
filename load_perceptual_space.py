from lib import helpers

helpers.printCurrentTime("start ./train_perceptual_space.py")





import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# CONFIG

from pyhocon import ConfigFactory

config = ConfigFactory.parse_file('./application.conf')
perceptual_space_dir = config.get_string('perceptual-space.file')

content = []

with open(perceptual_space_dir, 'r') as f:
  content = f.readlines()



perceptual_space = {}
amazon_ids = []
amazon_titles = []

for line in content[1:]: # skip first line of content which are the labels
  entries = line.split(';')

  amazon_id = entries[0].replace('"', '')
  amazon_ids.append(amazon_id)
  
  amazon_title = entries[1].replace('"', '')
  amazon_titles.append(amazon_title)
  
  vector = map(lambda e: float(e), entries[2:])
  
  sparse_vector = []
  for index in range(0, len(vector)):
    sparse_vector.append((index, vector[index]))
  
  perceptual_space[amazon_id] = sparse_vector




# test cosine distance
#aid1 = "0767810880"
#aid2 = "B0000E32V3"
#
#vec1 = vector_by_amazon_id[aid1]
#vec2 = vector_by_amazon_id[aid2]
#
#from gensim.matutils import cossim
#
#sim = cossim(vec1, vec2)
#
#print(sim)





helpers.printCurrentTime("end ./train_perceptual_space.py")