from lib import helpers

helpers.printCurrentTime("start ./run_evaluation.py")





from load_perceptual_space import perceptual_space, amazon_ids
from train_doc2vec import doc2vec_model, doc2vec_labels
#from train_doc2vec_concat import doc2vec_model_concat
from train_dictionary_corpus import dictionary, corpus
from train_tfidf import tfidf, corpus_tfidf
from train_lsi import lsi, corpus_tfidf_lsi
#from train_lda import lda, corpus_lda
from amazon_statistics import number_of_reviews_by_asin, bow_by_asin





# Initialize all the models I need

perceptual_space
print("perceptual_space")

doc2vec_model
print("doc2vec_model")

tfidf
print("tfidf")

corpus_tfidf
print("corpus_tfidf")

lsi
print("lsi")

corpus_tfidf_lsi
print("corpus_tfidf_lsi")

#lda
#print("lda")

#corpus_lda
#print("corpus_lda")




# Take some amazon ids
ids = amazon_ids#[:100]





# Similarities
perceptual_space_sims = []
doc2vec_sims = []
#doc2vec_concat_sims = []
lsi_sims = []
tfidf_sims = []
#lda_sims = []

for index1 in range(0, len(ids)-1):
  for index2 in range(index1+1, len(ids)):
    aid1 = ids[index1]
    aid2 = ids[index2]
    
    if aid1 in number_of_reviews_by_asin and aid2 in number_of_reviews_by_asin:
      if number_of_reviews_by_asin[aid1] >= 20 and number_of_reviews_by_asin[aid2] >= 20:
        
        
        # perceptual space
        vec1 = perceptual_space[aid1]
        vec2 = perceptual_space[aid2]
        
        from gensim.matutils import cossim
        
        perceptual_space_sim = cossim(vec1, vec2)
        
        
        # doc2vec
        doc2vec_sim = "unknown"
        try:
          doc2vec_sim = doc2vec_model.similarity("ASIN_"+aid1, "ASIN_"+aid2)
        except KeyError, e:
          doc2vec_sim = "unknown"
        
        
        # doc2vec concat
        #doc2vec_concat_sim = "unknown"
        #try:
        #  doc2vec_concat_sim = doc2vec_model_concat.similarity("ASIN_"+aid1, "ASIN_"+aid2)
        #except KeyError, e:
        #  doc2vec_concat_sim = "unknown"
        
        
        
        # lsi
        vec1_lsi = "unknown"
        vec2_lsi = "unknown"
        try:
          vec1_lsi = dictionary.doc2bow(bow_by_asin[aid1]) # transform words to numbers
          vec2_lsi = dictionary.doc2bow(bow_by_asin[aid2]) # transform words to numbers
          
          tfidf1 = tfidf[vec1_lsi] # transform number vecs to tfidf vecs
          tfidf2 = tfidf[vec2_lsi] # transform number vecs to tfidf vecs
        
          lsi1 = lsi[tfidf1]
          lsi2 = lsi[tfidf2]
          
          lsi_sim = cossim(lsi1, lsi2)
        except KeyError, e:
          vec1_lsi = "unknown"
          vec2_lsi = "unknown"
        
        
        
        # tfidf
        vec1_tfidf = "unknown"
        vec2_tfidf = "unknown"
        try:
          vec1_tfidf = dictionary.doc2bow(bow_by_asin[aid1]) # transform words to numbers
          vec2_tfidf = dictionary.doc2bow(bow_by_asin[aid2]) # transform words to numbers
          
          tfidf1 = tfidf[vec1_lsi] # transform number vecs to tfidf vecs
          tfidf2 = tfidf[vec2_lsi] # transform number vecs to tfidf vecs
          
          tfidf_sim = cossim(tfidf1, tfidf2)
        except KeyError, e:
          vec1_tfidf = "unknown"
          vec2_tfidf = "unknown"
          
          
          
        
        # lda
        # vec1_lda = "unknown"
        # vec2_lda = "unknown"
        # try:
        #   vec1_lda = dictionary.doc2bow(bow_by_asin[aid1]) # transform words to numbers
        #   vec2_lda = dictionary.doc2bow(bow_by_asin[aid2]) # transform words to numbers
        #   
        #   #tfidf1 = tfidf[vec1_lsi] # transform number vecs to tfidf vecs
        #   #tfidf2 = tfidf[vec2_lsi] # transform number vecs to tfidf vecs
        # 
        #   lda1 = lda[vec1_lda]
        #   lda2 = lda[vec2_lda]
        #   
        #   lda_sim = cossim(lda1, lda2)
        # except KeyError, e:
        #   vec1_lda = "unknown"
        #   vec2_lda = "unknown"
        
        
        
        # assign
        #if doc2vec_sim != "unknown" and doc2vec_concat_sim != "unknown" and vec1_lsi != "unknown" and vec2_lsi != "unknown":
        #if doc2vec_sim != "unknown" and vec1_lsi != "unknown" and vec1_tfidf != "unknown" and vec2_tfidf != "unknown" and vec2_lsi != "unknown" and vec1_lda != "unknown" and vec2_lda != "unknown":
        if doc2vec_sim != "unknown" and vec1_lsi != "unknown" and vec1_tfidf != "unknown" and vec2_tfidf != "unknown" and vec2_lsi != "unknown":
          perceptual_space_sims.append(perceptual_space_sim)
          doc2vec_sims.append(doc2vec_sim)
          #doc2vec_concat_sims.append(doc2vec_concat_sim)
          lsi_sims.append(lsi_sim)
          tfidf_sims.append(tfidf_sim)
          #lda_sims.append(lda_sim)
          
        
        # output
        # print("\n"+str(aid1)+" <-> "+str(aid2))
        # print('per.sp.: '+str(perceptual_space_sim))
        # print('doc2vec: '+str(doc2vec_sim))





# compute Spearman Correlation
from scipy.stats import spearmanr

corr_perc_doc = spearmanr(perceptual_space_sims, doc2vec_sims)
#corr_perc_doc_conc = spearmanr(perceptual_space_sims, doc2vec_concat_sims)
corr_perc_lsi = spearmanr(perceptual_space_sims, lsi_sims)
corr_perc_tfidf = spearmanr(perceptual_space_sims, tfidf_sims)
#corr_perc_lda = spearmanr(perceptual_space_sims, lda_sims)


print("\nSpearman Correlation perc.sp. and doc2vec:")
print(corr_perc_doc)

#print("\nSpearman Correlation perc.sp. and doc2vec_concat:")
#print(corr_perc_doc_conc)

print("\nSpearman Correlation perc.sp. and tfidf:")
print(corr_perc_tfidf)

print("\nSpearman Correlation perc.sp. and lsi:")
print(corr_perc_lsi)

#print("\nSpearman Correlation perc.sp. and lda:")
#print(corr_perc_lda)



helpers.printCurrentTime("end ./run_evaluation.py")