default:
	cat README

run:
	nohup python run_evaluation.py &

train-all: train-doc2vec train-tfidf-lsi-lda-hdp

train-doc2vec:
	python train_doc2vec.py

train-tfidf-lsi-lda-hdp:
	python train_tfidf_lsi_lda_hdp.py