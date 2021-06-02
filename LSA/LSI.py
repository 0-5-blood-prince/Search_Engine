from sentenceSegmentation import SentenceSegmentation
from tokenization import Tokenization
from inflectionReduction import InflectionReduction
from stopwordRemoval import StopwordRemoval
from informationRetrieval import InformationRetrieval
from evaluation import Evaluation

from sys import version_info
import argparse
import json
import matplotlib.pyplot as plt

# Input compatibility for Python 2 and Python 3
if version_info.major == 3:
    pass
elif version_info.major == 2:
    try:
        input = raw_input
    except NameError:
        pass
else:
    print ("Unknown python version - input function not safe")


class SearchEngine:

	def __init__(self, args):
		self.args = args

		self.tokenizer = Tokenization()
		self.sentenceSegmenter = SentenceSegmentation()
		self.inflectionReducer = InflectionReduction()
		self.stopwordRemover = StopwordRemoval()
		self.informationRetriever = InformationRetrieval()
		self.evaluator = Evaluation()


	def segmentSentences(self, text):
		"""
		Call the required sentence segmenter
		"""
		if self.args.segmenter == "naive":
			return self.sentenceSegmenter.naive(text)
		elif self.args.segmenter == "punkt":
			return self.sentenceSegmenter.punkt(text)

	def tokenize(self, text):
		"""
		Call the required tokenizer
		"""
		if self.args.tokenizer == "naive":
			return self.tokenizer.naive(text)
		elif self.args.tokenizer == "ptb":
			return self.tokenizer.pennTreeBank(text)

	def reduceInflection(self, text):
		"""
		Call the required stemmer/lemmatizer
		"""
		return self.inflectionReducer.reduce(text)

	def removeStopwords(self, text):
		"""
		Call the required stopword remover
		"""
		return self.stopwordRemover.fromList(text)


	def preprocessQueries(self, queries):
		"""
		Preprocess the queries - segment, tokenize, stem/lemmatize and remove stopwords
		"""

		# Segment queries
		segmentedQueries = []
		for query in queries:
			segmentedQuery = self.segmentSentences(query)
			segmentedQueries.append(segmentedQuery)
		json.dump(segmentedQueries, open(self.args.out_folder + "segmented_queries.txt", 'w'))
		# Tokenize queries
		tokenizedQueries = []
		for query in segmentedQueries:
			tokenizedQuery = self.tokenize(query)
			tokenizedQueries.append(tokenizedQuery)
		json.dump(tokenizedQueries, open(self.args.out_folder + "tokenized_queries.txt", 'w'))
		# Stem/Lemmatize queries
		reducedQueries = []
		for query in tokenizedQueries:
			reducedQuery = self.reduceInflection(query)
			reducedQueries.append(reducedQuery)
		json.dump(reducedQueries, open(self.args.out_folder + "reduced_queries.txt", 'w'))
		# Remove stopwords from queries
		stopwordRemovedQueries = []
		for query in reducedQueries:
			stopwordRemovedQuery = self.removeStopwords(query)
			stopwordRemovedQueries.append(stopwordRemovedQuery)
		json.dump(stopwordRemovedQueries, open(self.args.out_folder + "stopword_removed_queries.txt", 'w'))

		preprocessedQueries = stopwordRemovedQueries
		return preprocessedQueries

	def preprocessDocs(self, docs):
		"""
		Preprocess the documents
		"""
		
		# Segment docs
		segmentedDocs = []
		for doc in docs:
			segmentedDoc = self.segmentSentences(doc)
			segmentedDocs.append(segmentedDoc)
		json.dump(segmentedDocs, open(self.args.out_folder + "segmented_docs.txt", 'w'))
		# Tokenize docs
		tokenizedDocs = []
		for doc in segmentedDocs:
			tokenizedDoc = self.tokenize(doc)
			tokenizedDocs.append(tokenizedDoc)
		json.dump(tokenizedDocs, open(self.args.out_folder + "tokenized_docs.txt", 'w'))
		# Stem/Lemmatize docs
		reducedDocs = []
		for doc in tokenizedDocs:
			reducedDoc = self.reduceInflection(doc)
			reducedDocs.append(reducedDoc)
		json.dump(reducedDocs, open(self.args.out_folder + "reduced_docs.txt", 'w'))
		# Remove stopwords from docs
		stopwordRemovedDocs = []
		for doc in reducedDocs:
			stopwordRemovedDoc = self.removeStopwords(doc)
			stopwordRemovedDocs.append(stopwordRemovedDoc)
		json.dump(stopwordRemovedDocs, open(self.args.out_folder + "stopword_removed_docs.txt", 'w'))

		preprocessedDocs = stopwordRemovedDocs
		return preprocessedDocs
	def plotter(self, precisions, recalls, fscores, MAPs, nDCGs, method):
				plt.figure()
				plt.plot(range(1, self.num_retrieved+1), precisions, label="Precision")
				plt.plot(range(1, self.num_retrieved+1), recalls, label="Recall")
				plt.plot(range(1, self.num_retrieved+1), fscores, label="F-Score")
				plt.plot(range(1, self.num_retrieved+1), MAPs, label="MAP")
				plt.plot(range(1, self.num_retrieved+1), nDCGs, label="nDCG")
				plt.legend()
				plt.title("Evaluation Metrics - Cranfield Dataset")
				plt.xlabel("rank")
				plt.savefig(args.out_folder + "eval_plot_"+ method + ".png")
	def lsi_evaluate(self, processedDocs, processedQueries, doc_ids,query_ids ):
				self.informationRetriever.buildIndex_lsi(processedDocs, doc_ids, self.dim_red)
				self.informationRetriever.svd_lsi(doc_ids)
				doc_IDs_ordered = self.informationRetriever.rank_lsi(processedQueries)
				qrels = json.load(open(args.dataset + "cran_qrels.json", 'r'))[:]
				precisions, recalls, fscores, MAPs, nDCGs = [], [], [], [], []
				query_lsi_precisions = []
				query_lsi_recalls = []
				for k in range(1, self.num_retrieved+1):
					precision, qprecision = self.evaluator.meanPrecision(
						doc_IDs_ordered, query_ids, qrels, k)
					precisions.append(precision)
					query_lsi_precisions.append(qprecision)
					recall, qrecall = self.evaluator.meanRecall(
						doc_IDs_ordered, query_ids, qrels, k)
					recalls.append(recall)
					query_lsi_recalls.append(qrecall)
					fscore = self.evaluator.meanFscore(
						doc_IDs_ordered, query_ids, qrels, k)
					fscores.append(fscore)
					print("Precision, Recall and F-score @ " +  
						str(k) + " : " + str(precision) + ", " + str(recall) + 
						", " + str(fscore))
					MAP = self.evaluator.meanAveragePrecision(
						doc_IDs_ordered, query_ids, qrels, k)
					MAPs.append(MAP)
					nDCG = self.evaluator.meanNDCG(
						doc_IDs_ordered, query_ids, qrels, k)
					nDCGs.append(nDCG)
					print("MAP, nDCG @ " +  
						str(k) + " : " + str(MAP) + ", " + str(nDCG))
				lsi_y , lsi_x = self.evaluator.precision_recall(query_lsi_precisions,query_lsi_recalls,query_ids)
				# Plot the metrics and save plot 
				# plt.figure()
				# plt.plot(lsi_x,lsi_y)
				# plt.title(" Precision Recall Curve")
				# plt.savefig(args.out_folder + "precision_recall_curve_lsi.png")
				self.plotter(precisions,recalls,fscores,MAPs,nDCGs,'lsi'+str(self.dim_red))
				return lsi_x, lsi_y , MAPs, recalls

	def basic_evaluate(self, processedDocs, processedQueries, doc_ids,query_ids ):
				self.informationRetriever.buildIndex_basic(processedDocs, doc_ids)
				doc_IDs_ordered = self.informationRetriever.rank_basic(processedQueries)
				qrels = json.load(open(args.dataset + "cran_qrels.json", 'r'))[:]
				precisions, recalls, fscores, MAPs, nDCGs = [], [], [], [], []
				query_basic_precisions = []
				query_basic_recalls = []
				for k in range(1, self.num_retrieved+1):
					precision, qprecision = self.evaluator.meanPrecision(
						doc_IDs_ordered, query_ids, qrels, k)
					precisions.append(precision)
					query_basic_precisions.append(qprecision)
					recall, qrecall = self.evaluator.meanRecall(
						doc_IDs_ordered, query_ids, qrels, k)
					recalls.append(recall)
					query_basic_recalls.append(qrecall)
					fscore = self.evaluator.meanFscore(
						doc_IDs_ordered, query_ids, qrels, k)
					fscores.append(fscore)
					print("Precision, Recall and F-score @ " +  
						str(k) + " : " + str(precision) + ", " + str(recall) + 
						", " + str(fscore))
					MAP = self.evaluator.meanAveragePrecision(
						doc_IDs_ordered, query_ids, qrels, k)
					MAPs.append(MAP)
					nDCG = self.evaluator.meanNDCG(
						doc_IDs_ordered, query_ids, qrels, k)
					nDCGs.append(nDCG)
					print("MAP, nDCG @ " +  
						str(k) + " : " + str(MAP) + ", " + str(nDCG))
				basic_y , basic_x = self.evaluator.precision_recall(query_basic_precisions,query_basic_recalls,query_ids)
				# Plot the metrics and save plot 
				# plt.figure()
				# plt.plot(basic_x,basic_y)
				# plt.savefig(args.out_folder + "precision_recall_curve_basic.png")
				self.plotter(precisions,recalls,fscores,MAPs,nDCGs,'basic')
				return basic_x, basic_y, MAPs
	def output_plot(self, doc_IDs_ordered, query_ids, qrels, method):
				precisions, recalls, fscores, MAPs, nDCGs = [], [], [], [], []
				query_precisions = []
				query_recalls = []
				for k in range(1, self.num_retrieved+1):
					precision, qprecision = self.evaluator.meanPrecision(
						doc_IDs_ordered, query_ids, qrels, k)
					precisions.append(precision)
					query_precisions.append(qprecision)
					recall, qrecall = self.evaluator.meanRecall(
						doc_IDs_ordered, query_ids, qrels, k)
					recalls.append(recall)
					query_recalls.append(qrecall)
					fscore = self.evaluator.meanFscore(
						doc_IDs_ordered, query_ids, qrels, k)
					fscores.append(fscore)
					print("Precision, Recall and F-score @ " +  
						str(k) + " : " + str(precision) + ", " + str(recall) + 
						", " + str(fscore))
					MAP = self.evaluator.meanAveragePrecision(
						doc_IDs_ordered, query_ids, qrels, k)
					MAPs.append(MAP)
					nDCG = self.evaluator.meanNDCG(
						doc_IDs_ordered, query_ids, qrels, k)
					nDCGs.append(nDCG)
					print("MAP, nDCG @ " +  
						str(k) + " : " + str(MAP) + ", " + str(nDCG))
				if method != "":
					self.plotter(precisions,recalls,fscores,MAPs,nDCGs,method)
				lsi_y , lsi_x = self.evaluator.precision_recall(query_precisions,query_recalls,query_ids)
				return lsi_x, lsi_y, MAPs
	def evaluateDataset(self):
		"""
		- preprocesses the queries and documents, stores in output folder
		- invokes the IR system
		- evaluates precision, recall, fscore, nDCG and MAP 
		  for all queries in the Cranfield dataset
		- produces graphs of the evaluation metrics in the output folder
		"""

		# Read queries
		queries_json = json.load(open(args.dataset + "cran_queries.json", 'r'))[:]
		query_ids, queries = [item["query number"] for item in queries_json], \
								[item["query"] for item in queries_json]
		# Process queries 
		processedQueries = self.preprocessQueries(queries)

		# Read documents
		docs_json = json.load(open(args.dataset + "cran_docs.json", 'r'))[:]
		doc_ids, docs = [item["id"] for item in docs_json], \
								[item["body"] for item in docs_json]
		# Process documents
		processedDocs = self.preprocessDocs(docs)

		self.dim_red = int(self.args.k)
		experiment = int(self.args.experiment)
		method = self.args.method
		# Build document index
		self.num_retrieved = int(self.args.num_retrieved)
		if experiment == 0:
			if method == 'hyp1':
				basic_x, basic_y, bMAPs = self.basic_evaluate(processedDocs,processedQueries,doc_ids,query_ids)
				lsi_x, lsi_y, lsiMAPs, lsirecalls = self.lsi_evaluate(processedDocs,processedQueries,doc_ids,query_ids)
				
				plt.figure()
				plt.plot(lsi_x,lsi_y,label = 'lsi_'+str(self.dim_red))
				plt.plot(basic_x, basic_y, label = 'basic')
				plt.title(" Precision Recall Curve for basic and LSI")
				plt.legend()
				plt.savefig(args.out_folder + "precision_recall_curve_basic_lsi.png")
			if method == 'hyp2':
				# divide queries into training and test set
				self.w = 2.0
				len_train = int((0.8)* len(processedQueries))
				len_dev = int((0.1)*len_train)
				len_train = len_train - len_dev
				train_queries = processedQueries[:len_train]
				train_ids = query_ids[:len_train]
				dev_queries = processedQueries[len_train:len_train + len_dev]
				dev_ids = query_ids[len_train:len_train + len_dev]
				test_queries = processedQueries[len_train+ len_dev:]
				test_ids = query_ids[len_train + len_dev:]
				self.informationRetriever.buildIndex_lsi(processedDocs, doc_ids, self.dim_red)
				self.informationRetriever.svd_lsi(doc_ids)
				qrels = json.load(open(args.dataset + "cran_qrels.json", 'r'))[:]
				train_qrels = []
				test_qrels = []
				for e in qrels:
					if int(e["query_num"]) in train_ids:
						train_qrels.append(e)
					else:
						test_qrels.append(e)
				# print(train_qrels)
				# return 
				X , doc_IDs_ordered_lsi, doc_IDs_ordered_sup = self.informationRetriever.train(train_queries, train_ids, train_qrels, self.w)
				lsi_x_train, lsi_y_train, lsi_train_MAPs = self.output_plot(doc_IDs_ordered_lsi, train_ids, train_qrels,"")
				sup_x_train, sup_y_train, sup_train_MAPs= self.output_plot(doc_IDs_ordered_sup, train_ids, train_qrels,"")

				plt.figure()
				plt.plot(lsi_x_train,lsi_y_train,label = 'lsi'+str(self.dim_red))
				plt.plot(sup_x_train,sup_y_train,label = 'sup'+str(self.dim_red))
				plt.title(" Precision Recall Curve for LSI and supervised training data")
				plt.legend()
				plt.savefig(args.out_folder + "precision_recall_curve_lsi_sup_train.png")


				doc_IDs_ordered_lsi, doc_IDs_ordered_sup = self.informationRetriever.test(test_queries, test_ids, test_qrels, X)
				lsi_x_test, lsi_y_test, lsi_test_MAPs = self.output_plot(doc_IDs_ordered_lsi, test_ids, test_qrels,"test_lsi")
				sup_x_test, sup_y_test, sup_test_MAPs = self.output_plot(doc_IDs_ordered_sup, test_ids, test_qrels,"test_sup")

				plt.figure()
				plt.plot(lsi_x_test,lsi_y_test,label = 'lsi'+str(self.dim_red))
				plt.plot(sup_x_test,sup_y_test,label = 'sup'+str(self.dim_red))
				plt.title(" Precision Recall Curve for LSI and supervised testing data")
				plt.legend()
				plt.savefig(args.out_folder + "precision_recall_curve_lsi_sup_test.png")

			if method == 'lsi':
				self.lsi_evaluate(processedDocs,processedQueries,doc_ids,query_ids)
			if method == 'basic':
				self.basic_evaluate(processedDocs,processedQueries,doc_ids,query_ids)
		if experiment == 1:
			if (method == 'hyp1'):
				dims = [10,20,50,100,150,200,250,300,350,400, 800]
				basic_x, basic_y,bMAPs = self.basic_evaluate(processedDocs,processedQueries,doc_ids,query_ids)
				MAPs = []
				Recalls = []
				T = []
				lsi_X = []
				lsi_Y = []
				for dim in dims:
					self.dim_red = dim
					lsi_x, lsi_y,lsi_MAPs,lsi_recalls = self.lsi_evaluate(processedDocs,processedQueries,doc_ids,query_ids)
					MAPs.append(lsi_MAPs[-1])
					Recalls.append(lsi_recalls[-1])
					T.append(dim)
					lsi_X.append(lsi_x)
					lsi_Y.append(lsi_y)
				
				plt.figure()
				for i in range(len(dims)):
					if dims[i] in [50, 200, 400]:
						plt.figure()
						plt.plot(lsi_X[i],lsi_Y[i],label = 'lsi'+str(T[i]))
						plt.plot(basic_x, basic_y, label = 'basic')
						plt.title(" Precision Recall Curve for basic and LSI with "+str(T[i])+" factors")
						plt.legend()
						plt.savefig(args.out_folder + "precision_recall_curve_basic_lsi_"+ str(T[i]) +".png")

				# plt.figure()
				# plt.plot(T, MAPs)
				# plt.title(" Mean Average Precision vs Dimensions")
				# plt.legend()
				# plt.savefig(args.out_folder + "MAP_exp.png")

				# plt.figure()
				# plt.plot(T, Recalls)
				# plt.title(" Average Recall vs Dimensions")
				# plt.legend()
				# plt.savefig(args.out_folder + "Recall_exp.png")
			if method == "hyp2":
				len_train = int((0.8)* len(processedQueries))
				len_dev = int((0.1)*len_train)
				len_train = len_train - len_dev
				train_queries = processedQueries[:len_train]
				train_ids = query_ids[:len_train]
				dev_queries = processedQueries[len_train:len_train + len_dev]
				dev_ids = query_ids[len_train:len_train + len_dev]
				test_queries = processedQueries[len_train+ len_dev:]
				test_ids = query_ids[len_train + len_dev:]
				ws = [0.1 , 0.2, 0.5, 0.8, 1.0 , 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
				train_maps = []
				sup_train_maps = []
				dev_maps = []
				sup_dev_maps = []
				for w in ws:
					self.w = w
					self.informationRetriever.buildIndex_lsi(processedDocs, doc_ids, self.dim_red)
					self.informationRetriever.svd_lsi(doc_ids)
					qrels = json.load(open(args.dataset + "cran_qrels.json", 'r'))[:]
					train_qrels = []
					test_qrels = []
					dev_qrels = []
					for e in qrels:
						if int(e["query_num"]) in train_ids:
							train_qrels.append(e)
						elif int(e["query_num"]) in dev_ids:
							dev_qrels.append(e)
						else:
							test_qrels.append(e)
					# print(train_qrels)
					# return 
					X , doc_IDs_ordered_lsi, doc_IDs_ordered_sup = self.informationRetriever.train(train_queries, train_ids, train_qrels, self.w)
					lsi_x_train, lsi_y_train, lsi_train_MAPs = self.output_plot(doc_IDs_ordered_lsi, train_ids, train_qrels,"")
					train_maps.append(lsi_train_MAPs[-1])
					sup_x_train, sup_y_train, sup_train_MAPs= self.output_plot(doc_IDs_ordered_sup, train_ids, train_qrels,"")
					sup_train_maps.append(sup_train_MAPs[-1])
					plt.figure()
					plt.plot(lsi_x_train,lsi_y_train,label = 'lsi_'+str(self.dim_red))
					plt.plot(sup_x_train,sup_y_train,label = 'sup_'+str(w))
					plt.title(" Precision Recall Curve for LSI and supervised training data w = "+str(w))
					plt.legend()
					plt.savefig(args.out_folder + "precision_recall_curve_lsi_sup_train w = "+str(w) + ".png")


					doc_IDs_ordered_lsi, doc_IDs_ordered_sup = self.informationRetriever.test(dev_queries, dev_ids, dev_qrels, X)
					lsi_x_dev, lsi_y_dev, lsi_dev_MAPs = self.output_plot(doc_IDs_ordered_lsi, dev_ids, dev_qrels,"")
					dev_maps.append(lsi_dev_MAPs[-1])
					sup_x_dev, sup_y_dev, sup_dev_MAPs = self.output_plot(doc_IDs_ordered_sup, dev_ids, dev_qrels,"dev_sup_w"+str(w))
					sup_dev_maps.append(sup_dev_MAPs[-1])
					plt.figure()
					plt.plot(lsi_x_dev,lsi_y_dev,label = 'lsi_'+str(self.dim_red))
					plt.plot(sup_x_dev,sup_y_dev,label = 'sup_'+str(w) )
					plt.title(" Precision Recall Curve for LSI and supervised Dev data w = "+str(w))
					plt.legend()
					plt.savefig(args.out_folder + "precision_recall_curve_lsi_sup_dev w = "+str(w) + ".png")
				plt.figure()
				plt.plot(ws, train_maps, label='lsi')
				plt.plot(ws, sup_train_maps, label='sup')
				plt.title(" Mean Average Precision vs W Training data")
				plt.legend()
				plt.savefig(args.out_folder + "MAP_train_exp.png")

				plt.figure()
				plt.plot(ws, dev_maps, label='lsi')
				plt.plot(ws, sup_dev_maps, label='sup')
				plt.title(" Mean Average Precision vs W Dev data")
				plt.legend()
				plt.savefig(args.out_folder + "MAP_dev_exp.png")

				


		
	def handleCustomQuery(self):
		"""
		Take a custom query as input and return top five relevant documents
		"""

		#Get query
		print("Enter query below")
		query = input()
		# Process documents
		processedQuery = self.preprocessQueries([query])[0]

		# Read documents
		docs_json = json.load(open(args.dataset + "cran_docs.json", 'r'))[:]
		doc_ids, docs = [item["id"] for item in docs_json], \
							[item["body"] for item in docs_json]
		# Process documents
		processedDocs = self.preprocessDocs(docs)

		# Build document index
		self.informationRetriever.buildIndex(processedDocs, doc_ids)
		# Rank the documents for the query
		doc_IDs_ordered = self.informationRetriever.rank([processedQuery])[0]

		# Print the IDs of first five documents
		print("\nTop five document IDs : ")
		for id_ in doc_IDs_ordered[:5]:
			print(id_)



if __name__ == "__main__":

	# Create an argument parser
	parser = argparse.ArgumentParser(description='main.py')

	# Tunable parameters as external arguments
	parser.add_argument('-dataset', default = "cranfield/", 
						help = "Path to the dataset folder")
	parser.add_argument('-out_folder', default = "output/", 
						help = "Path to output folder")
	parser.add_argument('-segmenter', default = "punkt",
	                    help = "Sentence Segmenter Type [naive|punkt]")
	parser.add_argument('-tokenizer',  default = "ptb",
	                    help = "Tokenizer Type [naive|ptb]")
	parser.add_argument('-k', default = '200',
						help = "k value LSA hidden feature size")
	parser.add_argument('-custom', action = "store_true", 
						help = "Take custom query as input ")
	parser.add_argument('-num_retrieved', default = '10', help = 'num docs retrieved')
	parser.add_argument('-experiment', default = '0', help = 'Whether to run experiments on k')
	parser.add_argument('-preprocess', default = 'Yes' , help = 'whether to preprocess docs and queries again' )
	parser.add_argument('-method', default = 'all', help = 'all methods will run by default unless the follwing are mentioned "lsi"  "basic"')
	# Parse the input arguments
	args = parser.parse_args()

	# Create an instance of the Search Engine
	searchEngine = SearchEngine(args)

	# Either handle query from user or evaluate on the complete dataset 
	if args.custom:
		searchEngine.handleCustomQuery()
	else:
		searchEngine.evaluateDataset()
