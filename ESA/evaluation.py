from util import *
# Add your import statements here




class Evaluation():

	def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The precision value as a number between 0 and 1
		"""

		precision = 0.0
		num = 0.0
		for i in range(k):
			if query_doc_IDs_ordered[i] in true_doc_IDs:
				num+=1
		#Fill in code here
		precision = num/k
		return precision


	def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean precision value as a number between 0 and 1
		"""

		meanPrecision = 0.0
		d = {}
		for q in query_ids:
			d[q] = set() 
			# print(q,d[q])
		# print(e["query_num"])
		# print(e["id"])
		# print(d)
		for e in qrels:
			# print(e["query_num"])
			# print(d.get(e["query_num"]))
			# print(e["query_num"],d[e["query_num"]])
			
			if int(e["query_num"]) in d:
				d[int(e["query_num"])].add(int(e["id"]))
		#Fill in code here
		q = []
		for i in range(len(query_ids)):
			prec = self.queryPrecision(doc_IDs_ordered[i],query_ids[i],d[query_ids[i]],k)
			meanPrecision += prec
			q.append(prec)
		meanPrecision /= len(query_ids)
		return meanPrecision , q

	
	def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The recall value as a number between 0 and 1
		"""

		recall = 0.0
		num = 0.0
		for i in range(k):
			if query_doc_IDs_ordered[i] in true_doc_IDs:
				num+=1
		#Fill in code here
		
		num_relev_docs = len(true_doc_IDs)
		recall = num/num_relev_docs
		return recall


	def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean recall value as a number between 0 and 1
		"""

		meanRecall = 0.0
		d = {}
		for q in query_ids:
			d[q] = set() 
		for e in qrels:
			if int(e["query_num"]) in d:
				d[int(e["query_num"])].add(int(e["id"]))
		#Fill in code here
		r = []
		for i in range(len(query_ids)):
			rec = self.queryRecall(doc_IDs_ordered[i],query_ids[i],d[query_ids[i]],k)
			meanRecall += rec
			r.append(rec)
		meanRecall /= len(query_ids)

		return meanRecall, r


	def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The fscore value as a number between 0 and 1
		"""

		fscore = 0.0
		precision = 0.0
		recall = 0.0
		num = 0.0
		for i in range(k):
			if query_doc_IDs_ordered[i] in true_doc_IDs:
				num+=1
		#Fill in code here
		num_relev_docs = len(true_doc_IDs)
		recall = num/num_relev_docs
		precision = num/k
		#Fill in code here
		if precision + recall == 0:
			return 0
		fscore = (2*precision*recall) / (precision + recall)
		return fscore


	def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value
		
		Returns
		-------
		float
			The mean fscore value as a number between 0 and 1
		"""

		meanFscore = 0.0
		d = {}
		for q in query_ids:
			d[q] = set() 
		for e in qrels:
			if int(e["query_num"]) in d:
				d[int(e["query_num"])].add(int(e["id"]))

		#Fill in code here
		for i in range(len(query_ids)):
			meanFscore += self.queryFscore(doc_IDs_ordered[i],query_ids[i],d[query_ids[i]],k)
		meanFscore /= len(query_ids)
		#Fill in code here

		return meanFscore
	

	def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list (Dict changed)
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The nDCG value as a number between 0 and 1
		"""

		nDCG = 0.0
		dcg = 0.0
		num = 0.0
		rel = []
		for i in range(k):
			if query_doc_IDs_ordered[i] in true_doc_IDs.keys():
				r =  true_doc_IDs[query_doc_IDs_ordered[i]]
				dcg += (5 - r) / math.log2(i+2)
				rel.append(r)
			else:
				rel.append(5)
		rel.sort() 
		idcg = 0.0
		for i in range(k):
			idcg += (5-rel[i])/math.log2(i+2)
		if idcg==0:
			return 0
		# print(dcg,idcg)
		nDCG = dcg / idcg
		
		return nDCG


	def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean nDCG value as a number between 0 and 1
		"""

		meanNDCG = 0.0
		d = {}
		for q in query_ids:
			d[q] = {}
		for e in qrels:
			if int(e["query_num"]) in d:
				d[int(e["query_num"])][int(e["id"])] = int(e["position"])

		#Fill in code here
		for i in range(len(query_ids)):
			meanNDCG += self.queryNDCG(doc_IDs_ordered[i],query_ids[i],d[query_ids[i]],k)
		meanNDCG /= len(query_ids)

		return meanNDCG


	def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of average precision of the Information Retrieval System
		at a given value of k for a single query (the average of precision@i
		values for i such that the ith document is truly relevant)

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The average precision value as a number between 0 and 1
		"""

		avgPrecision = 0.0
		num = 0.0
		for i in range(k):
			if query_doc_IDs_ordered[i] in true_doc_IDs:
				num+=1
				precision = num/(i+1)
				avgPrecision += precision
		#Fill in code here
		if num==0:
			return 0
		avgPrecision /= num
		return avgPrecision


	def meanAveragePrecision(self, doc_IDs_ordered, query_ids, q_rels, k):
		"""
		Computation of MAP of the Information Retrieval System
		at given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The MAP value as a number between 0 and 1
		"""

		meanAveragePrecision = 0.0
		d = {}
		for q in query_ids:
			d[q] = set() 
		for e in q_rels:
			if int(e["query_num"]) in d:
				d[int(e["query_num"])].add(int(e["id"]))

		#Fill in code here
		for i in range(len(query_ids)):
			meanAveragePrecision += self.queryAveragePrecision(doc_IDs_ordered[i],query_ids[i],d[query_ids[i]],k)
		meanAveragePrecision /= len(query_ids)
		#Fill in code here

		return meanAveragePrecision
	def precision_recall( self, precisions,recalls, query_ids):
		######   precisions    ranks *  num_queries
		# pass
		query_tup_list = [{} for i in range(len(query_ids))]
		for i in range(len(query_ids)):
			q = []
			for j in range(len(precisions)):
				q.append((recalls[j][i],-precisions[j][i]))
			q.sort()
			for item in q:
				if query_tup_list[i].get(item[0],None) != None:
					query_tup_list[i][item[0]] = max(-item[1],query_tup_list[i][item[0]])
				else:
					query_tup_list[i][item[0]] = -item[1]
		#interpolation 0 0.01 0.02 ... 1
		interpo = 21.0
		avg = [0.0 for i in range(int(interpo))]
		for i in range(len(query_ids)):
			for k in range(int(interpo)):
				p = -1
				for  a, b in query_tup_list[i].items():
					if a >= (k / (interpo-1) ):
						p = max(p,b)
				if p!=-1:
					avg[k] = avg[k] + p 
		avg = [avg[i]/(len(query_ids)) for i in range(len(avg))]
		x_axis = [i/(interpo-1) for i in range( int(interpo) ) ]
		return avg, x_axis
			

				



		