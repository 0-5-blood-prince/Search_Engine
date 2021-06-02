from util import *
import numpy as np
# Add your import statements here




class InformationRetrieval():

	def __init__(self):
		self.index = None
	
	def buildIndex_basic(self, docs, docIDs):
			"""
			Builds the document index in terms of the document
			IDs and stores it in the 'index' class variable

			Parameters
			----------
			arg1 : list
				A list of lists of lists where each sub-list is
				a document and each sub-sub-list is a sentence of the document
			arg2 : list
				A list of integers denoting IDs of the documents
			Returns
			-------
			None
			"""
			self.terms_list = set()
			self.term_doc_freq = {}
			self.index = {}
			self.num_docs = len(docs)
			self.doc_len = {}
			self.doc_id = docIDs.copy()
			doc_terms = {}
			for i in range(self.num_docs):
				doc_terms[docIDs[i]] = []
				for sentence in docs[i]:
					for term in sentence:
						if term not in self.terms_list:
							self.terms_list.add(term)
						if self.index.get((term, docIDs[i]),0.0) == 0.0:
							doc_terms[docIDs[i]].append(term)
						self.index[(term, docIDs[i])] = self.index.get((term,docIDs[i]),0.0)+1.0
			for term in self.terms_list:
				for id in docIDs:
					if self.index.get((term,id),0) != 0.0:
						self.term_doc_freq[term] = 1.0+self.term_doc_freq.get(term,0.0)
			for k in self.index.keys():
				self.index[k] = self.index[k]*math.log10(self.num_docs/(self.term_doc_freq.get(k[0],0.0)+1.0))
			for id in docIDs:
				v = 0.0
				for term in doc_terms[id]:
					v += (math.pow(self.index.get((term,id),0.0),2.0))
				self.doc_len[id] = math.sqrt(v)
			# print(list(self.doc_len.values())[:4])
			# print(list(self.index.keys())[:4],list(self.index.values())[:4])
			return

	def rank_basic(self, queries):
			"""
			Rank the documents according to relevance for each query

			Parameters
			----------
			arg1 : list
				A list of lists of lists where each sub-list is a query and
				each sub-sub-list is a sentence of the query
			

			Returns
			-------
			list
				A list of lists of integers where the ith sub-list is a list of IDs
				of documents in their predicted order of relevance to the ith query
			"""

			doc_IDs_ordered = []
			query_dic = {}
			query_len = {}
			query_terms = [[] for i in range(len(queries))]
			for i in range(len(queries)):
				for sentence in queries[i]:
					for term in sentence:
						if query_dic.get((term, i),0.0) == 0.0:
							query_terms[i].append(term)
						query_dic[(term, i)] = query_dic.get((term, i),0.0)+1.0
			for k in query_dic.keys():
				query_dic[k] = query_dic[k]*math.log10(self.num_docs/(self.term_doc_freq.get(k[0],0.0)+1.0))
			for id in range(len(queries)):
				v = 0.0
				for term in self.terms_list:
					v += (math.pow(query_dic.get((term,id),0.0),2.0))
				query_len[id] = math.sqrt(v)
			for i in range(len(queries)):
				buff = []
				for d in self.doc_id:
					if self.doc_len[d] == 0.0:
						buff.append((0.0,d))
						continue
					dot = 0.0
					for term in query_terms[i]:
						dot += (query_dic.get((term,i),0.0)*self.index.get((term,d),0.0))
					buff.append((dot/(query_len[i]*self.doc_len[d]),d))
				buff.sort(reverse=True)
				doc_IDs_ordered.append([i[1] for i in buff])
			return doc_IDs_ordered
		
	def buildIndex_lsi(self, docs, docIDs, dim):
		"""
		Builds the document index in terms of the document
		IDs and stores it in the 'index' class variable

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is
			a document and each sub-sub-list is a sentence of the document
		arg2 : list
			A list of integers denoting IDs of the documents
		Returns
		-------
		None
		"""
		self.dim = dim
		### Set of Terms
		self.terms_list = set()
		### GLobal measure
		self.term_doc_freq = {}
		### term - doc inverse index
		self.index = {}
		self.num_docs = len(docs)
		### Length of document vector
		self.doc_len = {}
		self.doc_id = docIDs.copy()
		## Terms in each doc
		doc_terms = {}
		for i in range(self.num_docs):
			doc_terms[docIDs[i]] = []

			for sentence in docs[i]:
				for term in sentence:
					if term not in self.terms_list:
						self.terms_list.add(term)

					if self.index.get((term, docIDs[i]),0.0)  == 0.0: ### get(key,0.0) return 0.0 if not available
						doc_terms[docIDs[i]].append(term)

					self.index[(term, docIDs[i])] = self.index.get((term,docIDs[i]),0.0)+1.0
		for term in self.terms_list:
			for id in docIDs:
				if self.index.get((term,id),0) != 0.0:
					self.term_doc_freq[term] = 1.0 + self.term_doc_freq.get(term,0.0)  
		### Using tf idf improved LSA performance
		for k in self.index.keys():
			self.index[k] = self.index[k] * math.log10(self.num_docs/(self.term_doc_freq.get(k[0],0.0)+1.0))  # (tf) * (idf) SMoothed idf

		
		# print(self.terms_list)

		# print(list(self.doc_len.values())[:4])
		# print(list(self.index.keys())[:4],list(self.index.values())[:4])
		return
	def svd_lsi(self , docIDs):
		self.num_index = []
		tl = list(self.terms_list)
		tl.sort()
		# print(tl)
		# for t in tl:
			# print(t)
		for i in range(len(tl)):
			a = [0.0 for j in range(self.num_docs)]
			for j in range(self.num_docs):
				a[j] = self.index.get((tl[i],docIDs[j]),0.0) 
			self.num_index.append(a)
		self.num_index = np.asarray(self.num_index)


		self.u, self.sig, self.v = np.linalg.svd(self.num_index)
		self.sig = np.diag(self.sig)
		# print(self.sig)
		self.u = self.u[:, :self.dim]
		self.sig = self.sig[:self.dim, :self.dim]
		# print(self.v.shape)
		self.v = self.v.T[:, :self.dim]
		
		# print(self.num_index)
		self.num_index = self.u @ self.sig @ self.v.T
		# print(self.num_index)
		# print(self.num_index.T.shape , self.u.shape , np.linalg.pinv(self.sig).shape)
		
		# self.transform_docs = self.num_index.T   @ self.u @ self.sig
		self.transform_docs = self.v
		# self.transform_docs = self.num_index.T
		for j in range(self.num_docs):
			self.doc_len[docIDs[j]] = np.linalg.norm(self.transform_docs[j])
		return 

	def rank_lsi(self, queries):
		"""
		Rank the documents according to relevance for each query

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is a query and
			each sub-sub-list is a sentence of the query
		

		Returns
		-------
		list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		"""

		doc_IDs_ordered = []
		query_dic = {}
		query_len = {}
		query_terms = [[] for i in range(len(queries))]
		for i in range(len(queries)):
			for sentence in queries[i]:
				for term in sentence:
					if query_dic.get((term, i),0.0) == 0.0:
						query_terms[i].append(term)
					query_dic[(term, i)] = query_dic.get((term, i),0.0)+1.0
		for k in query_dic.keys():
			query_dic[k] = query_dic[k]*math.log10(self.num_docs/(self.term_doc_freq.get(k[0],0.0)+1.0))


		query_list = []
		tl = list(self.terms_list)
		tl.sort()
		for i in range(len(queries)):
			a = [0.0 for j in range(len(tl))]
			for j in range(len(tl)):
				a[j] = query_dic.get((tl[j],i),0.0) 
			query_list.append(a)
		query_list = np.asarray(query_list)

		print(query_list.shape , self.u.shape , np.linalg.pinv(self.sig).shape)
		self.transform_queries = query_list  @ self.u @ (self.sig)



		
		# for id in range(len(queries)):
		# 	v = 0.0
		# 	for term in self.terms_list:
		# 		v += (math.pow(query_dic.get((term,id),0.0),2.0))
		# 	query_len[id] = math.sqrt(v)
		self.query_len = [0 for i in range(len(queries))]
		for j in range(len(queries)):
			self.query_len[j] = np.linalg.norm(self.transform_queries[j])

		for i in range(len(queries)):
			buff = []
			for j in range(self.num_docs):
				dot = self.transform_queries[i] @ self.transform_docs[j]
				d = self.doc_id[j]
				if self.query_len[i] == 0 or self.doc_len[d] == 0:
					# print("Bro",i,self.query_len[i],d,self.doc_len[d])  ### 471 and 995 are NULL Docs
					continue 
				buff.append((dot/(self.query_len[i]*self.doc_len[d]),d))
			buff.sort(reverse=True)
			doc_IDs_ordered.append([i[1] for i in buff])
		return doc_IDs_ordered

	def train(self, queries, query_ids, qrels, w):
		
		query_dic = {}
		query_len = {}
		query_terms = [[] for i in range(len(queries))]
		for i in range(len(queries)):
			for sentence in queries[i]:
				for term in sentence:
					if query_dic.get((term, i),0.0) == 0.0:
						query_terms[i].append(term)
					query_dic[(term, i)] = query_dic.get((term, i),0.0)+1.0
		for k in query_dic.keys():
			query_dic[k] = query_dic[k]*math.log10(self.num_docs/(self.term_doc_freq.get(k[0],0.0)+1.0))


		query_list = []
		tl = list(self.terms_list)
		tl.sort()
		for i in range(len(queries)):
			a = [0.0 for j in range(len(tl))]
			for j in range(len(tl)):
				a[j] = query_dic.get((tl[j],i),0.0) 
			query_list.append(a)
		query_list = np.asarray(query_list)

		# print(query_list.shape , self.u.shape , np.linalg.pinv(self.sig).shape)
		self.transform_queries = query_list @ self.u @ (self.sig)


		#   Supervised learning is done in context space so below transformation not necessary 
		# if you want to do it do this transformationa and use num_index instead of tranform docs
		# self.transform_queries = self.u @ self.sig @ np.transpose(self.transform_queries)
		
		
		self.query_len = [0 for i in range(len(queries))]
		for j in range(len(queries)):
			self.query_len[j] = np.linalg.norm(self.transform_queries[j])
			self.transform_queries[j] /= self.query_len[j]
		self.transform_queries = self.transform_queries.T 
		
		shape_Q = self.transform_queries.shape
		# assert(shape_Q[0]==self.dim and shape_Q[1]==len(queries))

		for j in range(self.num_docs):
			if self.doc_len[self.doc_id[j]]!=0:
				self.transform_docs[j] /= self.doc_len[self.doc_id[j]]
		self.transform_docs = self.transform_docs.T 

		shape_D = self.transform_docs.shape
		# assert(shape_D[0]==self.dim and shape_D[1]==self.num_docs)
		print(self.transform_queries.shape,self.transform_docs.shape)

		Q_concat = np.concatenate((self.transform_queries, self.transform_docs),axis=1) 
		# Q_concat = self.transform_queries
		A = np.zeros((self.num_docs,len(queries)))
		d = {}
		for q in query_ids:
			d[q] = set() 
		for e in qrels:
			if int(e["query_num"]) in d:
				d[int(e["query_num"])].add(int(e["id"]))

		for i in range(len(queries)):
			for j in range(self.num_docs):
				if self.doc_id[j] in d[query_ids[i]]:
					A[j][i] = 1
		# print(A)
		A = A * w
		A_concat = np.concatenate((A,(self.transform_docs.T @ self.transform_docs)), axis=1)
		# A_concat = A
		print(A_concat)
		print("Norm A concat :",np.linalg.norm(A_concat))
		#solving M*
		q,r = np.linalg.qr(self.transform_docs.T)

		y = q.T @ A_concat
		M_star = np.linalg.pinv(r) @ y 

		print("Norm A concat - DtM:",np.linalg.norm( (A_concat- (self.transform_docs.T @ M_star) ) ))
		#solving X*

		print("Norm M*t :", np.linalg.norm(M_star.T))
		q,r = np.linalg.qr(Q_concat.T)

		y = q.T @ M_star.T 
		X_star_trans = np.linalg.pinv(r) @ y 

		print("Norm M*t - Qt Xt:", np.linalg.norm( (M_star.T - (Q_concat.T @ X_star_trans) ) ))
		X_star = X_star_trans.T
		# assert(X_star.shape==(self.dim,self.dim))

		sim_matrix_sup = self.transform_docs.T @ X_star @ Q_concat 
		sim_matrix_sup = sim_matrix_sup[:,:(len(queries))]
		sim_matrix_lsi = self.transform_docs.T @ self.transform_queries
		# print(sim_matrix_lsi)
		# print(sim_matrix_sup)
		# print(sim_matrix_lsi)
		# print(sim_matrix_sup)
		doc_IDs_ordered_lsi = []
		doc_IDs_ordered_sup = []
		for i in range(len(queries)):
			buff_lsi = []
			buff_sup = []
			for j in range(self.num_docs):
				d = self.doc_id[j]
				buff_lsi.append((sim_matrix_lsi[j][i],d))
				buff_sup.append((sim_matrix_sup[j][i],d))
			buff_lsi.sort(reverse=True)
			buff_sup.sort(reverse=True)
			doc_IDs_ordered_lsi.append([i[1] for i in buff_lsi])
			doc_IDs_ordered_sup.append([i[1] for i in buff_sup])
		# for i in range(self.num_docs):
		# 	print(A.T[0][i],sim_matrix_lsi.T[0][i],sim_matrix_sup.T[0][i])
		return X_star  , doc_IDs_ordered_lsi, doc_IDs_ordered_sup
	def test(self, queries, query_ids, qrels, X):
		query_dic = {}
		query_len = {}
		query_terms = [[] for i in range(len(queries))]
		for i in range(len(queries)):
			for sentence in queries[i]:
				for term in sentence:
					if query_dic.get((term, i),0.0) == 0.0:
						query_terms[i].append(term)
					query_dic[(term, i)] = query_dic.get((term, i),0.0)+1.0
		for k in query_dic.keys():
			query_dic[k] = query_dic[k]*math.log10(self.num_docs/(self.term_doc_freq.get(k[0],0.0)+1.0))


		query_list = []
		tl = list(self.terms_list)
		tl.sort()
		for i in range(len(queries)):
			a = [0.0 for j in range(len(tl))]
			for j in range(len(tl)):
				a[j] = query_dic.get((tl[j],i),0.0) 
			query_list.append(a)
		query_list = np.asarray(query_list)

		print(query_list.shape , self.u.shape , np.linalg.pinv(self.sig).shape)
		self.transform_queries = query_list  @ self.u @ (self.sig)


		#   Supervised learning is done in context space so below transformation not necessary 
		# if you want to do it do this transformationa and use num_index instead of tranform docs
		# self.transform_queries = self.u @ self.sig @ np.transpose(self.transform_queries)
		
		
		self.query_len = [0 for i in range(len(queries))]
		for j in range(len(queries)):
			self.query_len[j] = np.linalg.norm(self.transform_queries[j])
			self.transform_queries[j] /= self.query_len[j]
		self.transform_queries = self.transform_queries.T 
		 
		shape_Q = self.transform_queries.shape
		# assert(shape_Q[0]==self.dim and shape_Q[1]==len(queries))

		Q_concat = np.concatenate((self.transform_queries, self.transform_docs),axis=1)
		# Q_concat = self.transform_queries
		
		# A = np.zeros((self.num_docs,len(queries)))
		# d = {}
		# for q in query_ids:
		# 	d[q] = set() 
		# for e in qrels:
		# 	if int(e["query_num"]) in d:
		# 		d[int(e["query_num"])].add(int(e["id"]))

		# for i in range(len(queries)):
		# 	for j in range(self.num_docs):
		# 		if self.doc_id[j] in d[query_ids[i]]:
		# 			A[j][i] = 1
		# print(A)
		# A_concat = np.concatenate((A,(self.transform_docs.T @ self.transform_docs)), axis=1)
		sim_matrix_sup = self.transform_docs.T @ X @ Q_concat 
		sim_matrix_sup = sim_matrix_sup[:,:(len(queries))]
		sim_matrix_lsi = self.transform_docs.T @ self.transform_queries
		print(sim_matrix_lsi)
		print(sim_matrix_sup)
		# print(sim_matrix_lsi)
		# print(sim_matrix_sup)
		doc_IDs_ordered_lsi = []
		doc_IDs_ordered_sup = []
		for i in range(len(queries)):
			buff_lsi = []
			buff_sup = []
			for j in range(self.num_docs):
				d = self.doc_id[j]
				buff_lsi.append((sim_matrix_lsi[j][i],d))
				buff_sup.append((sim_matrix_sup[j][i],d))
			buff_lsi.sort(reverse=True)
			buff_sup.sort(reverse=True)
			doc_IDs_ordered_lsi.append([i[1] for i in buff_lsi])
			doc_IDs_ordered_sup.append([i[1] for i in buff_sup])
		return doc_IDs_ordered_lsi, doc_IDs_ordered_sup



	
		

		
