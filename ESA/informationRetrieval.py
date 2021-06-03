from util import *

# Add your import statements here




class InformationRetrieval():

	def __init__(self):
		# self.index = None
		# self.index_v1 = None
		self.index_v2 = None
		# self.doc_embeddings_schema = {
		# 	'transform': [
		# 		{
		# 			'concat': [
		# 				{
		# 					'transform': [
		# 						('word2vec', {'pretrained': 'en-glove'}),
		# 						('pool', {'pool_option': 'mean'})
		# 					]
		# 				},
		# 				{
		# 					'transform': [
		# 						('word2vec', {'pretrained': 'en-crawl'}),
		# 						('pool', {'pool_option': 'mean'})
		# 					]
		# 				},
		# 				{
		# 					'transform': [
		# 						'tfidf',
		# 					]
		# 				}
		# 			]
		# 			('svd',{'n_components': 3000})
		# 		}
		# 	]
		# }
		# self.emb = TextWiser(Embedding.Compound(schema=self.doc_embeddings_schema))
		self.unwanted_symbols = str.maketrans('', '', string.digits+"'+?:;,=.!@#$()/\\")
		self.unwanted_symbols[ord('-')] = ord(' ')
		self.stw = stopwords.words('english')
		self.lemmatizer = WordNetLemmatizer()
		self.corrector = jamspell.TSpellCorrector()
		self.corrector.LoadLangModel('en.bin')
		# self.wiki2vec = Wikipedia2Vec.load('enwiki_20180420_100d.pkl')
		# self.wiki2vec = Wikipedia2Vec.load('enwiki_20180420_win10_100d.pkl')
		# self.wdim = 100
		# self.emb4 = TextWiser(Embedding.Word(word_option=WordOptions.elmo, pretrained='small'),Transformation.Pool(pool_option=PoolOptions.mean))
		# self.emb4 = TextWiser(Embedding.Doc2Vec())
		# self.emb = TextWiser(Embedding.Word(word_option=WordOptions.word2vec, pretrained='en-extvec'),Transformation.Pool(pool_option=PoolOptions.mean)) # en-glove
		self.emb = TextWiser(Embedding.TfIdf()) # Transformation.NMF(n_components=500)
		# self.emb2 = TextWiser(Embedding.Word(word_option=WordOptions.word2vec, pretrained='en-crawl'),Transformation.Pool(pool_option=PoolOptions.mean))
		# self.emb3 = TextWiser(Embedding.Word(word_option=WordOptions.word2vec, pretrained='en-glove'),Transformation.Pool(pool_option=PoolOptions.mean))
		# self.emb = TextWiser(Embedding.USE(), Transformation.Pool(pool_option=PoolOptions.mean))
		# self.model_v3 = api.load('fasttext-wiki-news-subwords-300')
		self.model_v3 = api.load('glove-wiki-gigaword-300')
		# self.model_v3 = api.load('conceptnet-numberbatch-17-06-300')

	def run_v4(self, docs, docIDs, queries):
		nq = len(queries); nd = len(docs)
		l = np.zeros(nd)
		for i in range(nd):
			var = docs[i].split()
			var = [word for word in var if word.lower() not in self.stw]
			var = [word.translate(self.unwanted_symbols) for word in var]
			var = [self.lemmatizer.lemmatize(word) for word in var if len(word) > 2]
			l[i] = len(var)
			docs[i] = self.corrector.FixFragment(' '.join(var))
			if docs[i] == '':
				docs[i] = 'xyz '*25
				l[i] = 25
		for i in range(nq):
			var = queries[i].split()
			var = [word for word in var if word.lower() not in self.stw]
			var = [word.translate(self.unwanted_symbols) for word in var]
			var = [self.lemmatizer.lemmatize(word) for word in var if len(word) > 2]
			queries[i] = self.corrector.FixFragment(' '.join(var))
			if queries[i] == '':
				queries[i] = 'xyz '*25
		print('Basic Doc Processing Completed')
		self.model_v3.init_sims(replace=True)
		query_results = []
		for q in range(nq):
			var = [self.model_v3.wmdistance(queries[q], docs[d])/l[d] for d in range(nd)]
			query_results.append([docIDs[k] for k in np.argsort(var)])
			print(q+1, np.min(var), np.max(var), np.mean(var), query_results[-1][:10])
		return query_results
		

	def run_v3(self, docs, docIDs, queries):
		nq = len(queries); nd = len(docs)
		for i in range(nd):
			var = docs[i].split()
			var = [word for word in var if word.lower() not in self.stw]
			var = [word.translate(self.unwanted_symbols) for word in var]
			var = [self.lemmatizer.lemmatize(word) for word in var if len(word) > 2]
			docs[i] = self.corrector.FixFragment(' '.join(var))
			if docs[i] == '':
				docs[i] = 'xyz '*25
		for i in range(nq):
			var = queries[i].split()
			var = [word for word in var if word.lower() not in self.stw]
			var = [word.translate(self.unwanted_symbols) for word in var]
			var = [self.lemmatizer.lemmatize(word) for word in var if len(word) > 2]
			queries[i] = self.corrector.FixFragment(' '.join(var))
			if queries[i] == '':
				queries[i] = 'xyz '*25
		print('Basic Doc Processing Completed')
		self.model_v3.init_sims(replace=True)
		simi_lookup = WmdSimilarity(docs, self.model_v3, num_best = nd)
		doc_vecs = self.emb.fit_transform(docs)
		query_vecs = self.emb.forward(queries)
		alfa = 2.0; beta = 1.0; nq = 20
		query_results = []
		simi_values = np.zeros(nq)
		for i in range(nq):
			simi = simi_lookup[queries[i]]
			wms = np.zeros(nd)
			for j in range(nd):
				wms[simi[j][0]] = simi[j][1]
			q = np.reshape(query_vecs[i],(-1,1))
			qnorm = np.linalg.norm(q)
			hur = np.zeros(nd)
			for j in range(nd):
				k = doc_vecs[j]
				val = np.dot(k,q)[0]/(qnorm*np.linalg.norm(k))
				hur[j] = -1*((alfa*val)+(wms[j]*beta))
			# query_results.append([docIDs[j[0]] for j in simi])
			# print(i+1, simi[0][1], query_results[-1][:10])
			query_results.append([docIDs[k] for k in np.argsort(hur)])
			# print(i+1,-1*np.min(hur),query_results[-1][:10])
		return query_results

	def pure_tfidf(self, docs, docIDs, queries):
		nq = len(queries); nd = len(docs)
		tf = TextWiser(Embedding.TfIdf())
		doc_vecs = tf.fit_transform(docs)
		query_vecs = tf.forward(queries)
		query_results = []
		for i in range(nq):
			q = np.reshape(query_vecs[i],(-1,1))
			qnorm = np.linalg.norm(q)
			hur = np.zeros(nd)
			for j in range(nd):
				k = doc_vecs[j]
				val = np.dot(k,q)[0]/(qnorm*np.linalg.norm(k))
				hur[j] = -1*val
			query_results.append([docIDs[k] for k in np.argsort(hur)])
			# print(i+1,-1*np.min(hur),query_results[-1][:10])
		return query_results



	def buildIndex_v2(self, docs, docIDs):
		for i in range(len(docs)):
			var = docs[i].split()
			var = [word for word in var if word.lower() not in self.stw]
			var = [word.translate(self.unwanted_symbols) for word in var]
			var = [self.lemmatizer.lemmatize(word) for word in var if len(word) > 2]
			docs[i] = self.corrector.FixFragment(' '.join(var))
			if docs[i] == '':
				docs[i] = 'xyz '*25
		print('Basic Doc Processing Completed')
		self.index_v2 = {}
		vecs = self.emb.fit_transform(docs)
		vecs = vecs/np.reshape(np.linalg.norm(vecs,axis=1),(-1,1))
		vecs2 = self.emb2.fit_transform(docs)
		vecs2 = vecs2/np.reshape(np.linalg.norm(vecs2,axis=1),(-1,1))
		vecs3 = self.emb3.fit_transform(docs)
		vecs3 = vecs3/np.reshape(np.linalg.norm(vecs3,axis=1),(-1,1))
		# vecs4 = self.emb4.fit_transform(docs)
		# vecs4 = vecs4/np.reshape(np.linalg.norm(vecs4,axis=1),(-1,1))
		for i,j in enumerate(docIDs):
			self.index_v2[j] = np.concatenate((vecs[i],vecs2[i],vecs3[i]))
			# self.index_v2[j] = vecs[i]
		# print(vecs[0][0])
		# print(np.shape(vecs[0]),np.shape(vecs2[0]),np.shape(self.index_v2[docIDs[0]]))
		# print(np.shape(vecs[0]))
		# print(sum(vecs[0]))
		return None

	def rank_v2(self, queries):
		for i in range(len(queries)):
			var = queries[i].split()
			var = [word for word in var if word.lower() not in self.stw]
			var = [word.translate(self.unwanted_symbols) for word in var]
			var = [self.lemmatizer.lemmatize(word) for word in var if len(word) > 2]
			queries[i] = self.corrector.FixFragment(' '.join(var))
			if queries[i] == '':
				queries[i] = 'xyz '*25
		nq = len(queries)
		vecs = self.emb.forward(queries)
		vecs = vecs/np.reshape(np.linalg.norm(vecs,axis=1),(-1,1))
		vecs2 = self.emb2.forward(queries)
		vecs2 = vecs2/np.reshape(np.linalg.norm(vecs2,axis=1),(-1,1))
		vecs3 = self.emb3.forward(queries)
		vecs3 = vecs3/np.reshape(np.linalg.norm(vecs3,axis=1),(-1,1))
		# vecs4 = self.emb4.forward(queries)
		# vecs4 = vecs4/np.reshape(np.linalg.norm(vecs4,axis=1),(-1,1))
		query_vectors = np.concatenate((vecs,vecs2,vecs3),1)
		# query_vectors = self.emb.forward(queries)
		# query_vectors = query_vectors/np.reshape(np.linalg.norm(query_vectors,axis=1),(-1,1))
		query_results = []
		# d = np.shape(vecs)[1]; w = 1.0
		# print(np.shape(vecs),np.shape(vecs2),np.shape(vecs3),np.shape(query_vectors))
		dids =  list(self.index_v2.keys())
		for c in range(nq):
			q = np.reshape(query_vectors[c],(-1,1))
			qnorm = np.linalg.norm(q)
			# qnorm1 = np.linalg.norm(q[:d])
			# qnorm2 = np.linalg.norm(q[d:])
			cos_sim = np.zeros(len(dids))
			for i,j in enumerate(dids):
				k = self.index_v2[j]
				# val1 = np.dot(k[:d],q[:d])[0]/(qnorm1*np.linalg.norm(k[:d]))
				# val2 = np.dot(k[d:],q[d:])[0]/(qnorm2*np.linalg.norm(k[d:]))
				# val = w*val1 + val2
				val = np.dot(k,q)[0]/(qnorm*np.linalg.norm(k))
				if not np.isnan(val):
					cos_sim[i] = -1*val
			query_results.append([dids[k] for k in np.argsort(cos_sim)])
			# print(c+1,np.min(cos_sim),query_results[-1][:10])
		return query_results

	def buildIndex_v1(self, docs, docIDs):
		self.index_v1 = {}
		for i,doc in enumerate(docs):
			self.index_v1[docIDs[i]] = np.zeros(self.wdim)
			for line in doc:
				for word in line:
					cwords = self.corrector.FixFragment(word.translate(self.unwanted_symbols))
					for cword in cwords.split():
						if len(cword) > 2:
							try:
								wvec = self.wiki2vec.get_word_vector(cword)
								self.index_v1[docIDs[i]] = self.index_v1[docIDs[i]] + np.array(wvec)
							except:
								for w in splitter.split(cword):
									if len(w) > 3:
										try:
											wvec = self.wiki2vec.get_word_vector(w)
											self.index_v1[docIDs[i]] = self.index_v1[docIDs[i]] + np.array(wvec)
										except:
											# print(w,cword)
											pass
		# print(self.index_v1[docIDs[0]])
		return None

	def buildIndex(self, docs, docIDs):
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
	
	def rank_v1(self, queries):
		query_results = []
		query_vectors = [np.zeros(self.wdim) for i in range(len(queries))]
		for i,doc in enumerate(queries):
			for line in doc:
				for word in line:
					cwords = self.corrector.FixFragment(word.translate(self.unwanted_symbols))
					for cword in cwords.split():
						if len(cword) > 2:
							try:
								wvec = self.wiki2vec.get_word_vector(cword)
								query_vectors[i] = query_vectors[i] + np.array(wvec)
							except:
								for w in splitter.split(cword):
									if len(w) > 3:
										try:
											wvec = self.wiki2vec.get_word_vector(w)
											query_vectors[i] = query_vectors[i] + np.array(wvec)
										except:
											# print(w,cword)
											pass
		dids =  list(self.index_v1.keys())
		for q in query_vectors:
			qnorm = np.linalg.norm(q)
			cos_sim = np.zeros(len(dids))
			for i,j in enumerate(dids):
				val = np.dot(self.index_v1[j],q)/(qnorm*np.linalg.norm(self.index_v1[j]))
				if not np.isnan(val):
					cos_sim[i] = -1*val
			query_results.append([dids[j] for j in np.argsort(cos_sim)])
		# print(queries[0])
		# print(query_results[0][:20])
		return query_results

	def rank(self, queries):
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
		# print(queries[0])
		# print(doc_IDs_ordered[0][:20])
		return doc_IDs_ordered




