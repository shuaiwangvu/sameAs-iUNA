# this is the class where SameAsEqSolver is defined
import networkx as nx
from SameAsEqGraph import get_simp_IRI, get_namespace, get_name
from pyvis.network import Network
import community
import collections
import matplotlib.pyplot as plt
from random import randint
import pandas as pd
import requests
from collections import Counter
from rfc3987 import  parse
import urllib.parse
from hdt import HDTDocument, IdentifierPosition
from z3 import *
import csv
from rdflib import Literal, XSD
from networkx.algorithms.connectivity import is_locally_k_edge_connected

UNKNOWN = 0
REMOVE = 1
KEEP = 2

hdt_source = None
hdt_label = None
hdt_comment = None

# hdt_source = HDTDocument("typeA.hdt")
# hdt_label = HDTDocument("label_May.hdt")
# hdt_comment = HDTDocument("comment_May.hdt")
#
# PATH_META = "/home/jraad/ssd/data/identity/metalink/metalink.hdt"
# hdt_metalink = HDTDocument(PATH_META)
#
# hdt_sameas_source = HDTDocument("sameas_source.hdt")

# PATH_LOD = "/scratch/wbeek/data/LOD-a-lot/data.hdt"
# hdt_lod = HDTDocument(PATH_LOD)

my_has_label_in_file = "https://krr.triply.cc/krr/metalink/def/hasLabelInFile" # a relation
my_has_comment_in_file = "https://krr.triply.cc/krr/metalink/def/hasCommentInFile" # a relation
rdfs_label = "http://www.w3.org/2000/01/rdf-schema#label"
rdfs_comment = 'http://www.w3.org/2000/01/rdf-schema#comment'
my_file_IRI_prefix = "https://krr.triply.cc/krr/metalink/fileMD5/" # followed by the MD5 of the data
my_file = "https://krr.triply.cc/krr/metalink/def/File"
my_exist_in_file = "https://krr.triply.cc/krr/metalink/def/existsInFile" # a relation
my_has_num_occurences_in_files = "https://krr.triply.cc/krr/metalink/def/numOccurences" #
my_redirect = "https://krr.triply.cc/krr/metalink/def/redirectedTo" # a relation


meta_eqSet = "https://krr.triply.cc/krr/metalink/def/equivalenceSet"
meta_comm = "https://krr.triply.cc/krr/metalink/def/Community"
meta_identity_statement = "https://krr.triply.cc/krr/metalink/def/IdentityStatement"
rdf_statement = "http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement"

rdf_subject = "http://www.w3.org/1999/02/22-rdf-syntax-ns#subject"
rdf_object = "http://www.w3.org/1999/02/22-rdf-syntax-ns#object"
rdf_predicate = "http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate"

rdf_type = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"


# find the metalink id for the subject-object pair.
def find_statement_id(subject, object):

	triples, cardinality = hdt_metalink.search_triples("", rdf_subject, subject)
	collect_statement_id_regarding_subject = set()

	for (s,p,o) in triples:
		collect_statement_id_regarding_subject.add(str(s))

	triples, cardinality = hdt_metalink.search_triples("", rdf_object, object)

	collect_statement_id_regarding_object = set()

	for (s,p,o) in triples:
		collect_statement_id_regarding_object.add(str(s))

	intersection = collect_statement_id_regarding_object.intersection(collect_statement_id_regarding_subject)

	if len(intersection) == 1:
		return list(intersection)[0]
	elif len(intersection) > 1:
		print ('ERROR!!!!')
		return None
	else:
		return None

# get all redirect in a list. If no redirect, then return []

import requests
from requests.exceptions import Timeout

NOTFOUND = 1
NOREDIRECT = 2
ERROR = 3
TIMEOUT = 4
REDIRECT = 5



# define a class of graph solver
class GraphSolver():
	# configure the weight file
	# location of each node

	def __init__(self, path_to_input_graph, path_to_gold_standard_graph, path_metalink_id = None):
		# input graph
		self.input_graph = nx.Graph()
		self.gold_standard_graph = nx.Graph()
		self.gold_standard_partition = []
		input_graph_data = pd.read_csv(path_to_input_graph)

		sources = input_graph_data['SUBJECT']
		targets = input_graph_data['OBJECT']
		edge_data = zip(sources, targets)

		for (s,t) in edge_data:
			self.input_graph.add_node(s, short_IRI = get_simp_IRI(s), namespace = get_namespace(s), group = 0)
			self.input_graph.add_node(t, short_IRI = get_simp_IRI(t), namespace = get_namespace(t), group = 0)
			self.input_graph.add_edge(s, t)

		goldstandard_graph_data = pd.read_csv(path_to_gold_standard_graph, delimiter='\t')

		entities = goldstandard_graph_data['Entity']
		annotations = goldstandard_graph_data['Annotation']
		goldstandard_data = zip(entities, annotations)
		ann_to_id = {}
		id_to_ann = {}
		acc_group_id = 0
		ann_to_id['unknown'] = 0
		id_to_ann[0] = 'unknown'
		for (e, ann) in goldstandard_data:
			group_id = 0
			if ann != 'unknown':
				if ann in ann_to_id.keys():
					group_id = ann_to_id[ann]
					id_to_ann[group_id] = ann
				else:
					acc_group_id += 1
					group_id += acc_group_id
					ann_to_id[ann] = group_id
					id_to_ann[group_id] = ann

			self.gold_standard_graph.add_node(e, short_IRI = get_simp_IRI(e), namespace = get_namespace(e), annotation = ann, group = ann_to_id[ann])
			#
			# print ('\nentity: ', e)
			# print ('annota: ', ann)
			# print ('group : ', ann_to_id[ann])

		# add edges from the input graph
		self.gold_standard_graph.add_edges_from(self.input_graph.edges())

		ct = Counter()
		for n in self.gold_standard_graph.nodes():
			group_id = self.gold_standard_graph.nodes[n]['group']
			ct[group_id] += 1

		print ('\n\n\n')
		for group in ct.keys():
			print ('Group ', group, 'is about ', id_to_ann[group], ' with ', ct[group], ' entities')

		print ('now compute removed edges in the gold standard')
		# edges
		count_removed = 0
		count_involving_unknown = 0
		for (s, t) in self.gold_standard_graph.edges():
			# print ('s = ', s, 'group =', self.gold_standard_graph.nodes[s]['group'])
			# print ('t = ', t, 'group =', self.gold_standard_graph.nodes[t]['group'])

			if self.gold_standard_graph.nodes[s]['group'] == UNKNOWN or self.gold_standard_graph.nodes[t]['group'] == UNKNOWN:
				self.gold_standard_graph.edges[s, t]['decision'] = UNKNOWN
				count_involving_unknown += 1
			elif self.gold_standard_graph.nodes[s]['group'] != self.gold_standard_graph.nodes[t]['group']:
				self.gold_standard_graph.edges[s, t]['decision'] = REMOVE
				count_removed += 1
			else:
				self.gold_standard_graph.edges[s, t]['decision'] = KEEP

		print ('there are in total ', count_removed, ' edges removed')
		print ('there are in total ', count_involving_unknown, ' edges involving unknown')

		for n in self.gold_standard_graph.nodes():
			self.gold_standard_partition.append(self.gold_standard_graph.nodes[n]['group'])
		# print ('gold standard coloring partition',self.gold_standard_partition)

		for (s, t) in self.input_graph.edges():
			self.input_graph.edges[s,t]['metalink_ID'] = None

		if path_metalink_id != None:
			print ('path to metalink id ', path_metalink_id )
			print ('<<< Adding metalink id to edges >>>')
			count_metalink_ids = 0
			with open(path_metalink_id,  newline='') as csvfile:
				reader = csv.DictReader(csvfile, delimiter=' ')
				for row in reader:
					# print(row['SUBJECT'], row['OBJECT'], row['METALINK_ID'])
					s = row['SUBJECT']
					t = row['OBJECT']
					id = row['METALINK_ID']
					self.input_graph.edges[s,t]['metalink_ID'] = id
					count_metalink_ids += 1
			print ('metalink ids added ', count_metalink_ids)

		# for visulization
		self.position = nx.spring_layout(self.input_graph)
		# net = Network()

		# based on this, we generate the corresponding edges
		self.namespace_graph = nx.Graph()
		self.source_graph = nx.Graph()

		# additional information
		self.redirect_graph = nx.DiGraph()
		self.encoding_equality_graph = nx.Graph()

		# type A B C graphs
		self.typeA_graph = nx.Graph()
		self.typeB_graph = nx.Graph()
		self.typeC_graph = nx.Graph()

		# weight graph
		# no need for this one since we have the original graph anyway

		# solving: (weighted unique name constraints, from UNA)
		# self.positive_UNC = [] # redirect and equivalence encoding
		# self.negative_UNC = [] # namespace or source


		# result
		self.result_partition = []
		self.result_graph = None

	def get_redirect_graph (self, path_to_redirect_graph):
		print ('\n\n<<< Getting redirect graph >>>')
		hdt_redi = HDTDocument(path_to_redirect_graph)
		triples, cardinality = hdt_redi.search_triples("", my_redirect, "")
		# my_redirect : https://krr.triply.cc/krr/metalink/def/redirectedTo
		for (s, _, t) in triples:
			# print ('redirect: ', s, ' ', t)
			self.redirect_graph.add_edge(s, t)

		print ('there are in total', len(self.redirect_graph.nodes()), 'nodes not in redirect graph')
		print ('there are in total', len(self.redirect_graph.edges()), 'edges not in redirect graph')

	def add_weight(self, path_to_weight):
		print ('<<< Getting the weights >>>')
		print ('weight file at ', path_to_weight)
		hdt_weight = HDTDocument(path_to_weight)
		for s, t in self.input_graph.edges():
			self.input_graph.edges[s,t]['weight'] = 0
			id = self.input_graph.edges[s, t]['metalink_ID']
			if id != None:
				triples, cardinality = hdt_weight.search_triples(id, my_has_num_occurences_in_files, "")
				if cardinality == 1:
					for (_,_,w) in triples:
						# w = Literal(w, datatype=XSD.integer)
						w = int(w[1:][:(w[1:].find('"'))])
						# print (w ,' has type: ',type(w))
						self.input_graph.edges[s,t]['weight'] = w
						# print ('weight = ', w)


	def get_encoding_equality_graph(self):
		print ('\n\n <<< Getting encoding equality graph >>>')
		#step 1: make an authority map to node
		authority_map = {}
		variance_map = {}

		for n in self.input_graph.nodes:
			# print ('\n\niri = ', n)
			rule='IRI'
			d = parse(n, rule) # ’IRI_reference’

			# print ('authority = ',d['authority'])
			if d['authority'] in authority_map.keys():
				authority_map[d['authority']].append(n)
			else:
				authority_map[d['authority']] = []
			# print ('authority map: ',authority_map[d['authority']])
		# step 2: construct a dictionary of each node against
			# first, do the decoding and add to the list

			uq = urllib.parse.unquote(n)
			# print ('uq = ', uq)
			variance_map[n] = set()
			if d != n:
				variance_map[n].add(uq)
			# second, get an ecoding of the current iri
			prefix, sign, name  = get_name(n)
			quote_name = urllib.parse.quote(name)
			new_iri = prefix + sign + quote_name
			# print ('new_iri = ', new_iri)
			if new_iri != n:
				variance_map[n].add(new_iri)

			# print ('prefix', prefix)
			# print ('sign', sign)
			# print ('name', name)
			# print ('quote_name', quote_name)
			# print ('new iri = ', new_iri)

			# print ('variance map = ',variance_map[n])

		encoding_equality_graph = nx.Graph()
		for iri_with_same_authority in authority_map.values():
			for i in iri_with_same_authority:
				for j in iri_with_same_authority:
					if i != j:
						# test if i is the same as any of j's variances
						for jv in variance_map[j]:
							if i == jv:
								encoding_equality_graph.add_edge(i, j)
		encoding_equivalence_edges = encoding_equality_graph.edges()
		print ('there are ', len (encoding_equivalence_edges), ' edges in the equivalence graph')
		for (s,t) in encoding_equivalence_edges:
			if not s in self.input_graph.nodes():
				print ('not in original graph ', s)
			if not t in self.input_graph.nodes():
				print ('not in original graph ', t)
		self.encoding_equality_graph = encoding_equality_graph

	def get_namespace_graph(self):
		print ('\n\n <<< Getting namespace graph >>>')
		namespace_to_entities = {}
		for e in self.input_graph.nodes():
			ns = self.input_graph.nodes[e]['namespace']
			# print (e)
			# print ('has name space', ns)
			if ns in namespace_to_entities.keys():
				namespace_to_entities[ns].append(e)
			else:
				namespace_to_entities[ns] = [e]

		for ns in namespace_to_entities.keys():
			ns_entities = namespace_to_entities[ns]
			# if len (ns_entities)>1:
				# print ('\nnamesapce ', ns)
				# print ('has ', len (ns_entities), ' entities:')
				# for e in namespace_to_entities[ns]:
				# 	print ('\t',e)

			for i, e in enumerate(ns_entities):
				for f in ns_entities[i+1:]:
					self.namespace_graph.add_edge(e, f)

		print ('The namespace graph has ', len (self.namespace_graph.edges()), ' attacking edges in total')

	def get_typeA_graph (self, typeA_filename):
		print ('\n<<< generating typeA graph>>>')
		hdt_source = HDTDocument(typeA_filename)

		# load the resources and
		source_files = []
		for e in self.input_graph.nodes():
			triples, cardinality = hdt_source.search_triples(e, "", "")
			for (_, _, file) in triples:
				source_files.append(file)
		print ('There are in total ', len (source_files), 'source files')

		file_to_entities = {}
		for e in self.input_graph.nodes():
			triples, cardinality = hdt_source.search_triples(e, "", "")
			for (e, _, file) in triples:
				if file not in file_to_entities.keys():
					file_to_entities [file] = [e]
				else:
					file_to_entities [file].append(e)

		for f in file_to_entities.keys():
			for i, s in enumerate(file_to_entities[f]):
				for t in list(file_to_entities[f])[i+1:]:
					self.typeA_graph.add_edge(s, t)
		print ('There are in total ', len (self.typeA_graph.edges()), 'attacking edges from this source file')

	def get_typeB_graph (self, typeB_filename):
		print ('\n<<<<generating typeB graph>>>')

		hdt_label = HDTDocument(typeB_filename)

		# load the resources and
		source_files = []
		for e in self.input_graph.nodes():
			triples, cardinality = hdt_label.search_triples(e, my_has_label_in_file, "")
			for (_, _, file) in triples:
				source_files.append(file)
		print ('There are in total ', len (source_files), 'label source files')

		file_to_entities = {}
		for e in self.input_graph.nodes():
			triples, cardinality = hdt_label.search_triples(e, my_has_label_in_file, "")
			for (e, _, file) in triples:
				if file not in file_to_entities.keys():
					file_to_entities [file] = [e]
				else:
					file_to_entities [file].append(e)


		for file in file_to_entities.keys():
			# if len(file_to_entities[file]) >1:
			# 	print ('\nfile ', file, ' has ', len (file_to_entities[file]), 'entities')

			for i, e in enumerate(file_to_entities[file]):
				# if len(file_to_entities[file]) >1:
				# 	print ('No.', i, ' = ', e)
				for f in list(file_to_entities[file])[i+1:]:
					self.typeB_graph.add_edge(e, f)
		print ('There are in total ', len (self.typeB_graph.edges()), 'attacking edges from this label file')

	def get_typeC_graph (self, typeC_filename):
		print ('\n<<<<generating typeC graph>>>')
		hdt_comment = HDTDocument(typeC_filename)
		# load the resources and
		source_files = []
		for e in self.input_graph.nodes():
			triples, cardinality = hdt_comment.search_triples(e, my_has_comment_in_file, "")
			for (_, _, file) in triples:
				source_files.append(file)
		print ('There are in total ', len (source_files), 'comment source files')

		file_to_entities = {}
		for e in self.input_graph.nodes():
			triples, cardinality = hdt_comment.search_triples(e, my_has_comment_in_file, "")
			for (e, _, file) in triples:
				if file not in file_to_entities.keys():
					file_to_entities [file] = [e]
				else:
					file_to_entities [file].append(e)

		for file in file_to_entities.keys():
		# 	if len(file_to_entities[file]) >1:
		# 		print ('\nfile ', file, ' has ', len (file_to_entities[file]), 'entities')

			for i, e in enumerate(file_to_entities[file]):
				# if len(file_to_entities[file]) >1:
				# 	print ('No.', i, ' = ', e)
				for f in list(file_to_entities[file])[i+1:]:
					self.typeC_graph.add_edge(e, f)
		print ('There are in total ', len (self.typeC_graph.edges()), 'attacking edges from this comment file')

	def add_redundency_weight(self):
		found = 0
		# for (s, t) in self.input_graph.edges():
		# 	if find_statement_id(s, t) != None:
		# 		found += 1
		# print ('found in metalink: ',found)
		ct = Counter()
		for (s, t) in self.input_graph.edges():
		# hdt_sameas_source
			sameas_statement_id = find_statement_id(s, t)
			if sameas_statement_id != None:
				triples, cardinality = hdt_sameas_source.search_triples(sameas_statement_id, my_exist_in_file, "")
				ct [cardinality] += 1
				if cardinality > 3:
					print ('source:\t ', s)
					print ('target:\t ', t)
					print ('metalink id:\t ', sameas_statement_id)
					print ('weight:\t ', cardinality)

		print ('counting ', ct)

	def show_input_graph(self):
		g = self.input_graph
		# sp = nx.spring_layout(g)
		nx.draw_networkx(g, pos=self.position, with_labels=False, node_size=25)
		plt.title('Input')
		plt.show()

	def show_result_graph(self):
		g = self.result_graph
		# sp = nx.spring_layout(g)
		nx.draw_networkx(g, pos=self.position, with_labels=False, node_size=25)
		plt.title('Result')
		plt.show()

	def show_redirect_graph(self):
		print ('\n\n <<< Getting redirect graph >>>')
		g = self.redirect_graph
		# sp = nx.spring_layout(g)
		# nx.draw_networkx(g, pos=self.position, with_labels=False, node_size=35)

		nx.draw_networkx(self.input_graph, pos=self.position, with_labels=False, node_size=25)
		nx.draw_networkx_edges(g, pos=self.position, edge_color='red', connectionstyle='arc3,rad=0.2')

		plt.title('Redirect')
		plt.show()

	def show_encoding_equivalence_graph(self):
		g = self.encoding_equality_graph
		# print ('now plot a graph with ', len (g.edges()), ' equivalence edges')
		# sp = nx.spring_layout(g)
		# nx.draw_networkx(g, pos=self.position, with_labels=False, node_size=35)

		nx.draw_networkx(self.input_graph, pos=self.position, with_labels=False, node_size=25)
		nx.draw_networkx_edges(g, pos=self.position, edge_color='blue', connectionstyle='arc3,rad=0.2')

		plt.title('Encoding Equivalence')
		plt.show()

	def show_namespace_graph(self):
		g = self.namespace_graph
		# sp = nx.spring_layout(g)
		# nx.draw_networkx(g, pos=self.position, with_labels=False, node_size=35)

		nx.draw_networkx(self.input_graph, pos=self.position, with_labels=False, node_size=25)
		nx.draw_networkx_edges(g, pos=self.position, edge_color='blue', connectionstyle='arc3,rad=0.2')

		plt.title('Namesapce Attacking edges')
		plt.show()

	def show_gold_standard_graph (self):
		g = self.gold_standard_graph
		# counter = collections.Counter(values)
		# print(counter)
		# sp = nx.spring_layout(g)
		edge_color = []
		for (s,t) in self.gold_standard_graph.edges:
			if self.gold_standard_graph.edges[s, t]['decision'] == UNKNOWN:
				edge_color.append('yellow')
			elif self.gold_standard_graph.edges[s, t]['decision'] == REMOVE:
				edge_color.append('red')
			else:
				edge_color.append('black')
			# edge_color.append(self.gold_standard_graph.edges[s, t]['decision'])

		# print ('edge_color = ', edge_color)

		nx.draw_networkx(g, pos=self.position, with_labels=False, node_size=35, node_color=self.gold_standard_partition, edge_color=edge_color)
		# plt.axes('off')
		plt.title('Gold standard')
		plt.show()



	def partition_leuven(self):
		g = self.input_graph
		self.result_graph = self.input_graph.copy()

		partition = community.best_partition(g)
		for node in self.input_graph.nodes():
			self.result_graph.nodes[node]['group'] = partition.get(node)
		self.partition = [partition.get(node) for node in self.input_graph.nodes()]

	def solve (self, method="leuven", attacking = 'source', weights_occ = False): # get partition
		print ('*********** SOLVING ************')
		if method == 'leuven':
			self.partition_leuven()
		elif method == 'namespace':
			self.partition_namespace()
		elif method == "smt":
			print ('\n\nsolving using smt')
			# resulting graph
			self.result_graph = nx.Graph()
			self.result_graph.add_nodes_from(self.input_graph.nodes())

			# encode the existing graph with weight 1
			o = Optimize()
			timeout = 1000 * 60 * 2 # depending on the size of the graph
			o.set("timeout", timeout)
			print('timeout = ',timeout/1000/60, 'mins')
			encode = {}
			soft_clauses = {}

			# STEP 1: the input graph (nodes, edges and their weights)
			print ('STEP 1: the input graph (nodes, edges and their weights)')
			weight_input = 1
			max_clusters = 10
			count_weighted_edges = 0

			encode_id = 1
			for n in self.input_graph.nodes():
				encode[n] = Int(str(encode_id))
				encode_id += 1
				o.add(encode[n] > Int(0))
				o.add(encode[n] < Int(max_clusters))


			for (left, right) in self.input_graph.edges():
				clause = (encode[left] == encode[right])
				if weights_occ == False:
					soft_clauses[clause] = weight_input
				else:
					# otherwise, use the weights from the
					w = self.input_graph.edges[left, right]['weight']
					# print ('I have weight',w)
					if w != None:
						if w == 0:
							soft_clauses[clause] = weight_input
						else:
							soft_clauses[clause] = w
							count_weighted_edges += 1
					else:
						print ('weighting error?!')
			print ('count_weighted_edges = ', count_weighted_edges)

			# STEP 2: the attacking edges
			print ('STEP 2: the attacking edges')
			if attacking == 'namespace':
				count_attacking = 0
				# add attacking edges: namespace
				weight_namespace = 2
				for (left, right) in self.namespace_graph.edges():

					if (left, right) not in self.redirect_graph and (left, right) not in self.encoding_equality_graph:
						clause = Not(encode[left] == encode[right])
						if clause in soft_clauses.keys():
							soft_clauses[clause] += weight_namespace
						else:
							soft_clauses[clause] = weight_namespace
						count_attacking += 1
				print ('count namesapce soft clauses = ', count_attacking)
			elif attacking == 'source':
				# add attacking edges: typeA
				count_attacking = 0
				weight_typeA = 1
				for (left, right) in self.typeA_graph.edges():

					if (left, right) not in self.redirect_graph and (left, right) not in self.encoding_equality_graph:
						# print (left, '-- A --', right)
						clause = Not(encode[left] == encode[right])
						if clause in soft_clauses.keys():
							soft_clauses[clause] += weight_typeA
						else:
							soft_clauses[clause] = weight_typeA
						count_attacking += 1
				print ('type A: count soft clauses = ', count_attacking)

				# add attacking edges: typeB
				count_attacking = 0
				weight_typeB = 1
				for (left, right) in self.typeB_graph.edges():

					if (left, right) not in self.redirect_graph and (left, right) not in self.encoding_equality_graph:
						# print (left, '-- B --', right)
						clause = Not(encode[left] == encode[right])
						if clause in soft_clauses.keys():
							soft_clauses[clause] += weight_typeB
						else:
							soft_clauses[clause] = weight_typeB
						count_attacking += 1
				print ('type B: count soft clauses = ', count_attacking)

				# add attacking edges: typeC
				count_attacking = 0
				weight_typeC = 1
				for (left, right) in self.typeC_graph.edges():

					if (left, right) not in self.redirect_graph and (left, right) not in self.encoding_equality_graph:
						# print (left, '-- C --', right)
						clause = Not(encode[left] == encode[right])
						if clause in soft_clauses.keys():
							soft_clauses[clause] += weight_typeC
						else:
							soft_clauses[clause] = weight_typeC
						count_attacking += 1
				print ('type C: count soft clauses = ', count_attacking)

				for clause in soft_clauses.keys():
					# print ('clause = ', clause)
					o.add_soft(clause, soft_clauses[clause])

			else:
				print ('prameter error')

			# STEP 3: the confirming edges
			# add confirming edges: encoding equivalence
			weight_encoding_equivalence = 5
			for (left, right) in self.encoding_equality_graph.edges():
				clause = (encode[left] == encode[right])
				if clause in soft_clauses.keys():
					soft_clauses[clause] += weight_encoding_equivalence
				else:
					soft_clauses[clause] = weight_encoding_equivalence


			# add confirming edges: redirect
			redi_undirected = self.redirect_graph.copy()
			redi_undirected = redi_undirected.to_undirected()
			print ('num of edges in undirected graph', len (redi_undirected.edges()))
			weight_redirect = 5
			number_redi_confirming_edges = 0
			for (left, right) in self.input_graph.edges():
				if left in redi_undirected.nodes() and right in redi_undirected.nodes():
					if nx.has_path(redi_undirected, left, right):
						clause = (encode[left] == encode[right])
						if clause in soft_clauses.keys():
							soft_clauses[clause] += weight_redirect
						else:
							soft_clauses[clause] = weight_redirect
						number_redi_confirming_edges += 1
					else:
						pass
						# print ('both in redi graph but not connected: ', left, right)
				else:
					pass # at least one of them is not in the redirect graph. so no need for checking
			print ('number of confirming edges added from the redirecting graph ', number_redi_confirming_edges)

			smt_result = o.check()
			print ('the SMT result is ', smt_result)
			# smt_result = o.maximize()
			if smt_result == 'unknown':
				print ('What!!!')
			else:
				# print ('start decoding')
				# print ('>encode length ', len(encode.keys()))
				identified_edges = []
				m = o.model()
				for arc in self.input_graph.edges():
					(left, right) = arc
					if m.evaluate(encode[left] == encode[right]) == False:
						identified_edges.append(arc)
					elif m.evaluate((encode[left] == encode[right])) == True:
						self.result_graph.add_edge(left, right)
					else:
						print ('error in decoding!')

			print ('After solving,,, there are ', len (identified_edges), ' removed')
			print ('After solving,,, there are ', len (self.result_graph.edges()), ' remaining edges')
			print ('After solving,,, there are ', len (self.result_graph.nodes()), ' remaining nodes')

			# compute the connected components (for undirected graphs)
			comps  = nx.connected_components(self.result_graph)
			for c in comps:
				if (len (c) == 1):
					print ('component: ', c)
			comps  = nx.connected_components(self.result_graph)
			print ('size list = ',[len(c) for c in sorted(comps, key=len, reverse=True)])

			# Finally, compute the partitions
			self.partition = None


	def test_violates_iUNA(self, context='namespace', exception = 'encoding_equivalence'):
		if context=='namespace':

			count_unknown = 0
			count_keep = 0
			count_remove = 0

			count_keep_violates_iUNA = 0
			count_keep_follows_iUNA = 0
			count_keep_encoding_equivalence = 0

			count_remove_violates_iUNA = 0
			count_remove_follows_iUNA = 0
			count_remove_encoding_equivalence = 0

			for (s,t) in self.gold_standard_graph.edges():
				if self.gold_standard_graph.edges[s, t]['decision'] == KEEP:
					count_keep += 1
					s_prefix = get_namespace(s)
					t_prefix = get_namespace(t)

					if s_prefix != None and t_prefix != None and s_prefix == t_prefix :
						count_keep_violates_iUNA += 1
						if exception == 'encoding_equivalence':
							if (s,t) in self.encoding_equality_graph.edges():
								count_keep_encoding_equivalence += 1

					if s_prefix != None and t_prefix != None and s_prefix != t_prefix:
						count_keep_follows_iUNA += 1

				elif self.gold_standard_graph.edges[s, t]['decision'] == UNKNOWN:
					count_unknown += 1
				elif self.gold_standard_graph.edges[s, t]['decision'] == REMOVE:
					count_remove += 1

					s_prefix = get_namespace(s)
					t_prefix = get_namespace(t)

					if s_prefix != None and t_prefix != None and s_prefix == t_prefix:
						count_remove_violates_iUNA += 1
						if exception == 'encoding_equivalence':
							if (s,t) in self.encoding_equality_graph.edges():
								count_remove_encoding_equivalence += 1

					if s_prefix != None and t_prefix != None and s_prefix != t_prefix:
						count_remove_follows_iUNA += 1


		print ('count unknown: ', count_unknown)

		print ('count keep: ', count_keep)
		print ('\t violating iUNA-namespace: ', count_keep_violates_iUNA)
		print ('\t\t considering encoding equivalence: -',count_keep_encoding_equivalence)
		print ('\t following iUNA-namespace: ', count_keep_follows_iUNA)


		print ('count remove: ', count_remove)
		print ('\t violating iUNA-namespace: ', count_remove_violates_iUNA)
		print ('\t\t considering encoding equivalence: -', count_remove_encoding_equivalence)
		print ('\t following iUNA-namespace: ', count_remove_follows_iUNA)

		print ('when not considering encoding equivalence:', count_keep_follows_iUNA + count_remove_violates_iUNA, ' : ', count_keep_violates_iUNA + count_remove_follows_iUNA)
		ratio1 = (count_keep_follows_iUNA + count_remove_violates_iUNA) / ( count_keep_follows_iUNA + count_remove_violates_iUNA + count_keep_violates_iUNA + count_remove_follows_iUNA)
		print ('the ratio of aligned v.s. against = ', ratio1)

		print ('\nwhen taking encoding equivalence into consideration', count_keep_follows_iUNA + count_remove_violates_iUNA - count_remove_encoding_equivalence  + count_keep_encoding_equivalence, ' : ', count_keep_violates_iUNA + count_remove_follows_iUNA - count_keep_encoding_equivalence + count_remove_encoding_equivalence)
		ratio2 = (count_keep_follows_iUNA + count_remove_violates_iUNA - count_remove_encoding_equivalence  + count_keep_encoding_equivalence) / (count_keep_follows_iUNA + count_remove_violates_iUNA  + count_keep_violates_iUNA + count_remove_follows_iUNA )
		print ('the ratio of aligned v.s. against = ', ratio2)

		return (ratio1, ratio2)

# def visualize(self):
# 	nt = Network('700px', '700px')
# 	nt.from_nx(self.input_graph.graph)
# 	# nt.enable_physics(True)
# 	nt.show_buttons(filter_=['physics'])
# 	nt.show('input.html')
#
# main
# graph_ids = [11116, 240577, 395175, 14514123]
# graph_ids = [11116]
# ratio1_avg = 0
# ratio2_avg = 0

# for graph_id in graph_ids:
# 	# graph_id = '11116'
# 	print ('\n\n\ngraph id = ', str(graph_id))
# 	gs = GraphSolver(path_to_input_graph = './Evaluate_May/' + str(graph_id) + '_edges_original.csv',
# 					path_to_gold_standard_graph = './Evaluate_May/'  + str(graph_id) + '_nodes_labelled.tsv')
#
# 	print (nx.info(gs.input_graph))
#
# 	print ('\ncomputing the encoding equivalence graph')
# 	# attacking edges:
# 	gs.get_namespace_graph()
# 	# confirming edges:
# 	gs.get_redirect_graph()
# 	gs.get_encoding_equality_graph()
#
# 	print ('\nNow work on the validation')
# 	#---test how many violates the iUNA when context = namespace
# 	ratio1,  ratio2 = gs.test_violates_iUNA (context='namespace', exception = 'encoding_equivalence')
# 	ratio1_avg += ratio1
# 	ratio2_avg += ratio2
#
# ratio1_avg = ratio1_avg/len(graph_ids)
# ratio2_avg = ratio2_avg/len(graph_ids)
#
# print ('ratio 1 avg = ', ratio1_avg)
# print ('ratio 2 avg = ', ratio2_avg)
#
#


# graph_ids = [11116]
graph_ids = [ 11116, 240577, 395175, 14514123]

for graph_id in graph_ids:
	# graph_id = '11116'
	print ('\n\n\ngraph id = ', str(graph_id))
	gs = GraphSolver(path_to_input_graph = str(graph_id) + '/' + str(graph_id) + '_edges_original.csv',
					path_to_gold_standard_graph = str(graph_id) + '/' + str(graph_id) + '_nodes_labelled.tsv',
					path_metalink_id = str(graph_id) + '/' + str(graph_id) + '_metalink_id.csv')

	print (nx.info(gs.input_graph), '\n\n')

	path_to_weights = str(graph_id) + '/' + str(graph_id) + '_sum_weight.hdt'
	gs.add_weight(path_to_weights)
	# print ('\ncomputing the encoding equivalence graph')
	# attacking edges:
	gs.get_namespace_graph()
	# confirming edges:
	path_to_redirect_graph = str(graph_id) + '/' + str(graph_id) + '_redirect.hdt'
	gs.get_redirect_graph(path_to_redirect_graph)

	gs.get_encoding_equality_graph()

	typeA_filename = str(graph_id) + '/' + str(graph_id) + "_explicit_source.hdt"
	typeB_filename = str(graph_id) + '/' + str(graph_id) + "_implicit_label_source.hdt"
	typeC_filename = str(graph_id) + '/' + str(graph_id) + "_implicit_comment_source.hdt"


	gs.get_typeA_graph(typeA_filename)
	gs.get_typeB_graph(typeB_filename)
	gs.get_typeC_graph(typeC_filename)

	gs.solve(method = 'smt', weights_occ = True)
	# gs.show_result_graph()


# --
# gs.get_encoding_equality_graph()
# gs.get_redirect_graph()
# gs.get_namespace_graph()
# gs.get_typeA_graph()
# gs.get_typeB_graph()
# gs.get_typeC_graph()
# gs.add_redundency_weight()

# -- visualization --
# gs.show_input_graph()
# gs.show_redirect_graph()
# gs.show_encoding_equivalence_graph()
# gs.show_namespace_graph()


# -- solve --

# -- show result --
# gs.show_gold_standard_graph()
