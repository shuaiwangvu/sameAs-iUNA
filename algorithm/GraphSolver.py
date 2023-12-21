# this is the class where the solver is defined
# the solver takes advantage of Z3, an SMT solver.

import networkx as nx
import networkx.algorithms.community as nx_comm
from cdlib import algorithms

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
from SameAsEqGraph import *
from extend_metalink import *
import csv
import random
import time
# import copy
from math import exp
import numpy as np

SMT_UNKNOWN = 0

GENERAL = 0
EFFECIENT = 1
FINETUNNED = 2


WITH_WEIGHT = False
WITH_DISAMBIG = False

# which_source = 'implicit_label_source'
# which_source = 'implicit_comment_source'
# ===================

UNKNOWN = 0
REMOVE = 1
KEEP = 2

hdt_source = None
hdt_label = None
hdt_comment = None
hdt_disambiguation = None

NOTFOUND = 1
NOREDIRECT = 2
ERROR = 3
TIMEOUT = 4
REDIRECT = 5

# qUNA is restricted to the following namespaces
restricted_prefix_list = ["http://dblp.rkbexplorer.com/id/",
"http://dbpedia.org/resource/",
"http://rdf.freebase.com/ns/m/",
"http://sws.geonames.org/",
"http://dbtune.org/musicbrainz/resource/",
"http://bio2rdf.org/uniprot:"]

debug = False

# define the class of graph solver
class GraphSolver():

	def __init__(self, dir, graph_id, weighting_scheme = 'w1', source = 'implicit_label_source'):
		# input graph
		# self.input_graph = nx.Graph()
		self.gold_standard_partition = []
		self.source_switch = source # by default the source is label
													# 'implicit_comment_source'
		path_to_nodes = dir + str(graph_id) +'.tsv'
		# path_to_edges = dir + str(graph_id) +'_edges.tsv'
		path_to_edges = dir + str(graph_id) +'_edges_with_Metalink_edge_id_and_error_degree.tsv'

		self.input_graph = load_undi_graph(path_to_nodes, path_to_edges)
		# self.input_graph = nx.Graph(self.input_graph)
		if debug:
			print ('number of nodes', self.input_graph.number_of_nodes())
			print ('number of edges ', self.input_graph.number_of_edges())
		self.error_edges_by_gold_standard = []
		for e in self.input_graph.edges():
			(s, t) = e
			if self.input_graph.nodes[s]['annotation'] != 'unknown' and self.input_graph.nodes[t]['annotation'] != 'unknown':
				if self.input_graph.nodes[s]['annotation'] != self.input_graph.nodes[t]['annotation']:
					self.error_edges_by_gold_standard.append(e)
		if debug:
			if len (self.error_edges_by_gold_standard) == 0:
				print ('Based on the gold standard, there is no edge that is erronous')
			else:
				print ('Based on the gold standard, there are at least', len (self.error_edges_by_gold_standard), ' error edges')
		# for visulization
		self.position = nx.spring_layout(self.input_graph)

		# for visulization together with redirect graph and ee graph
		# big = self.input_graph.copy()
		# big.add_nodes_from(self.redirect_graph.nodes())
		# big.add_edges_from(self.redirect_graph.edges())
		# big.add_nodes_from(self.encoding_equality_graph.nodes())
		# big.add_edges_from(self.encoding_equality_graph.edges())
		# self.position = nx.spring_layout(big)
		# self.map_color = {}

		# additional information
		path_to_redi_graph_nodes = dir + str(graph_id) +'_redirect_nodes.tsv'
		path_to_redi_graph_edges = dir + str(graph_id) +'_redirect_edges.nt'
		self.redirect_graph = load_redi_graph(path_to_redi_graph_nodes, path_to_redi_graph_edges)
		self.redi_undirected = self.redirect_graph.copy()
		self.redi_undirected = self.redi_undirected.to_undirected()


		path_ee = dir + str(graph_id) + '_encoding_equivalent.hdt'
		self.encoding_equality_graph = load_encoding_equivalence(path_ee)
		# print ('[ee] number of nodes', self.encoding_equality_graph.number_of_nodes())
		# print ('[ee] number of edges', self.encoding_equality_graph.number_of_edges())
		# for e in self.encoding_equality_graph.edges():
		# 	print ('found encoding equivalence: ',e)

		# path_to_explicit_source = dir + str(graph_id) + "_explicit_source.hdt"
		# load_explicit(path_to_explicit_source, self.input_graph)
		path_to_implicit_label_source = dir + str(graph_id) + "_implicit_label_source.hdt"
		# print ('path_to_implicit_label_source = ', path_to_implicit_label_source)
		load_implicit_label_source(path_to_implicit_label_source, self.input_graph)

		path_to_implicit_comment_source = dir + str(graph_id) + "_implicit_comment_source.hdt"
		# print ('path_to_implicit_comment_source = ', path_to_implicit_comment_source)
		load_implicit_comment_source(path_to_implicit_comment_source, self.input_graph)

		# load the weight graph
		path_to_edge_weights = dir + str(graph_id) + "_weight.tsv"
		load_edge_weights (path_to_edge_weights, self.input_graph)

		# solving: (weighted unique name constraints, from UNA)
		# self.positive_UNC = [] # redirect and equivalence encoding
		# self.negative_UNC = [] # namespace or source
		self.attacking_edges = []
		# result
		self.removed_edges = set()
		self.entity_to_partition = {}
		self.partition_to_entities = {}

		self.result_graph = None

		if weighting_scheme == 'w1': # corresponds to w1 in the paper
			 # for the remaining edges
			# print ('using w1')
			self.default_weight = 5
			self.weight_iUNA_uneq_edge = 2 #17 # weight of attacking edge
			self.weight_encoding_equivalence = 1 #5
			self.weight_redirect = 1 #5
		elif weighting_scheme == 'w2':  # corresponds to w2 in the paper
			# print ('using w2')
			self.default_weight = 31
			self.weight_iUNA_uneq_edge = 16 # weight of attacking edge
			self.weight_encoding_equivalence = 5
			self.weight_redirect = 5
		elif weighting_scheme == 'w3':  # corresponds to w2 in the paper
			# print ('using w2')
			self.default_weight = 10
			self.weight_iUNA_uneq_edge = 3 # weight of attacking edge
			self.weight_encoding_equivalence = 2
			self.weight_redirect = 5
		elif weighting_scheme == 'w4':  # corresponds to w2 in the paper
			# print ('using w2')
			self.default_weight = 20
			self.weight_iUNA_uneq_edge = 3 # weight of attacking edge
			self.weight_encoding_equivalence = 4
			self.weight_redirect = 3
		elif weighting_scheme == 'w5':  # corresponds to w2 in the paper
			# print ('using w2')
			self.default_weight = 15
			self.weight_iUNA_uneq_edge = 6 # weight of attacking edge
			self.weight_encoding_equivalence = 5
			self.weight_redirect = 5
		else:
			print ('ERROR in MODE')

		# self.max_equivalent_classes = 2 + int(len(self.input_graph.nodes())/150)
		# attacking related
		self.beta = 0.5 #
		self.rate_for_remaining_other_edges = 0.12

		self.reduced_weight_disambiguation = 5
		self.increased_weight_geq_2 = 2

		# additional information

		# parameters about weights on the edges
		self.weights_occ = WITH_WEIGHT
		self.consider_disambiguation = WITH_DISAMBIG
		# self.alpha = 2 # soft_clauses[clause] = default_weight + (w * alpha)




	def show_input_graph(self):
		g = self.input_graph
		# sp = nx.spring_layout(g)
		nx.draw_networkx(g, pos=self.position, with_labels=False, node_size=25)
		plt.title('Input')
		plt.show()

	# display the graph without the removed edges
	def show_result_graph(self):
		g = self.result_graph
		# sp = nx.spring_layout(g)
		nx.draw_networkx(g, pos=self.position, with_labels=False, node_size=25)
		plt.title('Result')
		plt.show()

	# plot the redirect graph
	def show_redirect_graph(self):
		print ('\n\n <<< Getting redirect graph >>>')
		g = self.redirect_graph
		# sp = nx.spring_layout(g)
		# nx.draw_networkx(g, pos=self.position, with_labels=False, node_size=35)

		nx.draw_networkx(self.input_graph, pos=self.position, with_labels=False, node_size=25)
		nx.draw_networkx_edges(g, pos=self.position, edge_color='red', connectionstyle='arc3,rad=0.2')

		plt.title('Redirect')
		plt.show()

	# display the graph with edges corresponding to encoding equivalence
	def show_encoding_equivalence_graph(self):
		g = self.encoding_equality_graph
		# print ('now plot a graph with ', len (g.edges()), ' equivalence edges')
		# sp = nx.spring_layout(g)
		# nx.draw_networkx(g, pos=self.position, with_labels=False, node_size=35)

		nx.draw_networkx(self.input_graph, pos=self.position, with_labels=False, node_size=25)
		nx.draw_networkx_edges(g, pos=self.position, edge_color='blue', connectionstyle='arc3,rad=0.2')

		plt.title('Encoding Equivalence')
		plt.show()

	# not in use anymore
	def show_namespace_graph(self):
		g = self.namespace_graph
		# sp = nx.spring_layout(g)
		# nx.draw_networkx(g, pos=self.position, with_labels=False, node_size=35)

		nx.draw_networkx(self.input_graph, pos=self.position, with_labels=False, node_size=25)
		nx.draw_networkx_edges(g, pos=self.position, edge_color='blue', connectionstyle='arc3,rad=0.2')

		plt.title('Namesapce Attacking edges')
		plt.show()

	# display the graph corresponding to the gold standard.
	# nodes annotated as ''unknown'' are colored in yellow
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

	# Use the Louvain algorithm for graph partitioning.
	# It takes a prameter resolution.
	# In our experiment, we take two parameters: 1.0 and 0.01
	def partition_louvain(self, res):
		g = self.input_graph
		self.result_graph = nx.Graph()
		self.result_graph.add_nodes_from(self.input_graph.nodes())
		self.removed_edges = set()

		# partition = nx_comm.louvain_communities(g, resolution = 0.0002, threshold = 1e-07)
		# coms = algorithms.louvain(g,  resolution=0.0001, randomize=False)
		# coms = algorithms.louvain(g,  resolution=0.0001, randomize=False)
		coms = algorithms.louvain(g, resolution=res, randomize=True)
		# coms = algorithms.louvain(g, weight='weight', resolution=1, randomize=True)
		# coms = algorithms.leiden(g)
		partition = coms.communities

		# update the result graph, and get the removed edges
		if debug:
			print ('There are in total ', len (partition),' partitions')
			print ('There are in total ', g.number_of_nodes(), ' nodes')
			print ('There are in total ', g.number_of_edges(), ' edges')
		for (i, p) in enumerate(partition):
			# for each community, we record it's node
			self.partition_to_entities [i] = list (p)

			# record that of each node
			for n in p:
				self.result_graph.nodes[n]['group'] = i
				self.entity_to_partition[n] = i


		for (s, t) in self.input_graph.edges():
			if self.result_graph.nodes[s]['group'] == self.result_graph.nodes[t]['group']:
				self.result_graph.add_edge(s, t)
			else:
				self.removed_edges.add((s,t))
		self.removed_edges = list (set(self.removed_edges))
		if debug:
			print ('edges removed ', len(self.removed_edges))


	# The Leiden algorithm is an imporove version of the
	# Louvain algorithm.
	# Our analysis shows that it is better for the detection of
	# communities. The resulting partitions are less likely to have
	# singltons (a subgraph with only one node).
	def partition_leiden(self):
		g = self.input_graph
		self.result_graph = nx.Graph()
		self.result_graph.add_nodes_from(self.input_graph.nodes())
		self.removed_edges = set()

		# partition = nx_comm.louvain_communities(g, resolution = 0.0002, threshold = 1e-07)
		# coms = algorithms.louvain(g,  resolution=0.0001, randomize=False)
		coms = algorithms.leiden(g)
		partition = coms.communities

		# update the result graph, and get the removed edges
		if debug:
			print ('There are in total ', len (partition),' partitions')
			print ('There are in total ', g.number_of_nodes(), ' nodes')
			print ('There are in total ', g.number_of_edges(), ' edges')
		for (i, p) in enumerate(partition):
			# for each community, we record it's node
			self.partition_to_entities [i] = list (p)

			# record that of each node
			for n in p:
				self.result_graph.nodes[n]['group'] = i
				self.entity_to_partition[n] = i

		for (s, t) in self.input_graph.edges():
			if self.result_graph.nodes[s]['group'] == self.result_graph.nodes[t]['group']:
				self.result_graph.add_edge(s, t)
			else:
				self.removed_edges.add((s,t))
		self.removed_edges = list (set(self.removed_edges))
		if debug:
			print ('edges removed ', len(self.removed_edges))

	# A simple method that takes a threshold and remove any edge that
	# has an error degree higher than the threshold
	# No refinement is included in this method
	def partition_metalink (self, threshold):
		# metalink_error_rate
		g = self.input_graph
		self.result_graph = nx.Graph()
		self.result_graph.add_nodes_from(self.input_graph.nodes())
		self.removed_edges = set()

		# self.partition_to_entities [i] = list (p)
		count_removed = 0
		for e in self.input_graph.edges():
			error_rate = float(self.input_graph.edges[e]['metalink_error_rate'])
			if error_rate <= threshold:
				(s,t) = e
				self.result_graph.add_edge(s,t)
			else:
				self.removed_edges.add((s,t))
				count_removed += 1

		# if debug:
		self.removed_edges = list (set(self.removed_edges))
		print ('edges removed ', len(self.removed_edges))
		# print ('[set]edges removed ', len(set(self.removed_edges)))

	# the method is used in the validation of UNAs
	# It samples pairs according to the beta parameters
	# and return a list of pairs that violates the qUNA
	def get_pairs_from_qUNA(self, method, graph, beta):
		# print ('graph.number_of_nodes = ', graph.number_of_nodes())
		# print ('exp(-1*graph.number_of_nodes()/2500) = ', exp(-1*graph.number_of_nodes()/2500))
		pairs = set()


		namespace_to_entities = {}
		for n in graph.nodes():
			p = get_prefix(n)
			if p in namespace_to_entities.keys():
				namespace_to_entities[p].append(n)
			else:
				namespace_to_entities[p] = [n]

		for p in namespace_to_entities.keys():
			j = len(namespace_to_entities[p])

			if j > 1 and len (pairs) < graph.number_of_nodes():
				num_to_try = int(j*(j-1)/2 * self.beta)

				tried = set()
				# print ('j', j)
				# print ('int(j*(j-1)/2 = ', int(j*(j-1)/2))
				# print ('num_to_try = ', num_to_try)
				# print ('#nodes ', graph.number_of_nodes())
				if num_to_try == 0:
					num_to_try = 1
				# print ('j = ', j, ' num try = ', num_to_try)
				for i in range(num_to_try):
					[left, right] = random.choices(namespace_to_entities[p], k=2)
					while (left, right) in tried:
						[left, right] = random.choices(namespace_to_entities[p], k=2)
					if self.violates_qUNA(left, right) and left != right:
						pairs.add((left, right))
					tried.add((left, right))

		# print ('paris generated : ', len (pairs))
		return list(pairs)

	# the method is used in the validation of UNAs
	# It samples pairs according to the beta parameters
	# and return a list of pairs that violates the nUNA
	def get_pairs_from_nUNA(self, method, graph, beta):
		# print ('graph.number_of_nodes = ', graph.number_of_nodes())
		# print ('exp(-1*graph.number_of_nodes()/2500) = ', exp(-1*graph.number_of_nodes()/2500))
		pairs = set()

		namespace_to_entities = {}
		for n in graph.nodes():
			p = get_prefix(n)
			if p in namespace_to_entities.keys():
				namespace_to_entities[p].append(n)
			else:
				namespace_to_entities[p] = [n]

		for p in namespace_to_entities.keys():
			j = len(namespace_to_entities[p])

			if j > 1 and len (pairs) < graph.number_of_nodes():
				num_to_try = int(j*(j-1)/2 * self.beta)

				tried = set()
				# print ('j', j)
				# print ('int(j*(j-1)/2 = ', int(j*(j-1)/2))
				# print ('num_to_try = ', num_to_try)
				if num_to_try == 0:
					num_to_try = 1
				# print ('j = ', j, ' num try = ', num_to_try)
				for i in range(num_to_try):
					[left, right] = random.choices(namespace_to_entities[p], k=2)
					while (left, right) in tried:
						[left, right] = random.choices(namespace_to_entities[p], k=2)
					if self.violates_nUNA(left, right) and left != right:
						pairs.add((left, right))
					tried.add((left, right))

		# print ('paris generated : ', len (pairs))
		return list(pairs)

	# the method is used in the validation of UNAs
	# It samples pairs according to the beta parameters
	# and return a list of pairs that violates the iUNA
	def get_pairs_from_iUNA(self, method, graph, beta):
		# print ('graph.number_of_nodes = ', graph.number_of_nodes())
		# print ('exp(-1*graph.number_of_nodes()/2500) = ', exp(-1*graph.number_of_nodes()/2500))
		pairs = set()
		if method == 'existing_edges':
			for e in graph.edges():
				(left, right) = e
				if self.violates_iUNA(left, right):
					pairs.add((left, right))
			if debug:
				print ('paris generated : ', len (pairs))
			return pairs
		elif method == 'generated_pairs':

			namespace_to_entities = {}
			for n in graph.nodes():
				p = get_prefix(n)
				if p in namespace_to_entities.keys():
					namespace_to_entities[p].append(n)
				else:
					namespace_to_entities[p] = [n]

			for p in namespace_to_entities.keys():
				j = len(namespace_to_entities[p])

				if j > 1 and len (pairs) < graph.number_of_nodes():
					num_to_try = int(j*(j-1)/2 * self.beta)

					tried = set()
					# print ('j', j)
					# print ('int(j*(j-1)/2 = ', int(j*(j-1)/2))
					# print ('num_to_try = ', num_to_try)
					if num_to_try == 0:
						num_to_try = 1
					# print ('j = ', j, ' num try = ', num_to_try)
					for i in range(num_to_try):
						[left, right] = random.choices(namespace_to_entities[p], k=2)
						while (left, right) in tried:
							[left, right] = random.choices(namespace_to_entities[p], k=2)
						if self.violates_iUNA(left, right) and left != right and len (pairs) < 2*graph.number_of_nodes():
							pairs.add((left, right))
						tried.add((left, right))
			if debug:
				print ('paris generated : ', len (pairs))
			return list(pairs)

	# In this method, we call the SMT solver
	# this method corresponds to the Algorith 1 in the paper.
	def solve_SMT (self, una): # gs.solve_SMT(una = selected_UNA, weighting_scheme = selected_weighting_scheme)
		collect_resulting_graphs = []
		collect_removed_edges = set()
		iter_result = self.solve_SMT_iter (self.input_graph, una)
		# if iter_result == SMT_UNKNOWN:
		# 	print ('not enough time for SMT, keep the graph as it is')
		# 	self.removed_edges = set()
		# 	self.result_graph = self.input_graph.copy()
		# 	return SMT_UNKNOWN
		# else:
		removed_edges, graphs = iter_result
		collect_removed_edges = collect_removed_edges.union(removed_edges)
		if len(removed_edges) == 0:
			self.result_graph = self.input_graph.copy()
			self.removed_edges = set()
		# print ('# first round removed edges = ', len(removed_edges))
		for g in graphs:
			if g.number_of_nodes()<=1:
				collect_resulting_graphs.append(g)
		graphs = list(filter(lambda x: x.number_of_nodes()>1, graphs))

		count_round = 1
		# print ('after round 1, there are still ', len(graphs), 'graphs. ')
		# print ('after round 1, there are ', len(collect_removed_edges), 'edges removed. ')
		count_correct = 0
		count_error = 0
		for (left,right) in collect_removed_edges:
			if self.input_graph.nodes[left]['annotation'] != 'unknown' and self.input_graph.nodes[right]['annotation'] != 'unknown':
				if self.input_graph.nodes[left]['annotation'] != self.input_graph.nodes[right]['annotation']:
					count_correct += 1
				else:
					count_error += 1
			# else:
			# 	print ('left : ', left)
			# 	print ('right: ', right)
		# print ('correct = ', count_correct)
		# print ('error = ', count_error)
		# if len(collect_removed_edges) != 0:
		# 	print ('precision now = ', count_correct/len(collect_removed_edges))

		condition = True
		if len (removed_edges) == 0:
			condition = False
			self.removed_edges = set()
			return None # finish here, no need to continue

		while condition:
			count_round += 1
			# print ('\n\nThis is round ', count_round)
			collect_graphs = []
			removed_edges = set()
			for g in graphs:
				iter_result = self.solve_SMT_iter(g, una)
				if iter_result != SMT_UNKNOWN:
					new_removed_edges, new_graphs = iter_result
					# print ('->removed ', len(new_removed_edges), ' edges')
					if len(new_removed_edges) > 0:
						removed_edges = removed_edges.union(new_removed_edges)
						if len (new_graphs) > 1: # stop condition: removed some edges but still connected
							collect_graphs += new_graphs
						else:
							collect_resulting_graphs += new_graphs
					else:
						collect_resulting_graphs += new_graphs
				else:
					collect_graphs.append(g) # give it a second chance

			collect_removed_edges = collect_removed_edges.union(removed_edges)

			# print ('before filtering, we have ', len (collect_graphs), ' graphs')
			# collect_resulting_graphs
			for g in collect_graphs:
				if g.number_of_nodes()<=1:
					collect_resulting_graphs.append(g)
			collect_graphs = list(filter(lambda x: x.number_of_nodes()>1, collect_graphs))
			# terminating condition:
			# 1) no edges removed: see above
			# 2) singleton graph
			# print ('after filtering, there are ', len(collect_graphs),  'graphs remaining')
			# for g in collect_graphs:
			# 	print ( '\t#nodes: ', g.number_of_nodes(), ' #edges: ', g.number_of_edges())

			if len (removed_edges) == 0 or len(collect_graphs) == 0:
				condition = False
			graphs = collect_graphs

		# print ('Overall, for this graph, we removed a total of ', len (collect_removed_edges))
		self.removed_edges = collect_removed_edges

		self.result_graph = self.input_graph.copy()
		self.result_graph.remove_edges_from(collect_removed_edges)

		# print (collect_resulting_graphs)
		sizes = [c.number_of_nodes() for c in collect_resulting_graphs]
		# print ('size list = ',[c for c in sorted(sizes, reverse=True)])

	# The method corresponds to Algorithm 2.
	# It recursively calls itself until it reaches a termination condition.
	def solve_SMT_iter (self, graph, una): # get partition
		if debug:
			print ('\n\nThis graph has ',graph.number_of_nodes(), ' nodes')
		# print( 'and ', graph.number_of_edges(), 'edges')
		# max_equivalent_classes = 2 + int(math.log10(len(graph.nodes())))
		max_equivalent_classes = 2 + int(len(graph.nodes())/50)
		if debug:
			print(' max equivalent classes: ', max_equivalent_classes)
		# print ('\n\nsolving using smt')
		# resulting graph
		result_graph = graph.copy()
		# result_graph.add_nodes_from(graph.nodes())

		# encode the existing graph with weight 1
		o = Optimize()
		timeout = int(1000 * 60 * (graph.number_of_nodes()/100 + 0.5)) # depending on the size of the graph
		# timeout = int(1000 * 60 * (graph.number_of_nodes()/20 + 0.2)) # depending on the size of the graph
		# timeout = int(1000 * 60 * 0.1) # depending on the size of the graph
		o.set("timeout", timeout)
		# print('timeout = ',timeout/1000, 'seconds')
		if debug:
			print('timeout = ',timeout/1000/60, 'mins')
		encode = {}
		soft_clauses = {}

		# STEP 1: the input graph (nodes, edges and their weights)
		# print ('STEP 1: the input graph (nodes, edges and their weights)')
		# default_weight = 35
		# reduced_weight_disambiguation = 1
		# self.max_equivalent_classes = 8 + int(len(graph.nodes())/150)
		# weights_occ = False

		# count_weighted_edges = 0

		encode_id = 1
		for n in graph.nodes():
			encode[n] = Int(str(encode_id))
			encode_id += 1
			o.add(encode[n] > Int(0))
			o.add(encode[n] < Int(max_equivalent_classes +1))

		count_ignored = 0

		t = nx.Graph(graph)
		to_remove = []
		# the method takes advantage of minimum spanning forest offered by networkx
		ms_edges = list(nx.minimum_spanning_edges(t, data= False))

		total_edges_considered = 0
		for (left, right) in graph.edges():
			ignore = False

			if (left, right) in ms_edges:
				ignore = False
			elif random.random () > (self.rate_for_remaining_other_edges):
				ignore = True
				count_ignored += 1

			if ignore == False:
				clause = (encode[left] == encode[right])
				total_edges_considered += 1
				soft_clauses[clause] = self.default_weight

				# if self.consider_disambiguation:
				# 	if left in self.dis_entities or right in self.dis_entities:
				# 		soft_clauses[clause] -= self.reduced_weight_disambiguation

				if self.weights_occ == True:
					# otherwise, use the weights from the
					w = graph.edges[left, right]['weight']
					# print ('!!!!!!! I have weight',w)
					if w != None:
						if w >= 2:
							soft_clauses[clause] += self.increased_weight_geq_2
						# else:
						# 	soft_clauses[clause] = self.default_weight
					else:
						print ('weighting error?!')



		# print ('count_weighted_edges = ', count_weighted_edges)
		# print ('count_ignored edges between DBpedia multilingual entities', count_ignored)
		# print ('total number of edges = ', total_edges_considered)
		# STEP 2: the attacking edges
		# print ('STEP 2: the attacking edges')
		if una == 'iUNA':
			uneq_pairs = self.get_pairs_from_iUNA(method = 'generated_pairs', graph=graph, beta = self.beta)
		elif una == 'qUNA':
			uneq_pairs = self.get_pairs_from_qUNA(method = 'generated_pairs', graph=graph, beta = self.beta)
		elif una == 'nUNA':
			uneq_pairs = self.get_pairs_from_nUNA(method = 'generated_pairs', graph=graph, beta = self.beta)

		if len (uneq_pairs) <= 1:
			removed_edges = set()
			resulting_graphs = [graph]
			return (removed_edges, resulting_graphs)

		# how many of these attacks are correct attacks ?
		count_correct_attack = 0
		count_mistaken_attack = 0
		for (left,right) in uneq_pairs:
			if graph.nodes[left]['annotation'] != 'unknown' and graph.nodes[right]['annotation'] != 'unknown':
				if graph.nodes[left]['annotation'] != graph.nodes[right]['annotation']:
					count_correct_attack += 1
				else:
					count_mistaken_attack += 1

		# print ('count  correct ', count_correct_attack, ' -> ', count_correct_attack/len(uneq_pairs))
		# print ('count mistaken ', count_mistaken_attack, ' -> ', count_mistaken_attack/len(uneq_pairs))
		# print ('This round, the weight is ', int(self.weight_iUNA_uneq_edge )) # * exp(-1*graph.number_of_nodes()/2500)))
		for (left,right) in uneq_pairs:
			clause = Not(encode[left] == encode[right])
			if clause in soft_clauses.keys():
				soft_clauses[clause] += int(self.weight_iUNA_uneq_edge) # * exp(-1*graph.number_of_nodes()/2500))
			else:
				soft_clauses[clause] = int(self.weight_iUNA_uneq_edge) # * exp(-1*graph.number_of_nodes()/2500))

		# STEP 3: the confirming edges
		# add confirming edges: encoding equivalence
		# weight_encoding_equivalence = 10
		number_ee_edges = 0
		for (left, right) in self.encoding_equality_graph.edges():
			if left in graph.nodes() and right in graph.nodes():
				clause = (encode[left] == encode[right])
				if clause in soft_clauses.keys():
					soft_clauses[clause] += self.weight_encoding_equivalence
				else:
					soft_clauses[clause] = self.weight_encoding_equivalence
				number_ee_edges +=1
		# print ('number of confirming edges added from the ee graph ', number_ee_edges)

		# add confirming edges: redirect
		# self.redi_undirected = self.redirect_graph.copy()
		# self.redi_undirected = redi_undirected.to_undirected()
		# print ('[Redirect] num of edges in undirected graph', len (redi_undirected.edges()))
		# weight_redirect = 10
		number_redi_confirming_edges = 0
		# for (left, right) in graph.edges():
		# 	if left in redi_undirected.nodes() and right in redi_undirected.nodes():
		for (left, right) in self.redi_undirected.edges():
			if left in graph.nodes() and right in graph.nodes():
				if nx.has_path(self.redi_undirected, left, right):
					clause = (encode[left] == encode[right])
					if clause in soft_clauses.keys():
						soft_clauses[clause] += self.weight_redirect
					else:
						soft_clauses[clause] = self.weight_redirect
					number_redi_confirming_edges += 1
				else:
					pass
					# print ('both in redi graph but not connected: ', left, right)
			else:
				pass # at least one of them is not in the redirect graph. so no need for checking
		# print ('number of confirming edges added from the redirecting graph ', number_redi_confirming_edges)


		# finally, add them to the solver.
		for clause in soft_clauses.keys():
			o.add_soft(clause, soft_clauses[clause])

		# Decode the result: the identified_edges (to remove)

		smt_result = o.check()
		removed_edges = set()
		if debug:
			print ('the SMT result is ', smt_result)
		if str(smt_result) == 'unknown':
			print ('unknown as the result (cannot guarantee the result to be optimal)!!!')
		# 	return SMT_UNKNOWN
		# else:
		m = o.model()
		for arc in graph.edges():
			(left, right) = arc
			if m.evaluate(encode[left] == encode[right]) == False:
				removed_edges.add(arc)
				result_graph.remove_edge(left, right)
			elif m.evaluate((encode[left] == encode[right])) == True:
				# result_graph.add_edge(left, right)
				pass
			else:
				print ('error in decoding!')
		successful_attacking_edges = []
		unsuccessful_attacking_edges = []
		for arc in uneq_pairs:
			(left, right) = arc
			l = m.evaluate(encode[left])
			r = m.evaluate(encode[right])

			if m.evaluate(encode[left] == encode[right]) == False:
				successful_attacking_edges.append(arc)
			elif m.evaluate((encode[left] == encode[right])) == True:
				unsuccessful_attacking_edges.append(arc)
			else:
				print ('error in decoding!')

		resulting_graphs = [result_graph.subgraph(c) for c in nx.connected_components(result_graph)]

		return (removed_edges, resulting_graphs)

	# compute the Omega measure
	def compute_omega(self):
		# first, compute the annotation -> entity relation
		annotation_to_entities = {}
		for n in self.input_graph.nodes():
			anno = self.input_graph.nodes[n]['annotation']
			if anno != 'unknown':
				if anno not in annotation_to_entities.keys():
					annotation_to_entities[anno] = [n] #
				else:
					annotation_to_entities[anno].append(n)
		# print ('annotations ', annotation_to_entities.keys())

		omega = 0
		for cc in nx.connected_components(self.result_graph):
			# for the entities with the same annotaiton (equivalent class within the cc)
			within_cc_annotation_to_entities = {}
			for n in cc:
				anno = self.input_graph.nodes[n]['annotation']
				if anno != 'unknown':
					if anno not in within_cc_annotation_to_entities.keys():
						within_cc_annotation_to_entities[anno] = [n] #
					else:
						within_cc_annotation_to_entities[anno].append(n)
			# for each equivalent class,
			for a in within_cc_annotation_to_entities.keys():
				Q = within_cc_annotation_to_entities[a]
				# C is the number of entities in cc without the unknown ones
				C = cc
				# O the number of entities with the same annotation
				O = annotation_to_entities[a]
				# print ('\nQ = ', len(Q))
				# print ('O = ', len(O))
				# print ('C = ', len(C))

				omega += (len(Q) * len(Q)/len(O) * len(Q)/len(C))/ self.input_graph.number_of_nodes()
		return omega

	def evaluate_partitioning_result(self):
		evaluation_result = {}
		evaluation_result['Omega'] = self.compute_omega()
		evaluation_result ['num_edges_removed'] = len(self.removed_edges)
		evaluation_result ['num_error_edges_gold_standard'] = len(self.error_edges_by_gold_standard)

		if len (self.removed_edges) == 0 :
			evaluation_result ['flag'] = 'invalid precision or recall'
			return evaluation_result
		if len (self.error_edges_by_gold_standard) == 0:
			evaluation_result ['flag'] = 'invalid precision or recall'
			return evaluation_result
		else:
			# print ('num of edges removed: ', len(self.removed_edges))

			count_removed_by_mistake = 0
			count_correctly_removed = 0
			count_removed_unknown = 0

			for e in self.removed_edges:
				(s, t) = e
				if self.input_graph.nodes[s]['annotation'] != 'unknown' and self.input_graph.nodes[t]['annotation'] != 'unknown':
					if self.input_graph.nodes[s]['annotation'] == self.input_graph.nodes[t]['annotation']:
						count_removed_by_mistake += 1
					else:
						count_correctly_removed += 1
				else:
					count_removed_unknown += 1

			# print ('\t #Correctly Removed = ', count_correctly_removed)
			# print ('\t #Mistakenly Removed = ', count_removed_by_mistake)
			# print ('\t # unknown = ', count_removed_unknown)

			# evaluation_result = {}
			evaluation_result ['num_edges_removed'] = len(self.removed_edges)

			evaluation_result ['flag'] = 'valid precision and recall'
			# print ('precision = ', count_correctly_removed/len (self.removed_edges))
			evaluation_result['precision'] = count_correctly_removed / len (self.removed_edges)
			# print ('recall = ', count_correctly_removed/len (self.error_edges))
			evaluation_result['recall'] = count_correctly_removed / len (self.error_edges_by_gold_standard)

			return evaluation_result

	def random_sample_error_rate(self):
		num_edges = self.input_graph.number_of_edges()
		# randomly sample num_edges (not reflexive) pairs
		sampled_pairs = []
		count_unknown = 0
		count_correct = 0
		count_error = 0
		while len(sampled_pairs) < num_edges:
			s = random.choice(list(self.input_graph.nodes()))
			t = random.choice(list(self.input_graph.nodes()))
			if s!=t and (s,t) not in sampled_pairs:
				sampled_pairs.append((s,t))
		for (s,t) in sampled_pairs:
			if self.input_graph.nodes[s]['annotation'] != 'unknown' and self.input_graph.nodes[t]['annotation'] != 'unknown':
				if self.input_graph.nodes[s]['annotation'] == self.input_graph.nodes[t]['annotation'] :
					count_correct += 1
				else:
					count_error += 1
			else:
				count_unknown += 1
		return (num_edges, count_correct, count_error)

	def random_sample_error_rate_UNA(self):
		num_edges = self.input_graph.number_of_edges()
		# randomly sample num_edges (not reflexive) pairs

		sampled_pairs = []
		count_unknown_nUNA = 0
		count_violates_nUNA = 0
		count_correct_nUNA = 0
		count_error_nUNA = 0

		while len(sampled_pairs) < num_edges:
			s = random.choice(list(self.input_graph.nodes()))
			t = random.choice(list(self.input_graph.nodes()))
			if s!=t and (s,t) not in sampled_pairs:
				sampled_pairs.append((s,t))
		for (s,t) in sampled_pairs:
			if self.violates_nUNA (s,t):
				count_violates_nUNA += 1
				if self.input_graph.nodes[s]['annotation'] != 'unknown' and self.input_graph.nodes[t]['annotation'] != 'unknown':
					if self.input_graph.nodes[s]['annotation'] != self.input_graph.nodes[t]['annotation'] :
						count_correct_nUNA += 1
					else:
						count_error_nUNA += 1
				else:
					count_unknown_nUNA += 1

		count_unknown_qUNA = 0
		count_violates_qUNA = 0
		count_correct_qUNA = 0
		count_error_qUNA = 0

		for (s,t) in sampled_pairs:
			if self.violates_qUNA (s,t):
				count_violates_qUNA += 1
				if self.input_graph.nodes[s]['annotation'] != 'unknown' and self.input_graph.nodes[t]['annotation'] != 'unknown':
					if self.input_graph.nodes[s]['annotation'] != self.input_graph.nodes[t]['annotation'] :
						count_correct_qUNA += 1
					else:
						count_error_qUNA += 1
				else:
					count_unknown_qUNA += 1

		count_unknown_iUNA = 0
		count_violates_iUNA = 0
		count_correct_iUNA = 0
		count_error_iUNA = 0

		for (s,t) in sampled_pairs:
			if self.violates_iUNA (s,t):
				count_violates_iUNA += 1
				if self.input_graph.nodes[s]['annotation'] != 'unknown' and self.input_graph.nodes[t]['annotation'] != 'unknown':
					if self.input_graph.nodes[s]['annotation'] != self.input_graph.nodes[t]['annotation'] :
						count_correct_iUNA += 1
					else:
						count_error_iUNA += 1
				else:
					count_unknown_iUNA += 1
		return (num_edges, count_violates_nUNA, count_correct_nUNA, count_error_nUNA, count_violates_qUNA, count_correct_qUNA, count_error_qUNA, count_violates_iUNA, count_correct_iUNA, count_error_iUNA)

	# test if two IRIs violates the nUNA
	def violates_nUNA (self, left, right):
		source_left =  self.input_graph.nodes[left][self.source_switch]
		source_right =  self.input_graph.nodes[right][self.source_switch]

		if len(set(source_left).difference(set(source_right))) > 0:
			return True
		else:
			return False

	# test if two IRIs violates the qUNA
	def violates_qUNA (self, left, right):
		source_left, _, _ = get_name(left)
		source_right, _,_ = get_name(right)

		if source_left == source_right:
			# if one is a redict of the other under DBpedia.org
			if 'http://dbpedia.org/resource' in source_left and 'http://dbpedia.org/resource' in source_right:
				if left in self.redi_undirected.nodes() and right in self.redi_undirected.nodes():
					if nx.has_path(self.redi_undirected, left, right):
						# print ('redi has path')
						return False
			return True
		else:
			return False

	# test if two IRIs violates the iUNA
	def violates_iUNA (self, left, right):
		source_left =  self.input_graph.nodes[left][self.source_switch]
		source_right =  self.input_graph.nodes[right][self.source_switch]


		if left in self.redirect_graph.nodes():
			if self.redirect_graph.nodes[left]['remark'] in ['Error', 'Timeout', 'NotFound', 'RedirectedUntilTimeout', 'RedirectedUntilError', 'RedirectedUntilNotFound']:
				return False
		if right in self.redirect_graph.nodes():
			if self.redirect_graph.nodes[right]['remark'] in ['Error', 'Timeout', 'NotFound', 'RedirectedUntilTimeout', 'RedirectedUntilError', 'RedirectedUntilNotFound']:
				return False

		prefix_left = get_prefix(left)
		prefix_right = get_prefix(right)

		if len(set(source_left).difference(set(source_right))) > 0 :

			if prefix_left != prefix_right:
				return False
			# test also if it is in redirected
			elif left in self.redi_undirected.nodes() and right in self.redi_undirected.nodes():
				if nx.has_path(self.redi_undirected, left, right):
					# print ('redi has path')
					return False
			elif left in self.encoding_equality_graph.nodes() and right in self.encoding_equality_graph.nodes():
				if nx.has_path(self.encoding_equality_graph, left, right):
					# print ('ee chas path')c
					return False

			# print ('violates iUNA')
			return True

		return False
