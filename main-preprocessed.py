# this is the class where SameAsEqSolver is defined
import networkx as nx
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
from SameAsEqGraph import *
from extend_metalink import *
import csv
import random
import time
# import copy
from math import exp

SMT_UNKNOWN = 0
import numpy as np

GENERAL = 0
EFFECIENT = 1
FINETUNNED = 2


MODE = FINETUNNED  # FINETUNNED EFFECIENT
WITH_WEIGHT = False
WITH_DISAMBIG = False

which_source = 'implicit_label_source'
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


# there are in total 28 entities. 14 each
# the training set (for the validation of the method)
validation_single = [96073, 712342, 9994282, 18688, 1140988, 25604]
validation_multiple = [33122, 11116, 12745, 6617,4170, 42616, 6927, 39036]
validation_set = validation_single + validation_multiple
# the evaluation set
evaluation_single = [9411, 9756, 97757, 99932, 337339, 1133953]
evaluation_multiple = [5723, 14872, 37544, 236350, 240577, 395175, 4635725, 14514123]
evaluation_set = evaluation_single + evaluation_multiple


gs = validation_set + evaluation_set


hard_graphs = [6927, 37544, 4635725]

restricted_prefix_list = ["http://dblp.rkbexplorer.com/id/",
"http://dbpedia.org/resource/",
"http://rdf.freebase.com/ns/m/",
"http://sws.geonames.org/",
"http://dbtune.org/musicbrainz/resource/",
"http://bio2rdf.org/uniprot:"]



# define a class of graph solver
class GraphSolver():
	# configure the weight file
	# location of each node

	def __init__(self, dir, id):
		# input graph
		# self.input_graph = nx.Graph()


		path_to_nodes = dir + str(id) +'.tsv'
		path_to_edges = dir + str(id) +'_edges.tsv'
		self.input_graph = load_graph(path_to_nodes, path_to_edges)
		# print ('number of nodes', self.input_graph.number_of_nodes())
		# print ('number of (directed) edges ', self.input_graph.number_of_edges())
		self.input_graph = nx.Graph(self.input_graph)
		# print ('number of nodes', self.input_graph.number_of_nodes())
		# print ('number of (undirected) edges ', self.input_graph.number_of_edges())
		self.error_edges = []
		for e in self.input_graph.edges():
			(s, t) = e
			if self.input_graph.nodes[s]['annotation'] != 'unknown' and self.input_graph.nodes[t]['annotation'] != 'unknown':
				if self.input_graph.nodes[s]['annotation'] != self.input_graph.nodes[t]['annotation']:
					self.error_edges.append(e)
		# print ('There are ', len (self.error_edges), ' error edges')

		# additional information
		path_to_redi_graph_nodes = dir + str(id) +'_redirect_nodes.tsv'
		path_to_redi_graph_edges = dir + str(id) +'_redirect_edges.hdt'
		self.redirect_graph = load_redi_graph(path_to_redi_graph_nodes, path_to_redi_graph_edges)
		self.redi_undirected = self.redirect_graph.copy()
		self.redi_undirected = self.redi_undirected.to_undirected()
		# print ('[redirect] number of nodes', self.redirect_graph.number_of_nodes())
		# print ('[redirect] number of edges', self.redirect_graph.number_of_edges())
		# for e in self.redirect_graph.edges():
		# 	print ('redi edge : ', e)

		path_ee = dir + str(id) + '_encoding_equivalent.hdt'
		self.encoding_equality_graph = load_encoding_equivalence(path_ee)
		# print ('[ee] number of nodes', self.encoding_equality_graph.number_of_nodes())
		# print ('[ee] number of edges', self.encoding_equality_graph.number_of_edges())
		# for e in self.encoding_equality_graph.edges():
		# 	print ('found encoding equivalence: ',e)

		# for visulization
		big = self.input_graph.copy()
		big.add_nodes_from(self.redirect_graph.nodes())
		big.add_edges_from(self.redirect_graph.edges())
		big.add_nodes_from(self.encoding_equality_graph.nodes())
		big.add_edges_from(self.encoding_equality_graph.edges())
		self.position = nx.spring_layout(big)
		self.map_color = {}

		path_to_explicit_source = dir + str(graph_id) + "_explicit_source.hdt"
		load_explicit(path_to_explicit_source, self.input_graph)
		path_to_implicit_label_source = dir + str(graph_id) + "_implicit_label_source.hdt"
		load_implicit_label_source(path_to_implicit_label_source, self.input_graph)
		path_to_implicit_comment_source = dir + str(graph_id) + "_implicit_comment_source.hdt"
		load_implicit_comment_source(path_to_implicit_comment_source, self.input_graph)

		# load disambiguation entities
		# path_to_disambiguation_entities = "sameas_disambiguation_entities.hdt"
		# self.dis_entities = load_disambiguation_entities(self.input_graph.nodes(), path_to_disambiguation_entities)
		# print ('there are ', len (self.dis_entities), ' entities about disambiguation (in this graph)')

		# load the weight graph
		path_to_edge_weights = dir + str(graph_id) + "_weight.tsv"
		load_edge_weights (path_to_edge_weights, self.input_graph)


		# solving: (weighted unique name constraints, from UNA)
		# self.positive_UNC = [] # redirect and equivalence encoding
		# self.negative_UNC = [] # namespace or source
		self.attacking_edges = []
		# result
		self.removed_edges = []
		self.entity_to_partition = {}
		self.partition_to_entities = {}

		self.result_graph = None

		# the parameters
		# if MODE == GENERAL:
		# 	self.rate_for_remainging_other_edges = 1
		# 	self.default_weight = 3
		# 	# self.reduced_weight_disambiguation = 0
		# 	self.weight_iUNA_uneq_edge = 1 #17 # weight of attacking edge
		# 	self.weight_encoding_equivalence = 1 #5
		# 	self.weight_redirect = 1 #5
		# el
		if MODE == EFFECIENT: # corresponds to w1 in the paper
			 # for the remaining edges
			self.default_weight = 5
			self.weight_iUNA_uneq_edge = 1 #17 # weight of attacking edge
			self.weight_encoding_equivalence = 1 #5
			self.weight_redirect = 1 #5
		elif MODE == FINETUNNED:  # corresponds to w2 in the paper
			self.default_weight = 31
			self.weight_iUNA_uneq_edge = 16 # weight of attacking edge
			self.weight_encoding_equivalence = 5
			self.weight_redirect = 5
		else:
			print ('ERROR in MODE')


		# self.max_equivalent_classes = 2 + int(len(self.input_graph.nodes())/150)
		# attacking related
		self.beta = 0.50 #
		self.reduced_weight_disambiguation = 5
		self.increased_weight_geq_2 = 2
		self.rate_for_remainging_other_edges = 0.15

		# additional information

		# parameters about weights on the edges
		self.weights_occ = WITH_WEIGHT
		self.consider_disambiguation = WITH_DISAMBIG
		# self.alpha = 2 # soft_clauses[clause] = default_weight + (w * alpha)


	def show_input_graph(self):
		plt.figure(figsize=(10,5))
		g = self.input_graph
		# sp = nx.spring_layout(g)
		nx.draw_networkx(g, pos=self.position, with_labels=False, node_size=25)
		plt.title('Input Graph')
		plt.show()


	def show_redirect_graph(self):
		# print ('\n\n <<< Getting redirect graph >>>')
		g = self.redirect_graph
		plt.figure(figsize=(10,5))

		# sp = nx.spring_layout(g)
		# nx.draw_networkx(g, pos=self.position, with_labels=False, node_size=25)
		# print ('there are ', len (g.edges()), ' edges in the redirect graph')
		# edges = set(g.edges()).intersection(self.input_graph.edges)
		# print ('drawing', len (edges), ' edges')
		# nx.draw_networkx(self.input_graph, pos=self.position, with_labels=False, node_size=25)
		nx.draw_networkx_nodes(self.input_graph, pos=self.position, node_size=25)
		nx.draw_networkx_edges(self.input_graph, edgelist= self.input_graph.edges(), pos=self.position, node_size=25)
		nx.draw_networkx_nodes(g, pos=self.position, node_size=25)
		# nx.draw_networkx(g, pos=self.position, with_labels=False, node_size=25)
		nx.draw_networkx_edges(g, edgelist = g.edges(), pos=self.position, edge_color='red', connectionstyle='arc,rad=0.2')

		plt.title('Redirect')
		plt.show()

	def show_encoding_equivalence_graph(self):
		plt.figure(figsize=(10,5))
		g = self.encoding_equality_graph
		# print ('now plot a graph with ', len (g.edges()), ' equivalence edges')
		# sp = nx.spring_layout(g)
		# nx.draw_networkx(g, pos=self.position, with_labels=False, node_size=25)
		# print ('there are ', len (g.edges()), ' edges in the ee graph')
		# edges = set(g.edges()).intersection(self.input_graph.edges)
		# print ('drawing', len (edges), ' edges')
		nx.draw_networkx(self.input_graph, pos=self.position, with_labels=False, node_size=25)
		nx.draw_networkx(g, pos=self.position, with_labels=False, node_size=25)
		nx.draw_networkx_edges(g, pos=self.position, edge_color='blue', connectionstyle='arc3,rad=0.2')

		plt.title('Encoding Equivalence')
		plt.show()
	#
	# def show_namespace_graph(self):
	# 	g = self.namespace_graph
	# 	# sp = nx.spring_layout(g)
	# 	# nx.draw_networkx(g, pos=self.position, with_labels=False, node_size=25)
	#
	# 	nx.draw_networkx(self.input_graph, pos=self.position, with_labels=False, node_size=25)
	# 	nx.draw_networkx_edges(g, pos=self.position, edge_color='blue', connectionstyle='arc3,rad=0.2')
	#
	# 	plt.title('Namesapce Attacking edges')
	# 	plt.show()

	def show_gold_standard_graph (self):
		plt.figure(figsize=(10,5))
		g = self.input_graph
		annotations = set()
		for n in g.nodes:
			if g.nodes[n]['annotation'] != 'unknown':
				annotations.add(g.nodes[n]['annotation'])

		for a in annotations:
			color_r = random.randint(0,255)/255
			color_g = random.randint(0,255)/255
			color_b = random.randint(0,255)/255
			rgb = [color_r,color_g,color_b]
			self.map_color[a] = rgb
		# print (map_color)
		node_color = []
		for n in g.nodes:
			if g.nodes[n]['annotation'] == 'unknown':
				node_color.append('yellow')
			else:
				node_color.append(self.map_color[g.nodes[n]['annotation']])

		remaining_edges = []
		edge_colors = []
		for (l, r) in g.edges():
		# remaining_edges
			if  g.nodes[l]['annotation'] == 'unknown' or g.nodes[r]['annotation'] == 'unknown' :
				edge_colors = ['yellow']
				# remaining_edges.append((l,r))
			elif g.nodes[l]['annotation'] == g.nodes[r]['annotation']:
				edge_colors = self.map_color[g.nodes[l]['annotation']]
				remaining_edges.append((l,r))
			else:
				pass

		nx.draw_networkx_nodes(g, pos=self.position, node_size=25, node_color=node_color)
		nx.draw_networkx_edges(g, pos=self.position, edgelist = remaining_edges, node_size=25)
		# plt.axes('off')
		plt.title('Gold standard')
		plt.show()

	def show_result_graph(self):
		plt.figure(figsize=(10,5))
		g = self.result_graph
		# sp = nx.spring_layout(g)

		node_to_color = {}
		for cc in nx.connected_components(g):
			# connected components
			color_r = random.randint(0,255)/255
			color_g = random.randint(0,255)/255
			color_b = random.randint(0,255)/255
			rgb = [color_r,color_g,color_b]
			for n in cc:
				node_to_color[n] = rgb

		node_color = []
		for n in g.nodes():
			node_color.append(node_to_color[n])

		# print (node_color)
		# nx.draw_networkx(g, pos=self.position, with_labels=False, node_size=25)
		nx.draw_networkx_nodes(g, pos=self.position, node_size=25, node_color=node_color)
		nx.draw_networkx_edges(g, pos=self.position, edgelist = self.result_graph.edges())
		plt.title('Solving Result')
		plt.show()


	def partition_leuven(self):
		g = self.input_graph
		self.result_graph = nx.Graph()
		self.result_graph.add_nodes_from(self.input_graph.nodes())

		partition = community.best_partition(g)

		# update the result graph, and get the removed edges
		for n in self.input_graph.nodes():
			p = partition.get(n)
			self.result_graph.nodes[n]['group'] = p

			self.entity_to_partition[n] = p

			if p in self.partition_to_entities.keys():
				self.partition_to_entities [p].append(n)
			else:
				self.partition_to_entities [p]= [n]

		for (s, t) in self.input_graph.edges():
			if self.result_graph.nodes[s]['group'] == self.result_graph.nodes[t]['group']:
				self.result_graph.add_edge(s, t)
			else:
				self.removed_edges.append((s,t))


	def violates_nUNA (self, left, right):
		source_left =  self.input_graph.nodes[left][which_source]
		source_right =  self.input_graph.nodes[right][which_source]

		if len(set(source_left).difference(set(source_right))) > 0:
			return True
		else:
			return False


	def violates_qUNA (self, left, right):
		source_left =  self.input_graph.nodes[left][which_source]
		source_right =  self.input_graph.nodes[right][which_source]
		flag_left = False
		flag_right = False

		if len(set(source_left).difference(set(source_right))) > 0:
			if left in self.redirect_graph.nodes():
				if self.redirect_graph.nodes[left]['remark'] in ['Error', 'NotFound']:
					return False

			if right in self.redirect_graph.nodes():
				if self.redirect_graph.nodes[right]['remark'] not in ['Error', 'NotFound']:
					return False

			for p in restricted_prefix_list:
				if p in left:
					flag_left = True
					break

			for p in restricted_prefix_list:
				if p in right:
					flag_right = True
					break

			if flag_left and flag_right:
				if (left, right) not in self.redirect_graph.edges():
					if (right, left) not in self.redirect_graph.edges():
						return False
				return True

			return True

		return False

	def violates_iUNA (self, left, right):
		source_left =  self.input_graph.nodes[left][which_source]
		source_right =  self.input_graph.nodes[right][which_source]


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



	def get_pairs_from_iUNA(self, method, graph, beta):
		# print ('graph.number_of_nodes = ', graph.number_of_nodes())
		# print ('exp(-1*graph.number_of_nodes()/2500) = ', exp(-1*graph.number_of_nodes()/2500))
		pairs = set()
		# if method == 'existing_edges':
		# 	for e in graph.edges():
		# 		(left, right) = e
		# 		if self.violates_iUNA(left, right):
		# 			pairs.add((left, right))
		# 	print ('paris generated : ', len (pairs))
		# 	return pairs
		# elif method == 'generated_pairs':

		prefix_to_entities = {}
		for n in graph.nodes():
			p = get_prefix(n)
			if p in prefix_to_entities.keys():
				prefix_to_entities[p].append(n)
			else:
				prefix_to_entities[p] = [n]

		for p in prefix_to_entities.keys():
			j = len(prefix_to_entities[p])
			if j > 1 and len (pairs) < graph.number_of_nodes():
				num_to_try = int(j*(j-1)/2 * self.beta)

				tried = set()
				# print ('j', j)
				# print ('int(j*(j-1)/2 = ', int(j*(j-1)/2))
				# print ('num_to_try = ', num_to_try)
				# if num_to_try == 0:
				# 	num_to_try = 1
				# print ('j = ', j, ' num try = ', num_to_try)
				for i in range(num_to_try):
					[left, right] = random.choices(prefix_to_entities[p], k=2)
					while (left, right) in tried:
						[left, right] = random.choices(prefix_to_entities[p], k=2)
					# print ('left = ', left)
					# print ('right = ', right)
					if left!=right:
						if self.violates_iUNA(left, right):
							pairs.add((left, right))
					tried.add((left, right))

		# print ('paris generated : ', len (pairs))
		return list(pairs)


	def get_pairs_from_qUNA(self, method, graph, beta):
		# print ('graph.number_of_nodes = ', graph.number_of_nodes())
		# print ('exp(-1*graph.number_of_nodes()/2500) = ', exp(-1*graph.number_of_nodes()/2500))
		pairs = set()
		# if method == 'existing_edges':
		# 	for e in graph.edges():
		# 		(left, right) = e
		# 		if self.violates_iUNA(left, right):
		# 			pairs.add((left, right))
		# 	print ('paris generated : ', len (pairs))
		# 	return pairs
		# elif method == 'generated_pairs':

		prefix_to_entities = {}
		for n in graph.nodes():
			p = get_prefix(n)
			if p in prefix_to_entities.keys():
				prefix_to_entities[p].append(n)
			else:
				prefix_to_entities[p] = [n]

		for p in prefix_to_entities.keys():
			j = len(prefix_to_entities[p])
			if j > 1 and len (pairs) < graph.number_of_nodes():
				num_to_try = int(j*(j-1)/2 * self.beta) + 1

				tried = set()
				# print ('j', j)
				# print ('int(j*(j-1)/2 = ', int(j*(j-1)/2))
				# print ('num_to_try = ', num_to_try)
				if num_to_try == 0:
					num_to_try = 1
				# print ('j = ', j, ' num try = ', num_to_try)
				for i in range(num_to_try):
					[left, right] = random.choices(prefix_to_entities[p], k=2)
					while (left, right) in tried:
						[left, right] = random.choices(prefix_to_entities[p], k=2)
					if self.violates_qUNA(left, right) and left != right and len (pairs) < 2*graph.number_of_nodes():
						pairs.add((left, right))
					tried.add((left, right))

		# print ('paris generated : ', len (pairs))
		return list(pairs)


	def solve_SMT (self, una, weighting_scheme): # gs.solve_SMT(una = selected_UNA, weighting_scheme = selected_weighting_scheme)
		collect_resulting_graphs = []
		collect_removed_edges = []
		iter_result = self.solve_SMT_iter (self.input_graph, una, weighting_scheme)
		if iter_result == SMT_UNKNOWN:
			print ('not enough time for SMT, keep the graph as it is')
			self.removed_edges = []
			self.result_graph = self.input_graph.copy()
			return SMT_UNKNOWN
		else:
			removed_edges, graphs = iter_result
			collect_removed_edges += removed_edges
			if len(removed_edges) == 0:
				self.result_graph = self.input_graph.copy()
				self.removed_edges = []
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
				self.removed_edges = []
				return None # finish here, no need to continue

			while condition:
				count_round += 1
				print ('\n\nThis is round ', count_round)
				collect_graphs = []
				removed_edges = []
				for g in graphs:
					iter_result = self.solve_SMT_iter(g, una, weighting_scheme)
					if iter_result != SMT_UNKNOWN:
						new_removed_edges, new_graphs = iter_result
						# print ('->removed ', len(new_removed_edges), ' edges')
						if len(new_removed_edges) > 0:
							removed_edges += new_removed_edges
							if len (new_graphs) > 1: # stop condition: removed some edges but still connected
								collect_graphs += new_graphs
							else:
								collect_resulting_graphs += new_graphs
						else:
							collect_resulting_graphs += new_graphs
					else:
						collect_graphs.append(g) # give it a second chance

				collect_removed_edges += removed_edges

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

			print ('Overall, for this graph, we removed a total of ', len (collect_removed_edges))
			self.removed_edges = collect_removed_edges

			self.result_graph = self.input_graph.copy()
			self.result_graph.remove_edges_from(collect_removed_edges)

			# print (collect_resulting_graphs)
			sizes = [c.number_of_nodes() for c in collect_resulting_graphs]
			# print ('size list = ',[c for c in sorted(sizes, reverse=True)])

	def solve_SMT_iter (self, graph, una, weighting_scheme): # get partition

		print ('\n\nThis graph has ',graph.number_of_nodes(), ' nodes')
		# print( 'and ', graph.number_of_edges(), 'edges')
		# max_equivalent_classes = 2 + int(math.log10(len(graph.nodes())))
		max_equivalent_classes = 2 + int(len(graph.nodes())/50)
		print(' max equivalent classes: ', max_equivalent_classes)
		# print ('\n\nsolving using smt')
		# resulting graph
		result_graph = graph.copy()
		# result_graph.add_nodes_from(graph.nodes())

		# encode the existing graph with weight 1
		o = Optimize()
		timeout = int(1000 * 60 * (graph.number_of_nodes()/100 + 0.2)) # depending on the size of the graph
		# timeout = int(1000 * 60 * 0.1) # depending on the size of the graph
		o.set("timeout", timeout)
		# print('timeout = ',timeout/1000, 'seconds')
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
		# print ('total # edges ', len (graph.edges()))

		# method 2: minimum_spanning_tree
		# T = nx.minimum_spanning_tree(self.input_graph)
		# # print(sorted(T.edges(data=True)))
		# print('minimum spanning tree has num of edges: ', len(T.edges()))

		# method 3: minimum spaninng forest + ignore DBpedia multilingual equivalence
		t = nx.Graph(graph)
		to_remove = []

		ms_edges = list(nx.minimum_spanning_edges(t, data= False))

		total_edges_considered = 0
		for (left, right) in graph.edges():
			ignore = False

			if (left, right) in ms_edges:
				ignore = False
			elif random.random () > (self.rate_for_remainging_other_edges):
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

		if len (uneq_pairs) <= 1:
			removed_edges = []
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
		removed_edges = []
		print ('the SMT result is ', smt_result)
		if str(smt_result) == 'unknown':
			print ('unknown as the result (cannot guarantee the correctness of the result)!!!')
			return SMT_UNKNOWN
		else:
			m = o.model()
			for arc in graph.edges():
				(left, right) = arc
				if m.evaluate(encode[left] == encode[right]) == False:
					removed_edges.append(arc)
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


	def compute_omega(self):

		# compute Omega
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
		evaluation_result ['num_error_edges_gold_standard'] = len(self.error_edges)

		if len (self.removed_edges) == 0 :
			evaluation_result ['flag'] = 'invalid precision or recall'
			return evaluation_result
		if len (self.error_edges) == 0:
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
			evaluation_result['recall'] = count_correctly_removed / len (self.error_edges)

			return evaluation_result


graph_ids = []
# validation_set + evaluation_set
export_dir = './log/'
which_method = 'smt' #
# which_method = 'louvain' #

NUM_ITER = 3

if which_method == 'louvain':
# if True:
	overall_logbook_filename = export_dir + which_method + '_overall' + '.log'
	overall_logbook_writer = open(overall_logbook_filename, 'w')
	overall_logbook_writer.write('\nmethod = ' + which_method)
	for which_set in ['validation', 'evaluation']:
		if which_set == 'validation':
			graph_ids =  validation_set
		else:
			graph_ids =  evaluation_set

		overall_logbook_writer.write ('\n********\ndataset = ' + which_set)

		overall_avg_precision = 0
		overall_avg_recall = 0
		overall_avg_omega = 0
		overall_avg_num_edges_removed = 0
		overall_avg_valid_result = 0
		overall_avg_invalid_result = 0

		overall_avg_termination_tp = 0
		overall_avg_termination_fp = 0
		overall_avg_termination_accuracy = 0

		for i in range (NUM_ITER): # repeat 5 times.
			logbook_filename = export_dir + which_method + '_' + which_set +'_Run' + str(i) + '.log'

			avg_precision = 0
			avg_recall = 0
			avg_termination_tp = 0
			avg_termination_fp = 0
			termination_tp = 0
			termination_tn = 0
			termination_fp = 0
			termination_fn = 0

			avg_omega = 0
			num_edges_removed = 0

			count_valid_result = 0
			count_invalid_result = 0
			start = time.time()
			count_graph_no_error_edges = 0
			for graph_id in graph_ids: # graph_ids:

				# removed_edge_name  = open( "convert_typeC_progress.tsv", 'w')
				# no_metalink_writer = csv.writer(file_no_metalink, delimiter='\t')
				# no_metalink_writer.writerow(['Count', 'Time'])
				filename_removed_edges = export_dir + which_method + '_' + which_set +'_Run' + str(i) + '_Graph' + str(graph_id) + '_removed_edges.tsv'
				edge_writer = csv.writer(open(filename_removed_edges, 'w'), delimiter='\t')
				edge_writer.writerow(['Source', 'Target'])

				print ('\n\n\ngraph id = ', str(graph_id))
				gold_dir = './gold/'
				gs = GraphSolver(gold_dir, graph_id)

				# gs.show_input_graph()
				# gs.show_gold_standard_graph()
				# gs.show_redirect_graph()
				# gs.show_encoding_equivalence_graph()
				# if which_method == "louvain":
				gs.partition_leuven()



				e_result = gs.evaluate_partitioning_result()

				if e_result ['num_edges_removed'] != 0:
					for (s, t) in gs.removed_edges:
						edge_writer.writerow([s, t]) #removed_edges

				if e_result ['num_error_edges_gold_standard'] == 0:
					count_graph_no_error_edges += 1

				num_edges_removed += e_result['num_edges_removed']
				avg_omega += e_result['Omega']
				if e_result['flag'] == 'valid precision and recall':
					p = e_result['precision']
					r = e_result['recall']
					m = e_result ['Omega']
					print ('precision =', p)
					print ('recall   =', r)
					print ('omega   =', m)
					count_valid_result += 1
					avg_precision += e_result['precision']
					avg_recall += e_result['recall']


				else:
					count_invalid_result += 1

				if e_result['num_edges_removed'] == 0:
					if e_result['num_error_edges_gold_standard'] == 0:
						termination_tp += 1
					else:
						termination_fp += 1
				else:
					if e_result['num_error_edges_gold_standard'] == 0:
						termination_fn += 1
					else:
						termination_tn += 1

				avg_termination_tp += termination_tp
				avg_termination_fp += temination_fp
				# avg_termination_accuracy += temination_tp /(termination_tp + temination_fp)
				# if (termination_tp + termination_fp) >0 :
				# 	avg_termination_tp = termination_tp / (termination_tp + termination_fp)
				# if (termination_tp+ termination_fn) > 0:
				# 	avg_termination_fp = termination_tp / (termination_tp+ termination_fn)

			# evaluation_result ['num_edges_removed'] = len(self.removed_edges)
			# evaluation_result ['num_error_edges_gold_standard'] = len(self.removed_edges)

			avg_omega /= len(graph_ids)
			# gs.show_result_graph()
			overall_avg_omega += avg_omega
			print ('The average Omega: ', avg_omega)
			print ('Count result [where precision and recall works] ', count_valid_result)
			print ('Count inresult [where precision and recall do not apply] ', count_invalid_result)

			if count_valid_result > 0:
				avg_precision /= count_valid_result
				avg_recall /= count_valid_result
				print ('*'*20)
				print ('There are ', len (graph_ids), ' graphs in evaluation')
				print ('   ', count_graph_no_error_edges, ' has no error edge')
				print ('The average precision: ', avg_precision)
				print ('The average recall: ', avg_recall)

				print ('Overall num edges removed ', num_edges_removed)

				overall_avg_precision += avg_precision
				overall_avg_recall += avg_recall

			overall_avg_num_edges_removed += num_edges_removed


			# if count_invalid_result > 0:
			avg_termination_tp /= len(graph_ids)
			avg_termination_fp /= len(graph_ids)
			# avg_termination_accuracy /= len(graph_ids)

			overall_avg_termination_tp += avg_termination_tp
			overall_avg_termination_fp += avg_termination_fp
			# overall_avg_termination_accuracy += avg_termination_accuracy

			overall_avg_valid_result += count_valid_result
			overall_avg_invalid_result += count_invalid_result

			end = time.time()
			hours, rem = divmod(end-start, 3600)
			minutes, seconds = divmod(rem, 60)
			time_formated = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
			print ('time taken: ' + time_formated)

		overall_avg_precision /= NUM_ITER
		overall_avg_recall /= NUM_ITER
		overall_avg_omega /= NUM_ITER
		overall_avg_num_edges_removed /= NUM_ITER
		overall_avg_valid_result /= NUM_ITER
		overall_avg_invalid_result /= NUM_ITER
		overall_avg_termination_tp /= NUM_ITER
		overall_avg_termination_fp /= NUM_ITER
		# overall_avg_termination_accuracy /= NUM_ITER

		print ('='*20)
		print ('total number of iterations over the dataset ', NUM_ITER)
		print ('OVERALL There are ', len (graph_ids), ' graphs')
		print ('   ' + str(count_graph_no_error_edges) + ' has no error edge')
		print ('OVERALL Count result [where precision and recall works] ', count_valid_result)
		print ('OVERALL Count inresult [where precision and recall do not apply] ', count_invalid_result)
		print ('OVERALL The average precision: ', avg_precision)
		print ('OVERALL The average recall: ', avg_recall)
		print ('OVERALL The average tp: [for termination]', overall_avg_termination_tp)
		print ('OVERALL The average fp: [for termination]', overall_avg_termination_fp)
		# print ('OVERALL The average accuracy: [for termination]', overall_avg_termination_accuracy)
		print ('OVERALL The average Omega: ', avg_omega)
		print ('OVERALL Overall num edges removed ', num_edges_removed)

		overall_logbook_writer.write ('\n\ntotal number of iterations over this dataset ' +str(NUM_ITER))
		overall_logbook_writer.write ('\nOVERALL There are '+str(len (graph_ids)) + ' graphs')
		overall_logbook_writer.write ('   ' + str(count_graph_no_error_edges) + ' has no error edge')
		overall_logbook_writer.write ('\nOVERALL Count result [where precision and recall works] '+ str(count_valid_result))
		overall_logbook_writer.write ('\nOVERALL Count result [where precision and recall do not apply] ' +str(count_invalid_result))
		overall_logbook_writer.write ('\nOVERALL The average precision: ' +str(avg_precision))
		overall_logbook_writer.write ('\nOVERALL The average recall: '+str(avg_recall))
		overall_logbook_writer.write ('\nOVERALL The average tp [for termination]: ' +str(overall_avg_termination_tp))
		overall_logbook_writer.write ('\nOVERALL The average fp [for termination]: '+str(overall_avg_termination_fp))
		# overall_logbook_writer.write ('\nOVERALL The average accuracy [for termination]: '+str(overall_avg_termination_accuracy))

		overall_logbook_writer.write ('\nOVERALL The average Omega: '+str(avg_omega))
		overall_logbook_writer.write ('\nOVERALL Overall num edges removed '+str(num_edges_removed))

elif which_method == 'smt':
	for selected_UNA in ['iUNA', 'qUNA']:
		for which_source in ['implicit_label_source', 'implicit_comment_source']:
			for selected_weighting_scheme in ['w2', 'w1']:

				additional = ''
				if WITH_WEIGHT:
					additional += '_[weights]'
				if WITH_DISAMBIG:
					additional += '_[disambiguation]'

				overall_logbook_filename = export_dir + which_method +'_' + selected_UNA+ '_' + which_source+ '_'+ selected_weighting_scheme + '_overall' + additional + '.log'
				overall_logbook_writer = open(overall_logbook_filename, 'w')
				overall_logbook_writer.write('\n method = ' + which_method)
				overall_logbook_writer.write('\n UNA = ' + selected_UNA)

				if WITH_WEIGHT:
					overall_logbook_writer.write('\n Additioinal info = weight')
				if WITH_DISAMBIG:
					overall_logbook_writer.write('\n Additioinal info = disambiguation')



				if selected_weighting_scheme == 'w1':
					MODE = EFFECIENT
					# print ('weighting scheme is w1: ',MODE)
				elif selected_weighting_scheme == 'w2':
					MODE = FINETUNNED
					# print ('weighting scheme is w2: ',MODE)

				overall_logbook_writer.write('\n source = ' + which_source)
				overall_logbook_writer.write('\n weighting scheme = ' + selected_weighting_scheme)

				for which_set in [ 'validation', 'evaluation']:
					time_taken = 0
					if which_set == 'validation':
						graph_ids =  validation_set
					else:
						graph_ids =  evaluation_set

					overall_logbook_writer.write ('\n********\ndataset = ' + which_set)

					overall_avg_precision = 0
					overall_avg_recall = 0
					overall_avg_omega = 0
					overall_avg_num_edges_removed = 0
					overall_avg_valid_result = 0
					overall_avg_invalid_result = 0

					overall_avg_termination_tp = 0
					overall_avg_termination_fp = 0
					overall_avg_termination_accuracy = 0

					overall_avg_timeout = 0
					for i in range (NUM_ITER): # repeat 5 times.
						logbook_filename = export_dir + which_method +'_' + selected_UNA+ '_'+ which_source +'_' + selected_weighting_scheme + '_' + which_set +'_Run' + str(i) + '.log'

						avg_precision = 0
						avg_recall = 0
						avg_termination_tp = 0
						avg_termination_fp = 0
						avg_termination_accuracy = 0

						termination_tp = 0
						termination_tn = 0
						termination_fp = 0
						termination_fn = 0


						avg_omega = 0
						num_edges_removed = 0

						count_valid_result = 0
						count_invalid_result = 0
						start = time.time()
						count_graph_no_error_edges = 0
						count_timeout = 0

						for graph_id in graph_ids: # graph_ids:

							# removed_edge_name  = open( "convert_typeC_progress.tsv", 'w')
							# no_metalink_writer = csv.writer(file_no_metalink, delimiter='\t')
							# no_metalink_writer.writerow(['Count', 'Time'])
							filename_removed_edges = export_dir + which_method +'_' + selected_UNA+ '_'+ which_source +'_' + selected_weighting_scheme + which_set +'_Run' + str(i) + '_Graph' + str(graph_id) + additional + '_removed_edges.tsv'
							edge_writer = csv.writer(open(filename_removed_edges, 'w'), delimiter='\t')
							edge_writer.writerow(['Source', 'Target'])

							print ('\n\n\ngraph id = ', str(graph_id))
							gold_dir = './gold/'
							gs = GraphSolver(gold_dir, graph_id)

							# gs.show_input_graph()
							# gs.show_gold_standard_graph()
							# gs.show_redirect_graph()
							# gs.show_encoding_equivalence_graph()

							solving_result = gs.solve_SMT(una = selected_UNA, weighting_scheme = selected_weighting_scheme)
							if solving_result  == SMT_UNKNOWN:
								count_timeout += 1

							e_result = gs.evaluate_partitioning_result()

							if e_result ['num_edges_removed'] != 0:
								for (s, t) in gs.removed_edges:
									edge_writer.writerow([s, t]) #removed_edges

							if e_result ['num_error_edges_gold_standard'] == 0:
								count_graph_no_error_edges += 1

							avg_omega += e_result['Omega']
							num_edges_removed += e_result['num_edges_removed']
							print ('omega   =', e_result ['Omega'])
							if e_result['flag'] == 'valid precision and recall':
								p = e_result['precision']
								r = e_result['recall']
								m = e_result ['Omega']
								print ('precision =', p)
								print ('recall   =', r)

								count_valid_result += 1
								avg_precision += e_result['precision']
								avg_recall += e_result['recall']
							else:
								count_invalid_result += 1


							if e_result['num_edges_removed'] == 0:
								if e_result['num_error_edges_gold_standard'] == 0:
									termination_tp += 1
								else:
									termination_fp += 1
							else:
								if e_result['num_error_edges_gold_standard'] == 0:
									termination_fn += 1
								else:
									termination_tn += 1

							avg_termination_tp += termination_tp
							avg_termination_fp += termination_fp
							# avg_termination_accuracy += termination_tp /(termination_tp + termination_fp)
							# if (termination_tp + termination_fp) >0 :
							# 	avg_termination_tp = termination_tp / (termination_tp + termination_fp)
							# if (termination_tp+ termination_fn) > 0:
							# 	avg_termination_fp = termination_tp / (termination_tp+ termination_fn)

						# evaluation_result ['num_edges_removed'] = len(self.removed_edges)
						# evaluation_result ['num_error_edges_gold_standard'] = len(self.removed_edges)

						avg_omega /= len(graph_ids)
						overall_avg_timeout += count_timeout
						# gs.show_result_graph()
						overall_avg_omega += avg_omega
						print ('The average Omega: ', avg_omega)
						print ('Count results with precision-recall', count_valid_result)
						print ('Count results without precision-recall', count_invalid_result)
						print ('Count timeout (SMT)', count_timeout)
						if count_valid_result > 0:
							avg_precision /= count_valid_result
							avg_recall /= count_valid_result
							print ('*'*20)
							print ('There are ', len (graph_ids), ' graphs in evaluation')
							print ('   ', count_graph_no_error_edges, ' has no error edge')
							print ('The average precision: ', avg_precision)
							print ('The average recall: ', avg_recall)

							print ('Overall num edges removed ', num_edges_removed)

						overall_avg_precision += avg_precision
						overall_avg_recall += avg_recall
						overall_avg_num_edges_removed += num_edges_removed


						# if count_invalid_result > 0:
						avg_termination_tp /= len(graph_ids)
						avg_termination_fp /= len(graph_ids)

						overall_avg_termination_tp += avg_termination_tp
						overall_avg_termination_fp += avg_termination_fp
						# overall_avg_termination_accuracy += avg_termination_accuracy

						overall_avg_valid_result += count_valid_result
						overall_avg_invalid_result += count_invalid_result

						end = time.time()
						time_taken += end-start
						hours, rem = divmod(end-start, 3600)
						minutes, seconds = divmod(rem, 60)
						time_formated = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
						print ('time taken: ' + time_formated)

					time_taken /= NUM_ITER
					overall_avg_precision /= NUM_ITER
					overall_avg_recall /= NUM_ITER
					overall_avg_omega /= NUM_ITER
					overall_avg_num_edges_removed /= NUM_ITER
					overall_avg_valid_result /= NUM_ITER
					overall_avg_invalid_result /= NUM_ITER
					overall_avg_termination_tp /= NUM_ITER
					overall_avg_termination_fp /= NUM_ITER
					# overall_avg_termination_accuracy /= NUM_ITER
					overall_avg_timeout /= NUM_ITER

					print ('='*20)
					print ('total number of iterations over the dataset ', NUM_ITER)
					print ('OVERALL There are ', len (graph_ids), ' graphs')
					print ('   ' + str(count_graph_no_error_edges) + ' has no error edge')
					print ('OVERALL Count results precision-recall', count_valid_result)
					print ('OVERALL Count results without precision-recall', count_invalid_result)
					print ('OVERALL COUNT SMT timeout ', overall_avg_timeout)
					print ('OVERALL The average precision: ', overall_avg_precision)
					print ('OVERALL The average recall: ', overall_avg_recall)
					print ('OVERALL The average tp: [for termination]', overall_avg_termination_tp)
					print ('OVERALL The average fp: [for termination]', overall_avg_termination_fp)
					# print ('OVERALL The average accuracy: [for termination]', overall_avg_termination_accuracy)
					print ('OVERALL The average Omega: ', avg_omega)
					print ('OVERALL Overall num edges removed ', num_edges_removed)

					hours, rem = divmod(time_taken, 3600)
					minutes, seconds = divmod(rem, 60)
					time_formated = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
					print ('avg time taken: ' + time_formated)

					overall_logbook_writer.write ('\n\ntotal number of iterations over this dataset ' +str(NUM_ITER))
					overall_logbook_writer.write ('\nOVERALL There are '+str(len (graph_ids)) + ' graphs')
					overall_logbook_writer.write ('   ' + str(count_graph_no_error_edges) + ' has no error edge')
					overall_logbook_writer.write ('\nOVERALL Count results with precision-recall '+ str(count_valid_result))
					overall_logbook_writer.write ('\nOVERALL Count results without precision-recall ' +str(count_invalid_result))
					overall_logbook_writer.write ('\nOVERALL Avg timeout ' +str(overall_avg_timeout))
					overall_logbook_writer.write ('\nOVERALL The average precision: ' +str(overall_avg_precision))
					overall_logbook_writer.write ('\nOVERALL The average recall: '+str(overall_avg_recall))
					overall_logbook_writer.write ('\nOVERALL The average tp [for termination]: ' +str(overall_avg_termination_tp))
					overall_logbook_writer.write ('\nOVERALL The average fp [for termination]: '+str(overall_avg_termination_fp))
					# overall_logbook_writer.write ('\nOVERALL The average accuracy [for termination]: '+str(overall_avg_termination_accuracy))
					overall_logbook_writer.write ('\nOVERALL The average Omega: '+str(avg_omega))
					overall_logbook_writer.write ('\nOVERALL Overall num edges removed '+str(num_edges_removed))
					overall_logbook_writer.write ('\n avg time taken: ' + time_formated)






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
