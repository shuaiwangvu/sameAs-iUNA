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
validate_single = [96073, 712342, 9994282, 18688, 1140988, 25604]
validate_multiple = [33122, 11116, 6617,  12745,4170, 42616, 6927, 39036]
validation_set = validate_single + validate_multiple

evaluation_single = [9411, 9756, 97757, 99932, 337339, 1133953]
evaluation_multiple = [5723, 14872, 37544, 236350, 240577, 395175, 4635725, 14514123]
evaluation_set = evaluation_single + evaluation_multiple


gs = validation_set + evaluation_set




# define a class of graph solver
class GraphSolver():
	# configure the weight file
	# location of each node

	def __init__(self, dir, id):
		# input graph
		self.input_graph = nx.Graph()
		self.gold_standard_partition = []

		path_to_nodes = dir + str(id) +'.tsv'
		path_to_edges = dir + str(id) +'_edges.tsv'
		self.input_graph = load_graph(path_to_nodes, path_to_edges)
		print ('number of nodes', self.input_graph.number_of_nodes())
		print ('number of edges ', self.input_graph.number_of_edges())
		self.error_edges = []
		for e in self.input_graph.edges():
			(s, t) = e
			if self.input_graph.nodes[s]['annotation'] != 'unknown' and self.input_graph.nodes[t]['annotation'] != 'unknown':
				if self.input_graph.nodes[s]['annotation'] != self.input_graph.nodes[t]['annotation']:
					self.error_edges.append(e)
		# print ('There are ', len (self.error_edges), ' error edges')
		# for visulization
		self.position = nx.spring_layout(self.input_graph)

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

		path_to_explicit_source = dir + str(graph_id) + "_explicit_source.hdt"
		load_explicit(path_to_explicit_source, self.input_graph)
		path_to_implicit_label_source = dir + str(graph_id) + "_implicit_label_source.hdt"
		load_implicit_label_source(path_to_implicit_label_source, self.input_graph)
		path_to_implicit_comment_source = dir + str(graph_id) + "_implicit_comment_source.hdt"
		load_implicit_comment_source(path_to_implicit_comment_source, self.input_graph)

		# disambiguation entities
		# load_disambiguation_entities
		path_to_disambiguation_entities = "sameas_disambiguation_entities.hdt"
		self.dis_entities = load_disambiguation_entities(self.input_graph.nodes(), path_to_disambiguation_entities)
		# print ('there are ', len (self.dis_entities), ' entities about disambiguation (in this graph)')

		# weight graph
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
		self.rate_for_remainging_other_edges = 0.2 # for the remaining edges
		self.reduced_weight_disambiguation = 5
		self.default_weight = 36
		# self.max_clusters = 2 + int(len(self.input_graph.nodes())/150)
		# attacking related
		self.beta = 0.8 #
		self.weight_iUNA_uneq_edge = 15 # weight of attacking edge
		# additional information
		self.weight_encoding_equivalence = 5
		self.weight_redirect = 5
		# parameters about weights on the edges
		self.weights_occ = False
		self.alpha = 2 # soft_clauses[clause] = default_weight + (w * alpha)


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

		# update the entity_to_partition

		# update the partition
		for k in self.partition_to_entities.keys():
			print ('partition ',k, ' has ', len(self.partition_to_entities[k]), ' entities')
		# print ('partition is like ', self.partition_to_entities)
		# print ('partition is like ', self.entity_to_partition)

	def violates_iUNA (self, left, right):
		source_left =  self.input_graph.nodes[left]['implicit_label_source']
		source_right =  self.input_graph.nodes[right]['implicit_label_source']
		prefix_left = get_prefix(left)
		prefix_right = get_prefix(right)

		if len(set(source_left).difference(set(source_right))) > 0 and prefix_left == prefix_right:
			if 'http://dbpedia.org/resource/' in left and 'http://dbpedia.org/resource/' in right:
				return False
			# print ('testing left = ', left)
			# print ('testing right = ', right)
			# test also if it is in redirected
			if left in self.redi_undirected.nodes() and right in self.redi_undirected.nodes():
				if nx.has_path(self.redi_undirected, left, right):
					# print ('redi has path')
					return False
			if left in self.encoding_equality_graph.nodes() and right in self.encoding_equality_graph.nodes():
				if nx.has_path(self.encoding_equality_graph, left, right):
					# print ('ee chas path')c
					return False

			# print ('violates iUNA')
			return True

			# if (left, right) in self.redirect_graph.edges() or (right, left) in self.redirect_graph.edges():
			# 	return False
			# elif (left, right) in self.encoding_equality_graph.edges() or (right, left) in self.encoding_equality_graph.edges():
			# 	return False
			# else:
			# 	return True

	def get_pairs_from_iUNA(self, method, graph, beta):
		print ('graph.number_of_nodes = ', graph.number_of_nodes())
		# print ('exp(-1*graph.number_of_nodes()/2500) = ', exp(-1*graph.number_of_nodes()/2500))
		pairs = set()
		if method == 'existing_edges':
			for e in graph.edges():
				(left, right) = e
				if self.violates_iUNA(left, right):
					pairs.add((left, right))
			print ('paris generated : ', len (pairs))
			return pairs
		elif method == 'generated_pairs':

			prefix_to_entities = {}
			for n in graph.nodes():
				p = get_prefix(n)
				if p in prefix_to_entities.keys():
					prefix_to_entities[p].append(n)
				else:
					prefix_to_entities[p] = [n]

			for p in prefix_to_entities.keys():
				j = len(prefix_to_entities[p])
				if j > 1:
					num_to_try = int(j*(j-1)/2 * self.beta)


					# print ('j', j)
					# print ('int(j*(j-1)/2 = ', int(j*(j-1)/2))
					# print ('num_to_try = ', num_to_try)
					if num_to_try == 0:
						num_to_try = 1
					# print ('j = ', j, ' num try = ', num_to_try)
					for i in range(num_to_try):
						[left, right] = random.choices(prefix_to_entities[p], k=2)
						if self.violates_iUNA(left, right) and left != right:
							pairs.add((left, right))

			print ('paris generated : ', len (pairs))
			return list(pairs)

	def solve_SMT (self):
		collect_resulting_graphs = []
		collect_removed_edges = []
		iter_result = self.solve_SMT_iter (self.input_graph)
		if iter_result == SMT_UNKNOWN:
			print ('not enough time for SMT')
			return None
		removed_edges, graphs = iter_result
		collect_removed_edges += removed_edges
		print ('# first round removed edges = ', len(removed_edges))
		for g in graphs:
			if g.number_of_nodes()<=1:
				collect_resulting_graphs.append(g)
		graphs = list(filter(lambda x: x.number_of_nodes()>1, graphs))



		count_round = 1
		# print ('after round 1, there are still ', len(graphs), 'graphs. ')
		print ('after round 1, there are ', len(collect_removed_edges), 'edges removed. ')

		condition = True
		if len (removed_edges) == 0:
			condition = False
			self.removed_edges = []
			return None # finish here, no need to continue

		while condition:
			count_round += 1
			print ('This is round ', count_round)
			collect_graphs = []
			removed_edges = []
			for g in graphs:
				iter_result = self.solve_SMT_iter(g)
				if iter_result != SMT_UNKNOWN:
					new_removed_edges, new_graphs = iter_result
					print ('->removed ', len(new_removed_edges), ' edges')
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
			for g in collect_graphs:
				print ( '\t#nodes: ', g.number_of_nodes(), ' #edges: ', g.number_of_edges())

			if len (removed_edges) == 0 or len(collect_graphs) == 0:
				condition = False
			graphs = collect_graphs

		print ('Overall, for this graph, we removed a total of ', len (collect_removed_edges))
		self.removed_edges = collect_removed_edges

		self.result_graph = self.input_graph.copy()
		self.result_graph.remove_edges_from(collect_removed_edges)

		# print (collect_resulting_graphs)
		sizes = [c.number_of_nodes() for c in collect_resulting_graphs]
		print ('size list = ',[c for c in sorted(sizes, reverse=True)])

	def solve_SMT_iter (self, graph): # get partition

		print ('\n\nThis graph has ',graph.number_of_nodes(), ' nodes')
		print( 'and ', graph.number_of_edges(), 'edges')
		max_clusters = 2 + int(len(graph.nodes())/30)
		print(' max clusters: ', max_clusters)
		# print ('\n\nsolving using smt')
		# resulting graph
		result_graph = graph.copy()
		# result_graph.add_nodes_from(graph.nodes())

		# encode the existing graph with weight 1
		o = Optimize()
		timeout = int(1000 * 60 * (graph.number_of_edges()/2000 + 0.2)) # depending on the size of the graph
		o.set("timeout", timeout)
		# print('timeout = ',timeout/1000, 'seconds')
		print('timeout = ',timeout/1000/60, 'mins')
		encode = {}
		soft_clauses = {}

		# STEP 1: the input graph (nodes, edges and their weights)
		# print ('STEP 1: the input graph (nodes, edges and their weights)')
		# default_weight = 35
		# reduced_weight_disambiguation = 1
		# self.max_clusters = 8 + int(len(graph.nodes())/150)
		# weights_occ = False

		# count_weighted_edges = 0

		encode_id = 1
		for n in graph.nodes():
			encode[n] = Int(str(encode_id))
			encode_id += 1
			o.add(encode[n] > Int(0))
			o.add(encode[n] < Int(max_clusters +1))

		count_ignored = 0
		# print ('total # edges ', len (graph.edges()))

		# method 2: minimum_spanning_tree
		# T = nx.minimum_spanning_tree(self.input_graph)
		# # print(sorted(T.edges(data=True)))
		# print('minimum spanning tree has num of edges: ', len(T.edges()))

		# method 3: minimum spaninng forest + ignore DBpedia multilingual equivalence
		t = nx.Graph(graph)
		to_remove = []
		for (left, right) in graph.edges():
			ignore = False
			#Method 1: ignore the edges between multilingual DBpedia entities.
			if 'dbpedia.org' in left and 'dbpedia.org' in right and 'http://dbpedia.org/resource/' not in left and 'http://dbpedia.org/resource/' not in right:
				# if there is a common dbpedia.org neighbor, then we ignore this.
				if get_authority(left) != get_authority(right):
					left_neigh = graph.neighbors(left)
					right_neigh = graph.neighbors(right)

					if len(set(left_neigh).difference(set(right_neigh))) > 0:
						to_remove.append((left, right))
		t.remove_edges_from(to_remove)
		ms_edges = list(nx.minimum_spanning_edges(t, data= False))
		# ms_edges = t.edges()
		print ('the num of remaining edges in the minimum spanning forest is ', len(ms_edges))
		total_edges_considered = 0
		for (left, right) in graph.edges():
			ignore = False
			#Method 1: ignore the edges between multilingual DBpedia entities.
			# if 'dbpedia.org' in left and 'dbpedia.org' in right and 'http://dbpedia.org/resource/' not in left and 'http://dbpedia.org/resource/' not in right:
			# 	# if there is a common dbpedia.org neighbor, then we ignore this.
			# 	if get_authority(left) != get_authority(right):
			# 		left_neigh = self.input_graph.neighbors(left)
			# 		right_neigh = self.input_graph.neighbors(right)
			#
			# 		if len(set(left_neigh).difference(set(right_neigh))) > 0:
			# 			if (random.random() <= 0.90):
			# 				ignore = True
			# 				count_ignored += 1
			# Method 2: minimum spanning tree
			# if (left, right) not in T.edges():
			# 	ignore = True
			# 	count_ignored += 1

			#Method 3: minimum spanning forest
			if (left, right) in ms_edges:
				ignore = False
			elif random.random () > (self.rate_for_remainging_other_edges):
				ignore = True
				count_ignored += 1
			# else:
			# 	ignore = True

			if ignore == False:
				clause = (encode[left] == encode[right])
				total_edges_considered += 1
				if left in self.dis_entities or right in self.dis_entities:
					soft_clauses[clause] = self.default_weight - self.reduced_weight_disambiguation
					# soft_clauses[clause] = default_weight
					# print ('weight = ', soft_clauses[clause])
				else:
					soft_clauses[clause] = self.default_weight

				if self.weights_occ == True:
					# otherwise, use the weights from the
					w = graph.edges[left, right]['weight']
					# print ('!!!!!!! I have weight',w)
					if w != None:
						soft_clauses[clause] = self.default_weight + (w * self.alpha)

					else:
						print ('weighting error?!')
				else:
					soft_clauses[clause] = self.default_weight

		# print ('count_weighted_edges = ', count_weighted_edges)
		# print ('count_ignored edges between DBpedia multilingual entities', count_ignored)
		print ('also includes some number of edges = ', total_edges_considered)
		# STEP 2: the attacking edges
		# print ('STEP 2: the attacking edges')


		# uneq_pairs = self.get_pairs_from_iUNA(method = 'existing_edges')
		uneq_pairs = self.get_pairs_from_iUNA(method = 'generated_pairs', graph=graph, beta = self.beta)

		# if graph.number_of_nodes() <30:
		# 	for (left, right) in uneq_pairs:
		# 		print ('attacking left = ', left)
		# 		print ('attacking right = ', right)

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

		print ('count  correct ', count_correct_attack, ' -> ', count_correct_attack/len(uneq_pairs))
		print ('count mistaken ', count_mistaken_attack, ' -> ', count_mistaken_attack/len(uneq_pairs))
		print ('This round, the weight is ', int(self.weight_iUNA_uneq_edge )) # * exp(-1*graph.number_of_nodes()/2500)))
		for (left,right) in uneq_pairs:
			clause = Not(encode[left] == encode[right])
			if clause in soft_clauses.keys():
				soft_clauses[clause] += int(self.weight_iUNA_uneq_edge) # * exp(-1*graph.number_of_nodes()/2500))
			else:
				soft_clauses[clause] = int(self.weight_iUNA_uneq_edge) # * exp(-1*graph.number_of_nodes()/2500))

		# STEP 3: the confirming edges
		# add confirming edges: encoding equivalence
		weight_encoding_equivalence = 10
		for (left, right) in self.encoding_equality_graph.edges():
			if left in graph.nodes() and right in graph.nodes():
				clause = (encode[left] == encode[right])
				if clause in soft_clauses.keys():
					soft_clauses[clause] += self.weight_encoding_equivalence
				else:
					soft_clauses[clause] = self.weight_encoding_equivalence


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

			# print('** Examining the attacking edges **')
			# print ('out of ', len (uneq_pairs), ' attacking edges')
			# print ('\t', len (successful_attacking_edges), ' are successful in attack')
			# print ('\t', len (unsuccessful_attacking_edges), ' have failed in attack')

		# comps  = nx.connected_components(result_graph)
		# print ('components: ', comps)
		# print ('# removed = ', len(removed_edges))
		# print ('size list = ',[len(c) for c in sorted(comps, key=len, reverse=True)])
		# resulting_graphs = []
		# for c in comps:
		# 	print('c = ', c)
		# 	gc = self.input_graph.subgraph(list(c)).copy()
		# 	resulting_graphs.append(gc)
		# G.subgraph(nodes).copy()
		resulting_graphs = [result_graph.subgraph(c) for c in nx.connected_components(result_graph)]
		# for r in resulting_graphs:
		# 	for n in r.nodes():
		# 		print (r.nodes[n]['annotation'])

		# Finally, compute the partitions
		# self.partition = None

		return (removed_edges, resulting_graphs)

	def evaluate_partitioning_result(self):
		if len (self.error_edges) == 0 :
			print ('total removed edges = ', len(self.removed_edges))
			return None
		else:
			print ('num of edges removed: ', len(self.removed_edges))

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

			evaluation_result = {}
			evaluation_result ['num_edges_removed'] = len(self.removed_edges)

			if self.removed_edges != []:
				print ('precision = ', count_correctly_removed/len (self.removed_edges))
				evaluation_result['precision'] = count_correctly_removed / len (self.removed_edges)
				print ('recall = ', count_correctly_removed/len (self.error_edges))
				evaluation_result['recall'] = count_correctly_removed / len (self.error_edges)
				return evaluation_result
			else:
				print ('no edge removed')
				return None






avg_precision = 0
avg_recall = 0
num_edges_removed = 0

count_valid_result = 0
count_invalid_result = 0

hard_graph_ids = [33122, 11116, 6927, 39036]
# graph_ids = hard_graph_ids
graph_ids = validate_multiple
# hard_graph_ids
start = time.time()
for graph_id in graph_ids:
	print ('\n\n\ngraph id = ', str(graph_id))
	dir = './gold/'
	gs = GraphSolver(dir, graph_id)
	# gs.partition_leuven()
	gs.solve_SMT()
	e_result = gs.evaluate_partitioning_result()
	if e_result != None:
		p = e_result['precision']
		r = e_result['recall']
		count_valid_result += 1
		avg_precision += e_result['precision']
		avg_recall += e_result['recall']
		num_edges_removed += e_result['num_edges_removed']
	else:
		count_invalid_result += 1


if count_valid_result > 0:
	avg_precision /= count_valid_result
	avg_recall /= count_valid_result
	print ('*'*20)
	print ('There are ', len (graph_ids), ' graphs in evaluation')
	print ('Count valid results ', count_valid_result)
	print ('Count invalid results ', count_invalid_result)
	print ('The average precision: ', avg_precision)
	print ('The average recall: ', avg_recall)
	print ('Overall num edges removed ', num_edges_removed)


else:
	print ('No valid result')

end = time.time()
hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
time_formated = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
print ('time taken: ' + time_formated)



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
