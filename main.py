# this is the class where SameAsEqSolver is defined
import networkx as nx
from SameAsEqGraph import get_simp_IRI, get_namespace
from pyvis.network import Network
import community
import collections
import matplotlib.pyplot as plt
from random import randint
import pandas as pd
import requests
from collections import Counter


# get all redirect in a list. If no redirect, then return []
def find_redirects (iri):
	try:
		# print ('test 2')
		collect_urls = []
		response = requests.get(iri, timeout=0.1, allow_redirects=True)
		if response.history:
			if response.url == iri:
				return []
			else:
				# print("Request was redirected")
				for resp in response.history:
					# print(resp.status_code, resp.url)
					collect_urls.append(resp.url)
				# print("Final destination:")
				# print(response.status_code, response.url)

				collect_urls.append(response.url)
				return collect_urls
		else:
			# print("Request was not redirected")
			return []
	except:
		# print ('error')
		return []







# define a class of graph solver
class GraphSolver():
	# configure the weight file
	# location of each node

	def __init__(self, path_to_graph):
		# input graph
		self.input_graph = nx.Graph()

		input_graph_data = pd.read_csv(path_to_graph)

		sources = input_graph_data['SUBJECT']
		targets = input_graph_data['OBJECT']
		edge_data = zip(sources, targets)

		for (s,t) in edge_data:
			self.input_graph.add_node(s, short_IRI = get_simp_IRI(s), namespace = get_namespace(s), group = 0)
			self.input_graph.add_node(t, short_IRI = get_simp_IRI(t), namespace = get_namespace(t), group = 0)
			self.input_graph.add_edge(s, t)

		# for visulization
		self.position = nx.spring_layout(self.input_graph)
		net = Network()

		# additional information
		self.redirect_graph = nx.DiGraph()
		self.encoding_equality_graph = nx.Graph()


		# solving: (weighted unique name constraints, from UNA)
		self.positive_UNC = [] # redirect and equivalence encoding
		self.negative_UNC = [] # namespace or source

		# based on this, we generate the corresponding edges
		self.attacking_edges = []
		self.confirming_edges = []

		# result
		self.partition = None
		self.result_graph = None

	def get_redirect_graph (self):
		redi_graph = nx.DiGraph()
		ct = Counter()
		new_nodes = set()
		count_nodes_not_in_graph = 0
		for n in self.input_graph.nodes:
			new_count_nodes_not_in_graph = 0
			redirected = find_redirects(n)
			ct[len(redirected)] += 1
			if len(redirected) >0: # indeed has been redirected,
				for m in redirected[1:]: # the first is the iri itself.
					if m not in self.input_graph.nodes:
						new_nodes.add(m)
					else:
						redi_graph.add_edge(n, m)

			# if len(redirected) != 0:
			# 	print ('was redirected ',len(redirected), ' times')
			# 	print ('now there are ', len(new_nodes), ' new nodes not in graph')
			count_nodes_not_in_graph += new_count_nodes_not_in_graph

		print (ct)
		print ('there are in total', len(new_nodes), 'new nodes not in the input graph')
		print('--')
		print ('there are in total', len(redi_graph.nodes()), 'new nodes not in redirect graph')
		print ('there are in total', len(redi_graph.edges()), 'new edges not in redirect graph')

		self.redirect_graph = redi_graph

	def get_encoding_equality_graph(self):
		equality_graph = nx.DiGraph()
		for n in self.input_graph.nodes:
			
		self.encoding_equality_graph = equality_graph


	def solve (self, method): # get partition
		if method == 'leuven':
			self.partition_leuven()
		elif method == 'namespace':
			self.partition_namespace()
		else:
			# first of all, compute the UNC based on namespace
			# pass

			self.result_graph  = self.input_graph.copy()

			for node in self.result_graph.nodes():
				self.result_graph.nodes[node]['group'] = randint(0,1)

			self.partition = [self.result_graph.nodes[node]['group'] for node in self.result_graph.nodes()]

	# def visualize(self):
	# 	nt = Network('700px', '700px')
	# 	nt.from_nx(self.input_graph.graph)
	# 	# nt.enable_physics(True)
	# 	nt.show_buttons(filter_=['physics'])
	# 	nt.show('input.html')
	#
	def partition_leuven(self):
		g = self.input_graph
		self.result_graph = self.input_graph.copy()

		partition = community.best_partition(g)
		for node in self.input_graph.nodes():
			self.result_graph.nodes[node]['group'] = partition.get(node)
		self.partition = [partition.get(node) for node in self.input_graph.nodes()]

	def partition_namespace(self):
		namespace_to_entities = {}
		for e in self.input_graph.nodes():
			ns = get_namespace(e)
			if ns in namespace_to_entities.keys():
				namespace_to_entities[ns].append(e)
			else:
				namespace_to_entities[ns] = [e]

		self.negative_UNC = namespace_to_entities.values()
		print ('The size of negative UMC: ', len (self.negative_UNC))
		# print (self.negative_UNC)

		# from this, we generate the negative UNC, with uniformed weight 1 (for now)
		for (s, t) in self.input_graph.edges():
			if self.input_graph.nodes[s]['namespace'] ==  self.input_graph.nodes[t]['namespace']:
				# add an attacking edges
				# print (self.input_graph.nodes[s]['namespace'])
				# print (self.input_graph.nodes[t]['namespace'])
				self.attacking_edges.append((s,t))
		print ('The size of attacking edges generated from negative UMC = ', len (self.attacking_edges))

		self.result_graph  = self.input_graph.copy()

		for node in self.result_graph.nodes():
			self.result_graph.nodes[node]['group'] = randint(0,1)

		self.partition = [self.result_graph.nodes[node]['group'] for node in self.result_graph.nodes()]


	def show_input_graph(self):
		g = self.input_graph
		# sp = nx.spring_layout(g)
		nx.draw_networkx(g, pos=self.position, with_labels=False, node_size=25)
		plt.show()

	def show_redirect_graph(self):
		g = self.redirect_graph
		# sp = nx.spring_layout(g)
		# nx.draw_networkx(g, pos=self.position, with_labels=False, node_size=35)
		nx.draw_networkx_edges(g, pos=self.position, edge_color='red', connectionstyle='arc3,rad=0.2')
		nx.draw_networkx(self.input_graph, pos=self.position, with_labels=False, node_size=25)
		plt.show()

	def show_result (self):
		g = self.result_graph
		# counter = collections.Counter(values)
		# print(counter)
		# sp = nx.spring_layout(g)
		nx.draw_networkx(g, pos=self.position, with_labels=False, node_size=35, node_color=self.partition)
		# plt.axes('off')
		plt.show()

# main
gs = GraphSolver('./Evaluate_May/11116_edges_original.csv')
print (nx.info(gs.input_graph))
gs.show_input_graph()
#
# gs.get_redirect_graph()
# gs.show_redirect_graph()

# gs.get_encoding_equality_graph()

# gs.solve('namespace')
# gs.solve('leuven')
#
# gs.show_result()
