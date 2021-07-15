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
		got_data = pd.read_csv(path_to_graph)

		sources = got_data['SUBJECT']
		targets = got_data['OBJECT']
		edge_data = zip(sources, targets)

		for (s,t) in edge_data:
			self.input_graph.add_node(s, short_IRI = get_simp_IRI(s), namespace = get_namespace(s), group = 0)
			self.input_graph.add_node(t, short_IRI = get_simp_IRI(t), namespace = get_namespace(t), group = 0)
			self.input_graph.add_edge(s, t)

		# for visulization
		self.position = nx.spring_layout(self.input_graph)
		net = Network()

		# solving: (weighted unique name constraints, from UNA)
		self.positive_UNC = [] # redirect and equivalence encoding
		self.negative_UNC = [] # namespace or source

		# based on this, we generate the corresponding edges
		self.attacking_edges = []
		self.confirming_edges = []

		# result
		self.partition = None
		self.result_graph = None

	def obtain_all_redirects(self):
		redirect_list = []
		for e in self.input_graph.nodes:
			if find_redirects(e) != []:
				redirect_list.append(len(find_redirects(e)))
				# print (e, len(find_redirects(e)))
			else:
				redirect_list.append(0)

		counter = collections.Counter(redirect_list)
		print(counter)
		count = sum (redirect_list)
		print ('average redirect = ', count/ self.input_graph.number_of_nodes())

	def obtain_redirect_graph(self):
		# how is the redirection graph overlapping with the input graph?

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


	def show_input(self):
		g = self.input_graph
		# sp = nx.spring_layout(g)
		nx.draw_networkx(g, pos=self.position, with_labels=False, node_size=35)
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
gs.show_input()

gs.obtain_all_redirects()
#
# gs.solve('namespace')
#
# gs.show_result()
