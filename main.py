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


# get all redirect in a list. If no redirect, then return []
def find_redirects (iri):
	try:
		# print ('test 2')
		collect_urls = []
		response = requests.get(iri, timeout=0.1, allow_redirects=True)

		# import urllib3
		#
		# resp = urllib3.request(
		#     "GET",
		#     "https://httpbin.org/delay/3",
		#     timeout=4.0
		# )

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
		# net = Network()

		# based on this, we generate the corresponding edges
		self.namespace_graph = nx.Graph()
		self.source_graph = nx.Graph()

		# additional information
		self.redirect_graph = nx.DiGraph()
		self.encoding_equality_graph = nx.Graph()


		# solving: (weighted unique name constraints, from UNA)
		# self.positive_UNC = [] # redirect and equivalence encoding
		# self.negative_UNC = [] # namespace or source


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

			if len(redirected) != 0:
				if len(redirected) > 10:
					print ('was redirected ',len(redirected), ' times')
					for index, iri in enumerate(redirected):
						print ('\tNo.',index, ' -> ', iri)
					print ('\n')
				# print ('now there are ', len(new_nodes), ' new nodes not in graph')
			count_nodes_not_in_graph += new_count_nodes_not_in_graph

		print (ct)
		print ('there are in total', len(new_nodes), 'new nodes not in the input graph')
		print('--')
		# print ('there are in total', len(redi_graph.nodes()), 'new nodes not in redirect graph')
		print ('there are in total', len(redi_graph.edges()), 'new edges not in redirect graph')

		self.redirect_graph = redi_graph

	def get_encoding_equality_graph(self):
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

	def get_namespace_graph(self):
		namespace_to_entities = {}
		for e in self.input_graph.nodes():
			ns = get_namespace(e)
			# print (e)
			# print ('has name space', ns)
			if ns in namespace_to_entities.keys():
				namespace_to_entities[ns].append(e)
			else:
				namespace_to_entities[ns] = [e]

		for ns in namespace_to_entities.keys():
			ns_entities = namespace_to_entities[ns]
			if len (ns_entities)>1:
				print ('\nnamesapce ', ns)
				print ('has ', len (ns_entities), ' entities:')
				for e in namespace_to_entities[ns]:
					print ('\t',e)

			for i, e in enumerate(ns_entities):
				for f in ns_entities[i+1:]:
					self.namespace_graph.add_edge(e, f)

		print ('The namespace has ', len (self.namespace_graph), ' attacking edges')


	def show_input_graph(self):
		g = self.input_graph
		# sp = nx.spring_layout(g)
		nx.draw_networkx(g, pos=self.position, with_labels=False, node_size=25)
		plt.title('Input')
		plt.show()

	def show_redirect_graph(self):
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
		nx.draw_networkx_edges(g, pos=self.position, edge_color='red', connectionstyle='arc3,rad=0.2')

		plt.title('Encoding Equivalence')
		plt.show()

	def show_namespace_graph(self):
		g = self.namespace_graph
		# sp = nx.spring_layout(g)
		# nx.draw_networkx(g, pos=self.position, with_labels=False, node_size=35)

		nx.draw_networkx(self.input_graph, pos=self.position, with_labels=False, node_size=25)
		nx.draw_networkx_edges(g, pos=self.position, edge_color='red', connectionstyle='arc3,rad=0.2')

		plt.title('Namesapce Attacking edges')
		plt.show()


	def show_result_graph (self):
		g = self.result_graph
		# counter = collections.Counter(values)
		# print(counter)
		# sp = nx.spring_layout(g)
		nx.draw_networkx(g, pos=self.position, with_labels=False, node_size=35, node_color=self.partition)
		# plt.axes('off')
		plt.title('Result')
		plt.show()

# main
gs = GraphSolver('./Evaluate_May/11116_edges_original.csv')
print (nx.info(gs.input_graph))

# gs.get_encoding_equality_graph()
# gs.get_redirect_graph()
gs.get_namespace_graph()

gs.show_input_graph()
# gs.show_redirect_graph()
# gs.show_encoding_equivalence_graph()
gs.show_namespace_graph()


# --solve --

# gs.show_result_graph()
