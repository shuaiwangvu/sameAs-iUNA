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


UNKNOWN = 0
REMOVE = 1
KEEP = 2

hdt_source = HDTDocument("typeA.hdt")
hdt_label = HDTDocument("label_May.hdt")
hdt_comment = HDTDocument("comment_May.hdt")

# PATH_LOD = "/scratch/wbeek/data/LOD-a-lot/data.hdt"
# hdt_lod = HDTDocument(PATH_LOD)

my_has_label_in_file = "https://krr.triply.cc/krr/metalink/def/hasLabelInFile" # a relation
my_has_comment_in_file = "https://krr.triply.cc/krr/metalink/def/hasCommentInFile" # a relation
rdfs_label = "http://www.w3.org/2000/01/rdf-schema#label"
rdfs_comment = 'http://www.w3.org/2000/01/rdf-schema#comment'


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

	def __init__(self, path_to_input_graph, path_to_gold_standard_graph):
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

		# solving: (weighted unique name constraints, from UNA)
		# self.positive_UNC = [] # redirect and equivalence encoding
		# self.negative_UNC = [] # namespace or source


		# result
		self.result_partition = []
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

	def get_namespace_graph(self):
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
			if len (ns_entities)>1:
				print ('\nnamesapce ', ns)
				print ('has ', len (ns_entities), ' entities:')
				for e in namespace_to_entities[ns]:
					print ('\t',e)

			for i, e in enumerate(ns_entities):
				for f in ns_entities[i+1:]:
					self.namespace_graph.add_edge(e, f)

		print ('The namespace has ', len (self.namespace_graph), ' attacking edges')

	def get_typeA_graph (self):
		print ('\ngenerating typeA graph')
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
			for i, e in enumerate(file_to_entities[f]):
				for f in list(file_to_entities[f])[i+1:]:
					self.typeA_graph.add_edge(e, f)
		print ('There are in total ', len (self.typeA_graph.edges()), 'attacking edges from this source file')



	def get_typeB_graph (self):
		print ('\ngenerating typeB graph')
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
			if len(file_to_entities[file]) >1:
				print ('\nfile ', file, ' has ', len (file_to_entities[file]), 'entities')

			for i, e in enumerate(file_to_entities[file]):
				if len(file_to_entities[file]) >1:
					print ('No.', i, ' = ', e)
				for f in list(file_to_entities[file])[i+1:]:
					self.typeB_graph.add_edge(e, f)
		print ('There are in total ', len (self.typeB_graph.edges()), 'attacking edges from this label file')


	def get_typeC_graph (self):
		# load the resources and
		pass


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
# main
gs = GraphSolver(path_to_input_graph = './Evaluate_May/11116_edges_original.csv',
				path_to_gold_standard_graph = './Evaluate_May/11116_nodes_labelled.tsv')
print (nx.info(gs.input_graph))

# --
# gs.get_encoding_equality_graph()
# gs.get_redirect_graph()
# gs.get_namespace_graph()
gs.get_typeA_graph()
gs.get_typeB_graph()

# -- visualization --
# gs.show_input_graph()
# gs.show_redirect_graph()
# gs.show_encoding_equivalence_graph()
# gs.show_namespace_graph()


# -- solve --

# -- show result --
# gs.show_gold_standard_graph()
