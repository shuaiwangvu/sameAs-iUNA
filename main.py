# this is the class where SameAsEqSolver is defined
import networkx as nx
from SameAsEqGraph import simp
from pyvis.network import Network
import community
import collections
import matplotlib.pyplot as plt
from random import randint
import pandas as pd



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
			self.input_graph.add_node(s, short_IRI = simp(s), group = 0)
			self.input_graph.add_node(t, short_IRI = simp(t), group = 0)
			self.input_graph.add_edge(s, t)

		# for visulization
		self.position = nx.spring_layout(self.input_graph)
		net = Network()

		# solving: (weighted unique name constraints)
		self.unique_name_constraints = None # weighted UNC
		# result
		self.partition = None
		self.result_graph = None


	def solve (self, method): # get partition
		if method == 'leuven':
			self.partition_leuven()
		else:
			self.result_graph  = self.input_graph.copy()
			print ('solve num of nodes = ', len (self.result_graph.nodes()))

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

		self.partition = community.best_partition(g)
		for node in self.input_graph.nodes():
			self.result_graph.nodes[node]['group'] = self.partition.get(node)

	def show_input(self):
		g = self.input_graph
		# sp = nx.spring_layout(g)
		nx.draw_networkx(g, pos=self.position, with_labels=False, node_size=35)
		plt.show()

	def show_result (self):
		g = self.result_graph
		values = [self.partition.get(node) for node in g.nodes()] # values are group ids
		# counter = collections.Counter(values)
		# print(counter)
		# sp = nx.spring_layout(g)
		nx.draw_networkx(g, pos=self.position, with_labels=False, node_size=35, node_color=values)
		# plt.axes('off')
		plt.show()

# main
gs = GraphSolver('./Evaluate_May/11116_edges_original.csv')
print (nx.info(gs.input_graph))
gs.show_input()

gs.solve('leuven')

gs.show_result()
