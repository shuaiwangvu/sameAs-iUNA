# this is an abstract class
import networkx as nx
import csv

class SameAsEqGraph:

	def get_nodes(self):
		pass

class SameAsEqInputGraph(SameAsEqGraph):

	def __init__(self, path_to_graph):
		super().__init__()
		self.graph = nx.Graph()
		eq_file = open(path_to_graph, 'r')
		reader = csv.DictReader(eq_file)
		for row in reader:
			s = row["SUBJECT"]
			o = row["OBJECT"]
			self.graph.add_edge(s, o)

	def get_nodes(self):
		return self.graph.nodes()


class SameAsEqResultGraph(SameAsEqGraph):
	def __init__(self):
		self.subgraphs = []
		super().__init__()

	def get_nodes(self):
		nodes = []
		for sg in self.subgraphs:
			nodes += sg.nodes()
		return nodes
