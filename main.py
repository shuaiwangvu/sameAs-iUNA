# this is the class where SameAsEqSolver is defined
import networkx as nx
from SameAsEqGraph import SameAsEqGraph, SameAsEqInputGraph, SameAsEqResultGraph
from pyvis.network import Network
import community
import collections
import matplotlib.pyplot as plt

# define a class of graph solver
class GraphSolver():
	# configure the weight file
	# location of each node

	def __init__(self, path_to_graph):
		self.input_graph = SameAsEqInputGraph(path_to_graph)
		self.pos = nx.spring_layout(self.input_graph.graph) # for visulization
		self.unique_name_constraints = None # weighted UNC
		self.result_graph = SameAsEqResultGraph()
		net = Network()

	def solve (self):
		cp  = self.input_graph.graph.copy()
		self.result_graph.subgraphs.append(cp)
		print ('solving, resulting subgraphs =',len(self.result_graph.subgraphs))

	def visualize(self):
		nt = Network('700px', '700px')
		nt.from_nx(self.input_graph.graph)
		# nt.enable_physics(True)
		nt.show_buttons(filter_=['physics'])
		nt.show('input.html')

	def partition_leuven(self):
		g = self.input_graph.graph
		partition = community.best_partition(g)
		values = [partition.get(node) for node in g.nodes()] # values are group ids
		counter=collections.Counter(values)
		print(counter)
		sp = nx.spring_layout(g)
		nx.draw_networkx(g, pos=sp, with_labels=False, node_size=35, node_color=values)
		# plt.axes('off')
		plt.show()



# main
gs = GraphSolver('./Evaluate_May/11116_edges_original.csv')
print (len(gs.input_graph.get_nodes()))
gs.solve()
print (len(gs.result_graph.get_nodes()))
gs.visualize()
# gs.partition_leuven()
