# this is the class where SameAsEqSolver is defined
import networkx as nx
from SameAsEqGraph import SameAsEqGraph, SameAsEqInputGraph, SameAsEqResultGraph


# define a class of graph solver
class GraphSolver():
	# configure the weight file
	# location of each node

	def __init__(self, path_to_graph):
		self.input_graph = SameAsEqInputGraph(path_to_graph)
		self.pos = nx.spring_layout(self.input_graph.graph)
		self.result_graph = SameAsEqResultGraph()


	def solve (self):
		cp  = self.input_graph.graph.copy()
		self.result_graph.subgraphs.append(cp)
		print ('solving, resulting subgraphs =',len(self.result_graph.subgraphs))

# main
gs = GraphSolver('./Evaluate_May/11116_edges_original.csv')
print (len(gs.input_graph.get_nodes()))
gs.solve()
print (len(gs.result_graph.get_nodes()))
