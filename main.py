# this is the class where SameAsEqSolver is defined
from SameAsEqGraph import SameAsEqGraph, SameAsEqInputGraph, SameAsEqResultGraph


# define a class of graph solver
class GraphSolver():
	g = SameAsEqInputGraph()
	h = SameAsEqResultGraph()

# main
gs = GraphSolver()
print (gs.g.get_nodes())
print (gs.h.get_nodes())
