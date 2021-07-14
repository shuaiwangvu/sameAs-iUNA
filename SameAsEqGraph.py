# this is an abstract class



class SameAsEqGraph:

	def get_nodes(self):
		pass

class SameAsEqInputGraph(SameAsEqGraph):

	def __init__(self):
		super().__init__()

	def get_nodes(self):
		return [1,2,3]


class SameAsEqResultGraph(SameAsEqGraph):
	def __init__(self):
		self.subgraphs = [1,2,3]
		super().__init__()

	def get_nodes(self):
		nodes = []
		for sg in self.subgraphs:
			nodes.append(sg*2)
		return nodes
