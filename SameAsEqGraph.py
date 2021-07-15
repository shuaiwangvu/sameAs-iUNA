# this is an abstract class
import networkx as nx
import pandas as pd
import tldextract

class SameAsEqGraph:

	def get_nodes(self):
		pass
	def get_edges(self):
		pass


def simp(e):
	# simplify this uri by introducing the namespace abbreviation
	ext = tldextract.extract(e)
	# ExtractResult(subdomain='af', domain='dbpedia', suffix='org')

	if 'dbpedia' == ext.domain and ext.subdomain != '' and ext.subdomain != None:
		namespace = ext.subdomain +'.'+ext.domain
	else :
		namespace = ext.domain

	short_IRI = ''

	if e.split('#') == [e] :
		if e.split('/') != [e]:
			name = e.split('/')[-1]
	else:
		name = e.split('/')[-1]

	if len (name) < 10:
		short_IRI  = namespace + ':' + name # this can be shortened !!!
	else:
		short_IRI = namespace + ':' + name[:10] + '...' # this can be shortened !!!

	print ('\n\noriginal e = ', e)
	print ('short_IRI =  ', short_IRI)

	return short_IRI



class SameAsEqInputGraph(SameAsEqGraph):

	def __init__(self, path_to_graph):
		super().__init__()
		self.graph = nx.Graph()
		got_data = pd.read_csv(path_to_graph)
		sources = got_data['SUBJECT']
		targets = got_data['OBJECT']
		# simplify the label
		sources_simplified = map(simp, sources)
		targets_simplified = map(simp, targets)
		edge_data = zip(sources_simplified, targets_simplified)
		for (s,t) in edge_data:
			# print ('source: ', s)
			# print ('simplified source: ', simp(s))
			# print ('target: ', t)
			self.graph.add_edge(s, t)


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
