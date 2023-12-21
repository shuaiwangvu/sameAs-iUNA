# In this class, we provide the utility functions that are used
# in the preprocessing and the GraphSolver class
# These functions are made to handle URIs and import/export of graphs
# Plaese contac the authors in case of any mistake
import networkx as nx
import pandas as pd
import tldextract
import csv
from hdt import HDTDocument, IdentifierPosition
from rfc3987 import  parse


def get_authority (e):
	return parse(e)['authority']

def get_simp_IRI(e):
	# simplify this uri by introducing the namespace abbreviation
	ext = tldextract.extract(e)
	# ExtractResult(subdomain='af', domain='dbpedia', suffix='org')

	if 'dbpedia' == ext.domain and ext.subdomain != '' and ext.subdomain != None:
		namespace = ext.subdomain +'.'+ext.domain
	else :
		namespace = ext.domain

	short_IRI = ''

	if e.split('/') == [e] :
		if e.split('#') != [e]:
			name = e.split('#')[-1]
	else:
		name = e.split('/')[-1]

	if len (name) < 10:
		short_IRI  = namespace + ':' + name
	else:
		short_IRI = namespace + ':' + name[:10] + '...'

	return short_IRI

def get_prefix (e):
	prefix, name, sign = get_name(e)
	return prefix


def get_name (e):
	name = ''
	namespace = ''
	sign = ''
	if e.rfind('/') == -1 : # the char '/' is not in the iri
		if e.split('#') != [e]: # but the char '#' is in the iri
			name = e.split('#')[-1]
			namespace = '#'.join(e.split('#')[:-1]) + '#'
			sign = '#'
		else:
			name = None
			sign = None
			namespace =  None
	else:
		name = e.split('/')[-1]
		namespace = '/'.join(e.split('/')[:-1]) + '/'
		sign = '/'

	return namespace, sign, name



def load_undi_graph (nodes_filename, edges_filename): # load undirected graph
	g = nx.Graph()
	with open(nodes_filename, 'r') as nodes_file:
	# nodes_file = open(nodes_filename, 'r')
		reader = csv.DictReader(nodes_file, delimiter='\t',)
		for row in reader:
			s = row["Entity"]
			a = row["Annotation"]
			c = row["Comment"]
			g.add_node(s, annotation = a, comment = c)
			g.nodes[s]['prefix'] = get_prefix(s)
		edges_file = open(edges_filename, 'r')
		reader = csv.DictReader(edges_file, delimiter='\t',)
		for row in reader:
			s = row["SUBJECT"]
			t = row["OBJECT"]
			id = row["METALINK_ID"]
			err = row["ERROR_DEGREE"]
			try:
				index = err.index('^')
				index -= 1
				error_rate = float(err[1:index])
			except Exception as e:
				print(row["ERROR_DEGREE"][2:5])
				print (row["ERROR_DEGREE"])
				raise

			if s!=t:
				g.add_edge(s, t, metalink_id = id, metalink_error_rate = error_rate, weight = 0)
			else:
				print ('FOUND reflexive EDGES!')
		nodes_file.close()
		edges_file.close()
		return g

def load_graph (nodes_filename, edges_filename):
	g = nx.DiGraph()
	with open(nodes_filename, 'r') as nodes_file:
		reader = csv.DictReader(nodes_file, delimiter='\t',)
		for row in reader:
			s = row["Entity"]
			a = row["Annotation"]
			c = row["Comment"]
			g.add_node(s, annotation = a, comment = c)

		edges_file = open(edges_filename, 'r')
		reader = csv.DictReader(edges_file, delimiter='\t',)
		for row in reader:
			s = row["SUBJECT"]
			t = row["OBJECT"]
			id = row["METALINK_ID"]
			err = row["ERROR_DEGREE"]
			try:
				index = err.index('^')
				index -= 1
				error_rate = float(err[1:index])
			except Exception as e:
				print(row["ERROR_DEGREE"][2:5])
				print (row["ERROR_DEGREE"])
				raise

			# print (error_rate)
			if s!=t:
				g.add_edge(s, t, metalink_id = id, metalink_error_rate = error_rate, weight = 0)
			else:
				print ('FOUND reflexive EDGES!')
		nodes_file.close()
		edges_file.close()
		return g




def load_edge_weights (path_to_edge_weights, graph):
	# print ('loading weights... ')
	with open(path_to_edge_weights, 'r') as edge_weights_file:
		reader = csv.DictReader(edge_weights_file, delimiter='\t',)
		for row in reader:
			s = row["SUBJECT"]
			t = row["OBJECT"]
			w = row["WEIGHT"]
			# print ('weight = ', w)
			# if (s, t) in graph.edges():
			# 	graph[s][t]['weight'] = int (w)
			if (s, t) in graph.edges ():
				graph[s][t]['weight'] += int (w)
				# if (s, t, {}) in graph.edges(data = True):
				# 	graph[s][t]['weight'] = int (w)
				# else:
				# 	graph[s][t]['weight'] += int (w)
			# else:
			# 	print('this edge is not there')
		edge_weights_file.close()




def load_explicit (path_to_explicit_source, graph):
	hdt_explicit = HDTDocument(path_to_explicit_source)
	for e in graph.nodes:
		graph.nodes[e]['explicit_source'] = []
	for e in graph.nodes:
		(triples, cardi) = hdt_explicit.search_triples(e, "", "")
		for (e, _, s) in triples:
			graph.nodes[e]['explicit_source'].append(s)


def load_implicit_label_source (path_to_implicit_label_source, graph):
	try:
		hdt_implicit_label = HDTDocument(path_to_implicit_label_source)
	except Exception as e:
		print ('FAIL while loading file ', path_to_implicit_label_source)
		raise

	for e in graph.nodes:
		graph.nodes[e]['implicit_label_source'] = []
	for e in graph.nodes:
		(triples, cardi) = hdt_implicit_label.search_triples(e, "", "")
		for (e, _, s) in triples:
			graph.nodes[e]['implicit_label_source'].append(s)


def load_implicit_comment_source (path_to_implicit_comment_source, graph):
	try:
		hdt_implicit_comment = HDTDocument(path_to_implicit_comment_source)
	except Exception as e:
		print ('FAIL while loading file ', path_to_implicit_comment_source)
		raise

	for e in graph.nodes:
		graph.nodes[e]['implicit_comment_source'] = []
	for e in graph.nodes:
		(triples, cardi) = hdt_implicit_comment.search_triples(e, "", "")
		for (e, _, s) in triples:
			graph.nodes[e]['implicit_comment_source'].append(s)

def load_encoding_equivalence (path_ee):
	ee_g = nx.Graph()
	hdt_ee = HDTDocument(path_ee)
	(triples, cardi) = hdt_ee.search_triples("", "", "")
	for (s,_,t) in triples:
		ee_g.add_edge(s, t)
	return ee_g

def load_redi_graph(path_to_redi_graph_nodes, path_to_redi_graph_edges):
	redi_g = nx.DiGraph()

	# print('loading redi_graph at ', path_to_redi_graph_edges)
	# try:
	# 	hdt_redi_edges = HDTDocument(path_to_redi_graph_edges)
	# except Exception as e:
	# 	print ('error while opening redi file ', path_to_redi_graph_edges)
	# 	raise
	#
	# (triples, cardi) = hdt_redi_edges.search_triples("", "", "")
	# for (s,_,t) in triples:
	# 	redi_g.add_edge(s,t)



	nodes_file = open(path_to_redi_graph_nodes, 'r')
	reader = csv.DictReader(nodes_file, delimiter='\t',)
	for row in reader:
		s = row["Entity"]
		r = row["Remark"]
		if s in redi_g.nodes():
			redi_g.add_node(s, remark = r)
	nodes_file.close()

	redi_file = open(path_to_redi_graph_edges, 'r')
	for l in redi_file:
		source = l.split(' ')[0]
		target = l.split(' ')[1]
		source = source[1:][:-1]
		target = target[1:][:-1]
		if source not in redi_g.nodes():
			redi_g.add_node(source, remark = 'newly found through redirection')
		if target not in redi_g.nodes():
			redi_g.add_node(target, remark = 'newly found through redirection')
		# print ("source ", source)
		# print ("target ", target)
		redi_g.add_edge(source, target)
	redi_file.close()

	return redi_g


def load_disambiguation_entities(nodes, path_to_disambiguation_entities):
	# sameas_disambiguation_entities.hdt
	hdt = HDTDocument(path_to_disambiguation_entities)
	entities = set()
	for n in nodes:
		(triples, cardi) = hdt.search_triples(n, "", "")
		if cardi > 0:
			entities.add(n)

	return list(entities)
