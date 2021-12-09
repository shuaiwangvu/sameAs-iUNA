# this is an abstract class
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
	prefix = ''
	sign = ''
	if e.rfind('/') == -1 : # the char '/' is not in the iri
		if e.split('#') != [e]: # but the char '#' is in the iri
			name = e.split('#')[-1]
			prefix = '#'.join(e.split('#')[:-1]) + '#'
			sign = '#'
		else:
			name = None
			sign = None
			prefix =  None
	else:
		name = e.split('/')[-1]
		prefix = '/'.join(e.split('/')[:-1]) + '/'
		sign = '/'

	return prefix, sign, name



def load_undi_graph (nodes_filename, edges_filename): # load undirected graph
	g = nx.Graph()
	nodes_file = open(nodes_filename, 'r')
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
		if s!=t:
			g.add_edge(s, t, metalink_id = id)
		else:
			print ('FOUND reflexive EDGES!')
	nodes_file.close()
	edges_file.close()
	return g

def load_graph (nodes_filename, edges_filename):
	g = nx.DiGraph()
	nodes_file = open(nodes_filename, 'r')
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
		if s!=t:
			g.add_edge(s, t, metalink_id = id)
		else:
			print ('FOUND reflexive EDGES!')
	nodes_file.close()
	edges_file.close()
	return g


def load_edge_weights (path_to_edge_weights, graph):
	# print ('loading weights... ')
	edge_weights_file = open(path_to_edge_weights, 'r')
	reader = csv.DictReader(edge_weights_file, delimiter='\t',)
	for row in reader:
		s = row["SUBJECT"]
		t = row["OBJECT"]
		w = row["WEIGHT"]
		# print ('weight = ', w)
		if (s, t) in graph.edges():
			graph[s][t]['weight'] = int (w)
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
	hdt_implicit_label = HDTDocument(path_to_implicit_label_source)
	for e in graph.nodes:
		graph.nodes[e]['implicit_label_source'] = []
	for e in graph.nodes:
		(triples, cardi) = hdt_implicit_label.search_triples(e, "", "")
		for (e, _, s) in triples:
			graph.nodes[e]['implicit_label_source'].append(s)


def load_implicit_comment_source (path_to_implicit_comment_source, graph):
	hdt_implicit_comment = HDTDocument(path_to_implicit_comment_source)
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
	hdt_redi_edges = HDTDocument(path_to_redi_graph_edges)
	(triples, cardi) = hdt_redi_edges.search_triples("", "", "")
	for (s,_,t) in triples:
		redi_g.add_edge(s,t)

	nodes_file = open(path_to_redi_graph_nodes, 'r')
	reader = csv.DictReader(nodes_file, delimiter='\t',)
	for row in reader:
		s = row["Entity"]
		r = row["Remark"]
		if s in redi_g.nodes():
			redi_g.add_node(s, remark = r)
	nodes_file.close()
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
