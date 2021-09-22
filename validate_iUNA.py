
import pandas as pd
import numpy as np
import datetime
import pickle
import time
import networkx as nx
import sys
import csv
from z3 import *
from bidict import bidict
import matplotlib.pyplot as plt
import tldextract
import json
import random
from collections import Counter
from hdt import HDTDocument, IdentifierPosition
import glob
from urllib.parse import urlparse
import gzip
from extend_metalink import *
import requests
from requests.exceptions import Timeout



# there are in total 28 entities. 14 each
validate_single = [96073, 712342, 9994282, 18688, 1140988, 25604]
validate_multiple = [6617, 4170, 42616, 39036, 33122, 6927, 11116, 12745]
validation_set = validate_single + validate_multiple

evaluation_single = [9411, 9756, 97757, 99932, 337339, 1133953]
evaluation_multiple = [5723, 14872, 37544, 236350, 240577, 395175, 4635725, 14514123]
evaluation_set = evaluation_single + evaluation_multiple


gs = validation_set + evaluation_set


def find_statement_id(subject, object):

	triples, cardinality = hdt_metalink.search_triples("", rdf_subject, subject)
	collect_statement_id_regarding_subject = set()

	for (s,p,o) in triples:
		collect_statement_id_regarding_subject.add(str(s))

	triples, cardinality = hdt_metalink.search_triples("", rdf_object, object)

	collect_statement_id_regarding_object = set()

	for (s,p,o) in triples:
		collect_statement_id_regarding_object.add(str(s))

	inter_section = collect_statement_id_regarding_object.intersection(collect_statement_id_regarding_subject)

	# do it the reverse way: (object, predicate, subject)
	triples, cardinality = hdt_metalink.search_triples("", rdf_object, subject)
	collect_statement_id_regarding_subject = set()

	for (s,p,o) in triples:
		collect_statement_id_regarding_subject.add(str(s))

	triples, cardinality = hdt_metalink.search_triples("", rdf_subject, object)

	collect_statement_id_regarding_object = set()

	for (s,p,o) in triples:
		collect_statement_id_regarding_object.add(str(s))

	inter_section2 = collect_statement_id_regarding_object.intersection(collect_statement_id_regarding_subject)

	if len (inter_section) >= 1:
		return list(inter_section)[0] #
	elif len (inter_section2) >= 1:
		# print ('\nfound one in reverse!: \n', subject, '\t', object)
		return list(inter_section2)[0] #:
	else:
		return None


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
		g.add_edge(s, t, metalink_id = id)

	return g

def load_redi_graph(path_to_redi_graph_nodes, path_to_redi_graph_edges):
	redi_g = nx.DiGraph()
	nodes_file = open(path_to_redi_graph_nodes, 'r')
	reader = csv.DictReader(nodes_file, delimiter='\t',)
	for row in reader:
		s = row["Entity"]
		r = row["Remark"]
		redi_g.add_node(s, remark = r)

	hdt_redi_edges = HDTDocument(path_to_redi_graph_edges)
	(triples, cardi) = hdt_redi_edges.search_triples("", "", "")
	for (s,_,t) in triples:
		redi_g.add_edge(s,t)
	return redi_g

def load_explicit (path_to_explicit_source, graph):
	hdt_explicit = HDTDocument(path_to_explicit_source)
	for e in graph.nodes:
		graph.nodes[e]['explicit_source'] = []
	for e in graph.nodes:
		(triples, cardi) = hdt_explicit.search_triples(e, "", "")
		for (e, _, s) in triples:
			graph.nodes[e]['explicit_source'].append(s)

	ct = Counter()
	for e in graph.nodes:
		sources = graph.nodes[e]['explicit_source']
		ct[len(sources)] += 1
	# for c in ct:
	# 	print (c ,' - ', ct[c])
	return ct


def load_implicit_label_source (path_to_implicit_label_source, graph):
	hdt_implicit_label = HDTDocument(path_to_implicit_label_source)
	for e in graph.nodes:
		graph.nodes[e]['implicit_label_source'] = []
	for e in graph.nodes:
		(triples, cardi) = hdt_implicit_label.search_triples(e, "", "")
		for (e, _, s) in triples:
			graph.nodes[e]['implicit_label_source'].append(s)

	ct = Counter()
	for e in graph.nodes:
		sources = graph.nodes[e]['implicit_label_source']
		ct[len(sources)] += 1
	# for c in ct:
	# 	print (c , ' - ', ct[c])
	return ct

def load_implicit_comment_source (path_to_implicit_comment_source, graph):
	hdt_implicit_comment = HDTDocument(path_to_implicit_comment_source)
	for e in graph.nodes:
		graph.nodes[e]['implicit_comment_source'] = []
	for e in graph.nodes:
		(triples, cardi) = hdt_implicit_comment.search_triples(e, "", "")
		for (e, _, s) in triples:
			graph.nodes[e]['implicit_comment_source'].append(s)

	ct = Counter()
	for e in graph.nodes:
		sources = graph.nodes[e]['implicit_comment_source']
		ct[len(sources)] += 1
	# for c in ct:
	# 	print (c, ' - ', ct[c])
	return ct


print ('in the validation dataset, there are ', validation_set, ' files (connected components)')

count_total_nodes = 0
count_nodes_with_explicit_source = 0
count_nodes_with_implicit_label_source = 0
count_nodes_with_implicit_comment_source = 0

for id in validation_set:
	print ('\n***************\nGraph ID =', id,'\n')
	dir = './gold/'
	path_to_nodes = dir + str(id) +'.tsv'
	path_to_edges = dir + str(id) +'_edges.tsv'
	g = load_graph(path_to_nodes, path_to_edges)
	print ('loaded ', g.number_of_nodes(), ' nodes and ', g.number_of_edges(), ' edges')
	count_total_nodes += g.number_of_nodes()
	# the num of erorrneous edges
	count_error_edges = 0
	for (s, t) in g.edges():
		if (g.nodes[s]['annotation'] != 'unknown'
			and g.nodes[t]['annotation'] != 'unknown'
			and g.nodes[s]['annotation'] != g.nodes[t]['annotation']):
			count_error_edges += 1
	print ('there are in total ', count_error_edges, ' errorous edges ')


	path_to_redi_graph_nodes = dir + str(id) +'_redirect_nodes.tsv'
	path_to_redi_graph_edges = dir + str(id) +'_redirect_edges.hdt'
	redi_graph = load_redi_graph(path_to_redi_graph_nodes, path_to_redi_graph_edges)
	print ('loaded the redi graph with ', redi_graph.number_of_nodes(), 'nodes and ', redi_graph.number_of_edges(), ' edges')

	print ('*'*20)

	# load explicit source
	path_to_explicit_source = dir + str(id) + '_explicit_source.hdt'
	ct_explicit = load_explicit(path_to_explicit_source, g)
	for c in ct_explicit.keys():
		if c != 0:
			count_nodes_with_explicit_source += ct_explicit[c]

	# load implicit label-like source
	path_to_implicit_label_source = dir + str(id) + '_implicit_label_source.hdt'
	ct_label = load_implicit_label_source(path_to_implicit_label_source, g)
	for c in ct_label.keys():
		if c != 0:
			count_nodes_with_implicit_label_source += ct_label[c]

	# load implicit comment-like source
	path_to_implicit_comment_source = dir + str(id) + '_implicit_comment_source.hdt'
	ct_comment = load_implicit_comment_source(path_to_implicit_comment_source, g)
	for c in ct_comment.keys():
		if c != 0:
			count_nodes_with_implicit_comment_source += ct_comment[c]

print ('There are in total ', count_total_nodes, ' nodes in the validation graphs')
print (count_nodes_with_explicit_source, ' has explicit sources')
print (count_nodes_with_implicit_label_source, ' has implicit label-like sources')
print (count_nodes_with_implicit_comment_source, ' has implicit comment-like sources')
