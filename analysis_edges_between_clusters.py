
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
all_multiple =  validate_multiple + evaluation_multiple

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
	g = nx.Graph()
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

def load_encoding_equivalence (path_ee, graph):
	ee_g = nx.Graph()
	hdt_ee = HDTDocument(path_ee)
	(triples, cardi) = hdt_ee.search_triples("", "", "")
	for (s,_,t) in triples:
		ee_g.add_edge(s, t)
	return ee_g

print ('in the validation dataset, there are ', validation_set, ' files (connected components)')

count_total_nodes = 0
count_total_edges = 0
count_total_error_edges = 0

count_total_redi_nodes = 0
count_total_redi_edges = 0
count_nodes_with_explicit_source = 0
count_nodes_with_implicit_label_source = 0
count_nodes_with_implicit_comment_source = 0
count_total_ee_edges = 0
id_to_graph = {}
for id in all_multiple:
	print ('\n***************\nGraph ID =', id,'\n')
	dir = './gold/'
	path_to_nodes = dir + str(id) +'.tsv'
	path_to_edges = dir + str(id) +'_edges.tsv'
	g = load_graph(path_to_nodes, path_to_edges)
	print ('loaded ', g.number_of_nodes(), ' nodes and ', g.number_of_edges(), ' edges')
	count_total_nodes += g.number_of_nodes()
	count_total_edges += g.number_of_edges()

	# the num of erorrneous edges
	annoation_to_nodes = {}
	for n in g.nodes():
		a = g.nodes[n]['annotation']
		if a in annoation_to_nodes.keys():
			annoation_to_nodes[a].append(n)
		else:
			annoation_to_nodes[a] = [n]
	# find the clusters with >=4 nodes
	nodes_to_ignore = set()
	nodes_in_clusters = set()
	annotation_in_clusters = []
	for a in annoation_to_nodes.keys():
		if a == 'unknown' or len(annoation_to_nodes[a]) < 4:
			nodes_to_ignore = nodes_to_ignore.union(annoation_to_nodes[a])
		else:
			annotation_in_clusters.append(a)
			nodes_in_clusters = nodes_in_clusters.union(annoation_to_nodes[a])

	for a in annoation_to_nodes.keys():
		if len(annoation_to_nodes[a]) >= 4:
			print ('for ', a, ' there are ', len(annoation_to_nodes[a]), ' nodes')

	print ('there are ', len(nodes_to_ignore), ' nodes to ignore')
	print ('there are ', len(nodes_in_clusters), ' nodes in clusters to study')
	print ('there are in total ', len(annotation_in_clusters),' (major) annotations')

	# count_pair = Counter()
	# for edge in g.edges():
	# 	# edges
	# 	(s, t) = edge
	# 	if g.nodes[s]['annotation'] != g.nodes[t]['annotation']:
	# 		if g.nodes[s]['annotation'] > g.nodes[t]['annotation']:
	# 			# print (g.nodes[s]['annotation'], ' - ', g.nodes[t]['annotation'])
	# 			count_pair[(g.nodes[s]['annotation'], g.nodes[t]['annotation'])] += 1
	# 		else:
	# 			count_pair[(g.nodes[t]['annotation'], g.nodes[s]['annotation'])] += 1
	# for p in count_pair:
	# 	print ('between ', p, ' there are  ', count_pair[p], ' edges')
	#
	# print ('now remove nodes to ignore')
	# g.remove_nodes_from(list(nodes_to_ignore))
	#
	# count_remain_error = 0
	# count_pair = Counter()
	# for edge in g.edges():
	# 	# edges
	# 	(s, t) = edge
	# 	if g.nodes[s]['annotation'] != g.nodes[t]['annotation']:
	# 		count_remain_error += 1
	# 		if g.nodes[s]['annotation'] > g.nodes[t]['annotation']:
	# 			# print (g.nodes[s]['annotation'], ' - ', g.nodes[t]['annotation'])
	# 			count_pair[(g.nodes[s]['annotation'], g.nodes[t]['annotation'])] += 1
	# 		else:
	# 			count_pair[(g.nodes[t]['annotation'], g.nodes[s]['annotation'])] += 1
	# for p in count_pair:
	# 	print ('between ', p, ' there are  ', count_pair[p], ' edges')
	# print ('remaining error ', count_remain_error)

	overall_avg = 0
	for n in g.nodes():

		if g.nodes[n]['annotation'] == 'unknown':
			# print ('for node n ', n)
			set_annotations = set()
			# print ('edges = ', g.edges(n))
			for (l, r) in g.edges(n):
				if g.nodes[r]['annotation'] != 'unknown':
					set_annotations.add(r)
			avg = len (set_annotations) / g.degree(n)
			overall_avg += avg

	# print (overall_avg)
	overall_avg /= g.number_of_nodes()
	print ('for unknown nodes, the average number of annotation it connects with is ', overall_avg)
	print ('Todo: What about overall average?')
