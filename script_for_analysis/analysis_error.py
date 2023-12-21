
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



def get_namespace_prefix (e):
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
collect_error_edges = set()
collect_edges = set()

count_error_edges = 0
count_correct_edges = 0

count_edges_involving_disambiguation = 0
count_edges_involving_disambiguation_error = 0
count_edges_involving_disambiguation_correct = 0
total_count_remain_error_edges = 0

for id in gs:
	print ('\n***************\nGraph ID =', id,'\n')
	dir = './gold/'
	path_to_nodes = dir + str(id) +'.tsv'
	path_to_edges = dir + str(id) +'_edges.tsv'
	g = load_graph(path_to_nodes, path_to_edges)
	print ('loaded ', g.number_of_nodes(), ' nodes and ', g.number_of_edges(), ' edges')
	count_total_nodes += g.number_of_nodes()
	count_total_edges += g.number_of_edges()
	# the num of erorrneous edges
	for e in g.edges():
		collect_edges.add(e)

	for (s, t) in g.edges():
		if (g.nodes[s]['annotation'] != 'unknown'
			and g.nodes[t]['annotation'] != 'unknown'
			and g.nodes[s]['annotation'] != g.nodes[t]['annotation']):
			count_error_edges += 1
			collect_error_edges.add((s, t))
		if (g.nodes[s]['annotation'] != 'unknown'
			and g.nodes[t]['annotation'] != 'unknown'
			and g.nodes[s]['annotation'] == g.nodes[t]['annotation']):
			count_correct_edges +=1
	print ('there are in total ', count_error_edges, ' errorous edges ')
	count_total_error_edges += count_error_edges

	# if the edges of disambiguates has higher rate of error than normal

	for (s, t) in g.edges():

		if g.nodes[s]['comment'] == 'disambiguation' or g.nodes[t]['comment'] == 'disambiguation':
			count_edges_involving_disambiguation += 1
			if (g.nodes[s]['annotation'] != 'unknown'
				and g.nodes[t]['annotation'] != 'unknown'
				and g.nodes[s]['annotation'] != g.nodes[t]['annotation']):
				count_edges_involving_disambiguation_error += 1

			if (g.nodes[s]['annotation'] != 'unknown'
				and g.nodes[t]['annotation'] != 'unknown'
				and g.nodes[s]['annotation'] == g.nodes[t]['annotation']):
				count_edges_involving_disambiguation_correct += 1

	# How many erronous edges remains after removing unknown and disambiguation nodes
	collection_disambiguation = set()
	collection_unknown = set()
	for n in g.nodes():
		if g.nodes[n]['annotation'] == 'unknown':
			collection_unknown.add(n)
		elif g.nodes[n]['comment'] == 'disambiguation':
			collection_disambiguation.add(n)
	g.remove_nodes_from(list(collection_disambiguation))
	g.remove_nodes_from(list(collection_unknown))
	print ('after removing', len (collection_disambiguation), ' disambiguation entities and ')
	print ('and ', len (collection_unknown), ' unknown entities')
	print ('there are ', g.number_of_nodes(), ' nodes')
	print ('there are ', g.number_of_edges(), ' edges')

	count_remain_error_edges = 0
	for (s, t) in g.edges():
		if (g.nodes[s]['annotation'] != 'unknown'
			and g.nodes[t]['annotation'] != 'unknown'
			and g.nodes[s]['annotation'] != g.nodes[t]['annotation']):
			count_remain_error_edges += 1

	print ('num of count_remain_error_edges = ', count_remain_error_edges)
	total_count_remain_error_edges += count_remain_error_edges





print ('In total, there are ', len(validation_set), 'files for validation\n')
print ('There are in total ', count_total_nodes, ' nodes in the validation graphs')
print ('There are in total ', count_total_edges, ' edges in the validation graphs\n')
print ('There are in total ', count_total_error_edges, ' error edges in the validation graphs\n')
print ('\t {:10.2f} %'.format(100*count_total_error_edges/count_total_edges))
print ('There are in total ', count_total_redi_nodes, ' nodes in the redirect graphs')
print ('There are in total ', count_total_redi_edges, ' edges in the redirect graphs\n')
print ('There are in total ', count_total_ee_edges, ' edges in the graph of encoding equivalence')

print (count_nodes_with_explicit_source, ' has explicit sources: {:10.2f} %'.format(100*count_nodes_with_explicit_source/count_total_nodes))
print (count_nodes_with_implicit_label_source, ' has implicit label-like sources: {:10.2f} %'.format(100*count_nodes_with_implicit_label_source/count_total_nodes))
print (count_nodes_with_implicit_comment_source, ' has implicit comment-like sources: {:10.2f} %'.format(100*count_nodes_with_implicit_comment_source/count_total_nodes))

prefix_ct_error = Counter()
for (s, t) in collect_error_edges:
	# find where they are from
	prefix_ct_error[get_namespace_prefix(s)] += 1
	prefix_ct_error[get_namespace_prefix(t)] += 1

prefix_ct = Counter()
for (s, t) in collect_edges:
	# find where they are from
	prefix_ct[get_namespace_prefix(s)] += 1
	prefix_ct[get_namespace_prefix(t)] += 1

prefix_error_rate = {}
for prefix in prefix_ct_error.keys():
	pct = prefix_ct_error[prefix]/prefix_ct[prefix]
	prefix_error_rate[prefix] = pct

prefix_error_rate = {k: v for k, v in sorted(prefix_error_rate.items(), key=lambda item: item[1])}

# for p in prefix_error_rate:
# 	print (p)
# 	print ('count edges: ',prefix_ct[p])
# 	print ('count error edges: ', prefix_ct_error[p])
# 	print (' gives pct error rate ', prefix_error_rate[p])
# 	print ('\n')

prefix_pair_ct = Counter()
for (s, t) in collect_edges:
	ps = get_namespace_prefix(s)
	pt = get_namespace_prefix(t)
	if ps > pt :
		prefix_pair_ct[(pt, ps)] += 1
	else:
		prefix_pair_ct[(ps, pt)] += 1

prefix_pair_ct_error = Counter()
for (s, t) in collect_error_edges:
	ps = get_namespace_prefix(s)
	pt = get_namespace_prefix(t)
	if ps > pt :
		prefix_pair_ct_error[(pt, ps)] += 1
	else:
		prefix_pair_ct_error[(ps, pt)] += 1

prefix_error_rate = {}
for pair in prefix_pair_ct.keys():
	pct = prefix_pair_ct_error[pair] / prefix_pair_ct[pair]
	prefix_error_rate[pair] = pct


prefix_error_rate = {k: v for k, v in sorted(prefix_error_rate.items(), key=lambda item: item[1])}

for pair in prefix_error_rate.keys():
	if prefix_error_rate[pair] >= 0.10: # prefix_pair_ct[pair] >= 5 and
		print (pair,',')
		# print ('pair = ', pair)
		# print ('prefix pair count ', prefix_pair_ct[pair])
		# print ('prefix pair error count ', prefix_pair_ct_error[pair])
		# print ('error rate = ', prefix_error_rate[pair])
		# print ('\n')

print ('overal error:', len (collect_error_edges))
print ('overal total',len (collect_edges))
print ('overall error rate ', count_error_edges / len (collect_edges))
print ('overall correct rate ', count_correct_edges / len (collect_edges))

print ('disambiguation total', count_edges_involving_disambiguation)
print ('disambiguation error', count_edges_involving_disambiguation_error)
print('disambiguation correct',count_edges_involving_disambiguation_correct)
print ('when disambiguation nodes are involved, error rate: ', count_edges_involving_disambiguation_error/ count_edges_involving_disambiguation)
print ('when disambiguation nodes are involved, correct rate: ', count_edges_involving_disambiguation_correct/ count_edges_involving_disambiguation)

print ('*'*20)
print ('how many pct of errors are due to disambiguation? ')
print (count_edges_involving_disambiguation_correct/count_error_edges)

print ('the pct of errors after removing disambiguation (and unknown) nodes')
print (total_count_remain_error_edges/len (collect_error_edges))
