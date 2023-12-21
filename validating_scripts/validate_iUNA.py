# this script tests the correctness of redirect and encoding equivalence
# validate_iUNA2 tests the correctness of randomly sampled edges.

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
from SameAsEqGraph import *


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
		g.nodes[s]['prefix'] = get_prefix(s)
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
count_total_correct_edges = 0
# ****** redirecting *****
count_total_redi_nodes = 0
count_total_redi_nodes_new = 0
count_total_redi_edges = 0

count_total_redi_edges_existing = 0
count_total_redi_edges_existing_correct = 0
count_total_redi_edges_existing_error = 0

count_total_redi_edges_not_existing = 0
count_total_redi_edges_not_existing_correct = 0
count_total_redi_edges_not_existing_error = 0

count_total_pairs_redi = 0
count_total_pairs_redi_correct = 0
count_total_pairs_redi_error = 0
# ***** source *******
count_nodes_with_explicit_source = 0
count_nodes_with_implicit_label_source = 0
count_nodes_with_implicit_comment_source = 0
# ****** Encoding equivalence *******
count_total_ee_edges = 0
count_total_ee_edges_existing = 0
count_total_ee_edges_existing_correct = 0
count_total_ee_edges_existing_error = 0
count_total_ee_edges_not_existing = 0
count_total_ee_edges_not_existing_correct = 0
count_total_ee_edges_not_existing_error = 0
count_total_pairs_ee = 0
count_total_pairs_ee_correct = 0
count_total_pairs_ee_error = 0
# ****** Validating iUNA *******
list_same_prefix_same_anno = []
list_same_prefix_diff_anno = []

list_same_labelsource_same_anno = []
list_same_labelsource_diff_anno = []

list_same_commentsource_same_anno = []
list_same_commentsource_diff_anno = []

list_same_explicitsource_same_anno = []
list_same_explicitsource_diff_anno = []


# list_same_condition_same_anno_with_redirect = []
# list_same_condition_diff_anno_with_redirect = []
# list_same_condition_same_anno_with_ee = []
# list_same_condition_diff_anno_with_ee = []


id_to_graph = {}
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
	count_error_edges = 0
	count_correct_edges = 0
	for (s, t) in g.edges():
		if (g.nodes[s]['annotation'] != 'unknown'
			and g.nodes[t]['annotation'] != 'unknown'
			and g.nodes[s]['annotation'] != g.nodes[t]['annotation']):
			count_error_edges += 1
		if (g.nodes[s]['annotation'] != 'unknown'
			and g.nodes[t]['annotation'] != 'unknown'
			and g.nodes[s]['annotation'] == g.nodes[t]['annotation']):
			count_correct_edges += 1
	print ('there are in total ', count_error_edges, ' errorous edges ')
	count_total_error_edges += count_error_edges
	count_total_correct_edges += count_correct_edges

	# redirection
	in_use_redirect_graph = nx.Graph()
	path_to_redi_graph_nodes = dir + str(id) +'_redirect_nodes.tsv'
	path_to_redi_graph_edges = dir + str(id) +'_redirect_edges.hdt'
	redi_graph = load_redi_graph(path_to_redi_graph_nodes, path_to_redi_graph_edges)
	print ('loaded the redi graph with ', redi_graph.number_of_nodes(), 'nodes and ', redi_graph.number_of_edges(), ' edges')
	count_total_redi_nodes += redi_graph.number_of_nodes()
	count_total_redi_edges += redi_graph.number_of_edges()
	for n in redi_graph.nodes():
		if n not in g.nodes():
			count_total_redi_nodes_new += 1
	for e in redi_graph.edges():
		(n,m) = e
		if e in g.edges():
			count_total_redi_edges_existing += 1
			in_use_redirect_graph.add_edge(n,m)
			if g.nodes[n]['annotation'] != 'unknown' and g.nodes[m]['annotation'] != 'unknown':
				if g.nodes[n]['annotation'] == g.nodes[m]['annotation']:
					count_total_redi_edges_existing_correct += 1
				if  g.nodes[n]['annotation'] != g.nodes[m]['annotation']:
					count_total_redi_edges_existing_error += 1

		elif e not in g.edges() and n in g.nodes() and m in g.nodes():
			in_use_redirect_graph.add_edge(n,m)
			count_total_redi_edges_not_existing += 1
			if g.nodes[n]['annotation'] != 'unknown' and g.nodes[m]['annotation'] != 'unknown':
				if g.nodes[n]['annotation'] == g.nodes[m]['annotation']:
					count_total_redi_edges_not_existing_correct += 1
				if  g.nodes[n]['annotation'] != g.nodes[m]['annotation']:
					count_total_redi_edges_not_existing_error += 1

	# convert it to a undirected graph
	redi_graph_undirected = nx.Graph(redi_graph)
	redi_nodes = g.nodes()
	redi_connect_pairs = []
	for i, n in enumerate(list(redi_nodes)[:-1]):
		for m in list(redi_nodes)[i+1:]:
			if n in redi_graph_undirected.nodes() and m in redi_graph_undirected.nodes():
				if (n,m) not in redi_graph_undirected.edges() and (n,m) not in g.edges():
					if nx.has_path(redi_graph_undirected, n, m):
						redi_connect_pairs.append((n,m))
						in_use_redirect_graph.add_edge(n, m)

	print ('# pairs that redirect to the same entity ', len (redi_connect_pairs))
	pair_correct = 0
	pair_error = 0
	for (n,m) in redi_connect_pairs:
		if g.nodes[n]['annotation'] != 'unknown' and g.nodes[m]['annotation'] != 'unknown':
			if g.nodes[n]['annotation'] == g.nodes[m]['annotation']:
				pair_correct += 1
			if  g.nodes[n]['annotation'] != g.nodes[m]['annotation']:
				pair_error += 1

	print ('pair correct = ', pair_correct)
	print ('pair error = ', pair_error)

	count_total_pairs_redi += len (redi_connect_pairs)
	count_total_pairs_redi_correct += pair_correct
	count_total_pairs_redi_error += pair_error


	# validating iUNA without encoding equivalence and redirect
	# load grpah of encoding equivalence
	in_use_ee_graph = nx.Graph()
	path_ee = dir + str(id) + '_encoding_equivalent.hdt'
	ee_graph = load_encoding_equivalence(path_ee, g)
	count_total_ee_edges += ee_graph.number_of_edges()
	for e in ee_graph.edges():
		# graph
		if e in g.edges():
			(n,m) = e
			count_total_ee_edges_existing += 1
			in_use_ee_graph.add_edge(n,m)
			if g.nodes[n]['annotation'] != 'unknown' and g.nodes[m]['annotation'] != 'unknown':
				if g.nodes[n]['annotation'] == g.nodes[m]['annotation']:
					count_total_ee_edges_existing_correct += 1
				if  g.nodes[n]['annotation'] != g.nodes[m]['annotation']:
					count_total_ee_edges_existing_error += 1
					# print (n, ' has annotation ', g.nodes[n]['annotation'])
					# print (m, ' has annotation ', g.nodes[m]['annotation'])
		elif e not in g.edges():
			(n,m) = e
			in_use_ee_graph.add_edge(n,m)
			count_total_ee_edges_not_existing += 1
			if g.nodes[n]['annotation'] != 'unknown' and g.nodes[m]['annotation'] != 'unknown':
				if g.nodes[n]['annotation'] == g.nodes[m]['annotation']:
					count_total_ee_edges_not_existing_correct += 1
				if  g.nodes[n]['annotation'] != g.nodes[m]['annotation']:
					count_total_ee_edges_not_existing_error += 1
	# convert it to a undirected graph
	ee_graph_undirected = nx.Graph(ee_graph)
	ee_nodes = g.nodes()
	ee_connect_pairs = []
	for i, n in enumerate(list(ee_nodes)[:-1]):
		for m in list(ee_nodes)[i+1:]:
			if n in ee_graph_undirected.nodes() and m in ee_graph_undirected.nodes():
				if (n,m) not in ee_graph_undirected.edges() and (n,m) not in g.edges():
					if nx.has_path(ee_graph_undirected, n, m):
						ee_connect_pairs.append((n,m))
						in_use_ee_graph.add_edge(n,m)

	print ('# pairs that encoding to the same entity ', len (ee_connect_pairs))
	ee_correct = 0
	ee_error = 0
	for (n,m) in ee_connect_pairs:
		if g.nodes[n]['annotation'] != 'unknown' and g.nodes[m]['annotation'] != 'unknown':
			if g.nodes[n]['annotation'] == g.nodes[m]['annotation']:
				ee_correct += 1
			if  g.nodes[n]['annotation'] != g.nodes[m]['annotation']:
				ee_error += 1

	print ('pair correct = ', ee_correct)
	print ('pair error = ', ee_error)

	count_total_pairs_ee += len (ee_connect_pairs)
	count_total_pairs_ee_correct += ee_correct
	count_total_pairs_ee_error += ee_error

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

	# next, validate the iUNA

	# Step 1: for every prefix, find all combinations
	prefix_to_entities = {}
	for n in g.nodes():
		p = get_prefix(n)
		if p not in prefix_to_entities.keys():
			prefix_to_entities[p] = [n]
		else:
			prefix_to_entities[p].append(n)
	samples = set()
	for p in prefix_to_entities.keys():
		if len (prefix_to_entities[p]) > 1:
			# for each prefix, we assemble n^2 / 3
			sample_for_this_prefix = set()
			sample_size = len (prefix_to_entities[p]) * len (prefix_to_entities[p]) / 3
			while len (sample_for_this_prefix) < sample_size:
				s = random.choice(prefix_to_entities[p])
				t = random.choice(prefix_to_entities[p])
				if s != t and (s, t) not in samples and (t, s) not in samples:
					sample_for_this_prefix.add((s,t))

			samples = samples.union(sample_for_this_prefix)

	count_same_condition_same_anno = 0
	count_same_condition_diff_anno = 0

	for (s,t) in samples:
		if g.nodes[s]['annotation'] != 'unknown' and  g.nodes[t]['annotation'] != 'unknown':
			if  g.nodes[s]['annotation'] ==  g.nodes[t]['annotation']:
				count_same_condition_same_anno += 1
			elif g.nodes[s]['annotation'] !=  g.nodes[t]['annotation']:
				count_same_condition_diff_anno += 1
	if len(samples) != 0:
		list_same_prefix_same_anno.append(count_same_condition_same_anno / len(samples))
		list_same_prefix_diff_anno.append(count_same_condition_diff_anno / len(samples))



	# Step 2-a: for every (label) source
	labelsource_to_entities = {}
	for n in g.nodes():
		p = g.nodes[n]['implicit_label_source']
		# p = g.nodes[n]['implicit_comment_source'] # the sources
		# pre = get_prefix(n)
		for l in p:
			if l not in labelsource_to_entities.keys():
				labelsource_to_entities[l] = [n]
			else:
				labelsource_to_entities[l].append(n)
	samples = []
	for p in labelsource_to_entities.keys():
		if len (labelsource_to_entities[p]) > 1:
			# print ('sampling from ', labelsource_to_entities[p])
			# for each prefix, we assemble n^2 / 3
			sample_for_this_labelsource = []
			sample_size = int(len (labelsource_to_entities[p]) * len (labelsource_to_entities[p]) / 3)
			# print ('amount expected: ',sample_size)
			# print ('', flush=True)
			while len (sample_for_this_labelsource) < sample_size:
				# print ('in the loop', flush=True)
				s = random.choice(labelsource_to_entities[p])
				t = random.choice(labelsource_to_entities[p])
				# print ('s = ',s)
				# print ('t = ',t)
				if s != t and (s, t) not in sample_for_this_labelsource and (t, s) not in sample_for_this_labelsource:
					sample_for_this_labelsource.append((s,t))
					# print ('\t loaded')

			samples = samples + sample_for_this_labelsource

	count_same_condition_same_anno = 0
	count_same_condition_diff_anno = 0

	for (s,t) in samples:
		if g.nodes[s]['annotation'] != 'unknown' and  g.nodes[t]['annotation'] != 'unknown':
			if  g.nodes[s]['annotation'] ==  g.nodes[t]['annotation']:
				count_same_condition_same_anno += 1
			elif g.nodes[s]['annotation'] !=  g.nodes[t]['annotation']:
				count_same_condition_diff_anno += 1
	if len(samples) != 0:
		list_same_labelsource_same_anno.append(count_same_condition_same_anno / len(samples))
		list_same_labelsource_diff_anno.append(count_same_condition_diff_anno / len(samples))
	print ('number of samples for labelsources ', len(samples))

	# Step 2-b: for every (comment) source
	commentsource_to_entities = {}
	for n in g.nodes():
		# p = g.nodes[n]['implicit_label_source']
		p = g.nodes[n]['implicit_comment_source'] # the sources
		# pre = get_prefix(n)
		for l in p:
			if l not in commentsource_to_entities.keys():
				commentsource_to_entities[l] = [n]
			else:
				commentsource_to_entities[l].append(n)
	samples = []
	for p in commentsource_to_entities.keys():
		if len (commentsource_to_entities[p]) > 1:
			# print ('sampling from ', labelsource_to_entities[p])
			# for each prefix, we assemble n^2 / 3
			sample_for_this_commentsource = []
			sample_size = int(len (commentsource_to_entities[p]) * len (commentsource_to_entities[p]) / 3)
			# print ('amount expected: ',sample_size)
			# print ('', flush=True)
			while len (sample_for_this_commentsource) < sample_size:
				# print ('in the loop', flush=True)
				s = random.choice(commentsource_to_entities[p])
				t = random.choice(commentsource_to_entities[p])
				# print ('s = ',s)
				# print ('t = ',t)
				if s != t and (s, t) not in sample_for_this_commentsource and (t, s) not in sample_for_this_commentsource:
					sample_for_this_commentsource.append((s,t))
					# print ('\t loaded')

			samples = samples + sample_for_this_commentsource

	count_same_condition_same_anno = 0
	count_same_condition_diff_anno = 0

	for (s,t) in samples:
		if g.nodes[s]['annotation'] != 'unknown' and  g.nodes[t]['annotation'] != 'unknown':
			if  g.nodes[s]['annotation'] ==  g.nodes[t]['annotation']:
				count_same_condition_same_anno += 1
			elif g.nodes[s]['annotation'] !=  g.nodes[t]['annotation']:
				count_same_condition_diff_anno += 1
	if len(samples) != 0:
		list_same_commentsource_same_anno.append(count_same_condition_same_anno / len(samples))
		list_same_commentsource_diff_anno.append(count_same_condition_diff_anno / len(samples))
	print ('number of samples for commentsources ', len(samples))

	# Step 2-c: for every (explicit) source
	explicitsource_to_entities = {}
	for n in g.nodes():
		# p = g.nodes[n]['implicit_label_source']
		p = g.nodes[n]['explicit_source'] # the sources
		# pre = get_prefix(n)
		for l in p:
			if l not in explicitsource_to_entities.keys():
				explicitsource_to_entities[l] = [n]
			else:
				explicitsource_to_entities[l].append(n)
	samples = []
	for p in explicitsource_to_entities.keys():
		if len (explicitsource_to_entities[p]) > 1:
			# print ('sampling from ', labelsource_to_entities[p])
			# for each prefix, we assemble n^2 / 3
			sample_for_this_explicitsource = []
			sample_size = int(len (explicitsource_to_entities[p]) * len (explicitsource_to_entities[p]) / 3)
			# print ('amount expected: ',sample_size)
			# print ('', flush=True)
			while len (sample_for_this_explicitsource) < sample_size:
				# print ('in the loop', flush=True)
				s = random.choice(explicitsource_to_entities[p])
				t = random.choice(explicitsource_to_entities[p])
				# print ('s = ',s)
				# print ('t = ',t)
				if s != t and (s, t) not in sample_for_this_explicitsource and (t, s) not in sample_for_this_explicitsource:
					sample_for_this_explicitsource.append((s,t))
					# print ('\t loaded')

			samples = samples + sample_for_this_explicitsource

	count_same_condition_same_anno = 0
	count_same_condition_diff_anno = 0

	for (s,t) in samples:
		if g.nodes[s]['annotation'] != 'unknown' and  g.nodes[t]['annotation'] != 'unknown':
			if  g.nodes[s]['annotation'] ==  g.nodes[t]['annotation']:
				count_same_condition_same_anno += 1
			elif g.nodes[s]['annotation'] !=  g.nodes[t]['annotation']:
				count_same_condition_diff_anno += 1
	if len(samples) != 0:
		list_same_explicitsource_same_anno.append(count_same_condition_same_anno / len(samples))
		list_same_explicitsource_diff_anno.append(count_same_condition_diff_anno / len(samples))
	print ('number of samples for explicitsources ', len(samples))

	# Step 3: for every (label) source and prefix
	# labelsource_to_entities = {}
	# for n in g.nodes():
	# 	# p = g.nodes[n]['implicit_label_source']
	# 	p = g.nodes[n]['implicit_comment_source'] # the sources
	# 	pre = get_prefix(n)
	# 	for l in p:
	# 		k = l + 'KBwithprefix' + pre
	# 		if k not in labelsource_to_entities.keys():
	# 			labelsource_to_entities[k] = [n]
	# 		else:
	# 			labelsource_to_entities[k].append(n)
	# samples = []
	# for p in labelsource_to_entities.keys():
	# 	if len (labelsource_to_entities[p]) > 1:
	# 		# print ('sampling from ', labelsource_to_entities[p])
	# 		# for each prefix, we assemble n^2 / 3
	# 		sample_for_this_labelsource = []
	# 		sample_size = int(len (labelsource_to_entities[p]) * len (labelsource_to_entities[p]) / 3)
	# 		# print ('amount expected: ',sample_size)
	# 		print ('', flush=True)
	# 		while len (sample_for_this_labelsource) < sample_size:
	# 			# print ('in the loop', flush=True)
	# 			s = random.choice(labelsource_to_entities[p])
	# 			t = random.choice(labelsource_to_entities[p])
	# 			# print ('s = ',s)
	# 			# print ('t = ',t)
	# 			if s != t and (s, t) not in sample_for_this_labelsource and (t, s) not in sample_for_this_labelsource:
	# 				sample_for_this_labelsource.append((s,t))
	# 				# print ('\t loaded')
	#
	# 		samples = samples + sample_for_this_labelsource

	# Step 4: for every (label) source and redirect
	# labelsource_to_entities = {}
	# for n in g.nodes():
	# 	# p = g.nodes[n]['implicit_label_source']
	# 	p = g.nodes[n]['implicit_comment_source'] # the sources
	# 	pre = get_prefix(n)
	# 	for l in p:
	# 		k = l
	# 		if k not in labelsource_to_entities.keys():
	# 			labelsource_to_entities[k] = [n]
	# 		else:
	# 			labelsource_to_entities[k].append(n)
	# samples = []
	# # in_use_redirect_graph
	# for p in labelsource_to_entities.keys():
	# 	if len (labelsource_to_entities[p]) > 1:
	# 		# print ('sampling from ', labelsource_to_entities[p])
	# 		# for each prefix, we assemble n^2 / 3
	# 		sample_for_this_labelsource = []
	# 		sample_size = int(len (labelsource_to_entities[p]) * len (labelsource_to_entities[p]) / 3)
	# 		# print ('amount expected: ',sample_size)
	# 		# print ('', flush=True)
	# 		while len (sample_for_this_labelsource) < sample_size:
	# 			# print ('in the loop', flush=True)
	# 			s = random.choice(labelsource_to_entities[p])
	# 			t = random.choice(labelsource_to_entities[p])
	# 			# print ('s = ',s)
	# 			# print ('t = ',t)
	# 			if s != t and (s, t) not in sample_for_this_labelsource and (t, s) not in sample_for_this_labelsource:
	# 				sample_for_this_labelsource.append((s,t))
	# 				# print ('\t loaded')
	#
	# 		samples = samples + sample_for_this_labelsource
	#
	#
	# # then test if these pairs follow iUNA
	# count_same_condition_same_anno = 0
	# count_same_condition_diff_anno = 0
	#
	#
	# for (s,t) in samples:
	# 	if g.nodes[s]['annotation'] != 'unknown' and  g.nodes[t]['annotation'] != 'unknown':
	# 		if  g.nodes[s]['annotation'] ==  g.nodes[t]['annotation']:
	# 			count_same_condition_same_anno += 1
	# 		elif g.nodes[s]['annotation'] !=  g.nodes[t]['annotation']:
	# 			count_same_condition_diff_anno += 1
	#
	# print ('# total sample = ', len (samples))
	#
	#
	# if len(samples) != 0:
	# 	list_same_condition_same_anno.append(count_same_condition_same_anno / len(samples))
	# 	list_same_condition_diff_anno.append(count_same_condition_diff_anno / len(samples))

	# remove the ones captured by redirect and do the calculation again.
	# count_captured_by_redirect = 0
	# new_samples_except_redirect = []
	# for (n,m) in samples:
	# 	if (n,m) not in in_use_redirect_graph.edges() and (m,n) not in in_use_redirect_graph.edges():
	# 		new_samples_except_redirect.append((n,m))
	# 	else:
	# 		count_captured_by_redirect += 1
	# print ('# caputed by redirect = ', count_captured_by_redirect)
	#
	# count_same_condition_same_anno = 0
	# count_same_condition_diff_anno = 0
	#
	# for (s,t) in new_samples_except_redirect:
	# 	if g.nodes[s]['annotation'] != 'unknown' and  g.nodes[t]['annotation'] != 'unknown':
	# 		if  g.nodes[s]['annotation'] ==  g.nodes[t]['annotation']:
	# 			count_same_condition_same_anno += 1
	# 		elif g.nodes[s]['annotation'] !=  g.nodes[t]['annotation']:
	# 			count_same_condition_diff_anno += 1
	#
	# if len(samples) != 0:
	# 	list_same_condition_same_anno_with_redirect.append(count_same_condition_same_anno / len(samples))
	# 	list_same_condition_diff_anno_with_redirect.append(count_same_condition_diff_anno / len(samples))
	#
	# count_captured_by_ee = 0
	# new_samples_except_ee = []
	# for (n,m) in samples:
	# 	if (n,m) not in in_use_ee_graph.edges() and (m,n) not in in_use_ee_graph.edges():
	# 		new_samples_except_ee.append((n,m))
	# 	else:
	# 		count_captured_by_ee += 1
	# print ('# caputed by ee = ', count_captured_by_ee)
	#
	# count_same_condition_same_anno = 0
	# count_same_condition_diff_anno = 0
	#
	# for (s,t) in new_samples_except_ee:
	# 	if g.nodes[s]['annotation'] != 'unknown' and  g.nodes[t]['annotation'] != 'unknown':
	# 		if  g.nodes[s]['annotation'] ==  g.nodes[t]['annotation']:
	# 			count_same_condition_same_anno += 1
	# 		elif g.nodes[s]['annotation'] !=  g.nodes[t]['annotation']:
	# 			count_same_condition_diff_anno += 1
	#
	# if len(samples) != 0:
	# 	list_same_condition_same_anno_with_ee.append(count_same_condition_same_anno / len(samples))
	# 	list_same_condition_diff_anno_with_ee.append(count_same_condition_diff_anno / len(samples))


# count_total_ee_edges_existing = 0
# count_total_ee_edges_existing_correct = 0
# count_total_ee_edges_existing_error = 0
# count_total_ee_edges_not_existing = 0
# count_total_ee_edges_not_existing_correct = 0
# count_total_ee_edges_not_existing_error = 0

print ('In total, there are ', len(validation_set), 'files for validation\n')
print ('There are in total ', count_total_nodes, ' nodes in the validation graphs')
print ('There are in total ', count_total_edges, ' edges in the validation graphs\n')
print ('There are in total ', count_total_error_edges, ' error edges in the validation graphs\n')
print ('There are in total ', count_total_correct_edges, ' correct edges in the validation graphs\n')
print ('so the error rate is between: ')
print ('\t {:10.2f} %'.format(100*count_total_error_edges/count_total_edges))
print ('\t {:10.2f} %'.format(100*count_total_correct_edges/count_total_edges))
print ('********** Redirection **********')
print ('There are in total ', count_total_redi_nodes, ' nodes in the redirect graphs')
print ('\tAmong them, ', count_total_redi_nodes_new, ' are new nodes not in the original graph')
print ('There are in total ', count_total_redi_edges, ' edges in the redirect graphs\n')
print ('\tAmong them, there are ', count_total_redi_edges_existing, ' about nodes in the original graph')
print ('\t\t correct ', count_total_redi_edges_existing_correct, ' and error ', count_total_redi_edges_existing_error)
print ('\tAmong them, there are ', count_total_redi_edges_not_existing, ' edges in the original graph')
print ('\t\t correct ', count_total_redi_edges_not_existing_correct,
' and error ', count_total_redi_edges_not_existing_error)
# count_total_redi_edges_not_existing = 0
# count_total_redi_edges_not_existing_correct = 0
# count_total_redi_edges_not_existing_error = 0
print ('\t# pairs redirects to the same nodes ', count_total_pairs_redi, ' pairs')
print ('\t\t correct ', count_total_pairs_redi_correct, ' and error ', count_total_pairs_redi_error)
print ('********** Encoding Equivalence **********')
print ('There are in total ', count_total_ee_edges, ' edges in the graph of encoding equivalence')
print ('\tAmong them, there are ', count_total_ee_edges_existing, ' edges in the original graph')
print ('\t\t correct ', count_total_ee_edges_existing_correct, ' error ', count_total_ee_edges_existing_error)
print ('\tAmong them, there are ', count_total_ee_edges_not_existing, ' edges NOT in the original graph')
print ('\t\t correct ', count_total_ee_edges_not_existing_correct, ' error ', count_total_ee_edges_not_existing_error)
print ('\t# pairs encodes to the same nodes ', count_total_pairs_ee, ' pairs')
print ('\t\t correct ', count_total_pairs_ee_correct, ' and error ', count_total_pairs_ee_error)

print ('*********** Sources *************')
print (count_nodes_with_explicit_source, ' has explicit sources: {:10.2f} %'.format(100*count_nodes_with_explicit_source/count_total_nodes))
print (count_nodes_with_implicit_label_source, ' has implicit label-like sources: {:10.2f} %'.format(100*count_nodes_with_implicit_label_source/count_total_nodes))
print (count_nodes_with_implicit_comment_source, ' has implicit comment-like sources: {:10.2f} %'.format(100*count_nodes_with_implicit_comment_source/count_total_nodes))

print ('************ iUNA prefix only ****')
avg_same_prefix_same_anno = np.sum(list_same_prefix_same_anno) / len(validation_set)
avg_same_prefix_diff_anno = np.sum(list_same_prefix_diff_anno) / len(validation_set)
print ('[prefix] avg (same anno) =  {:10.2f} %'.format( avg_same_prefix_same_anno *100))
print ('[prefix] avg (diff anno) =  {:10.2f} %'.format( avg_same_prefix_diff_anno *100))

avg_same_labelsource_same_anno = np.sum(list_same_labelsource_same_anno) / len(validation_set)
avg_same_labelsource_diff_anno = np.sum(list_same_labelsource_diff_anno) / len(validation_set)
print ('[label] avg (same anno)  =  {:10.2f} %'.format( avg_same_labelsource_same_anno *100))
print ('[label] avg (diff anno) =  {:10.2f} %'.format( avg_same_labelsource_diff_anno *100))


avg_same_commentsource_same_anno = np.sum(list_same_commentsource_same_anno) / len(validation_set)
avg_same_commentsource_diff_anno = np.sum(list_same_commentsource_diff_anno) / len(validation_set)
print ('[comment] avg (same anno) =  {:10.2f} %'.format( avg_same_commentsource_same_anno *100))
print ('[comment] avg (diff anno) =  {:10.2f} %'.format( avg_same_commentsource_diff_anno *100))

avg_same_explicitsource_same_anno = np.sum(list_same_explicitsource_same_anno) / len(validation_set)
avg_same_explicitsource_diff_anno = np.sum(list_same_explicitsource_diff_anno) / len(validation_set)
print ('[explicit] avg (same anno) =  {:10.2f} %'.format( avg_same_explicitsource_same_anno *100))
print ('[explicit] avg (diff anno) =  {:10.2f} %'.format( avg_same_explicitsource_diff_anno *100))

# avg_same_condition_same_anno_with_ee = np.sum(list_same_condition_same_anno_with_ee) / len(validation_set)
# avg_same_condition_diff_anno_with_ee = np.sum(list_same_condition_diff_anno_with_ee) / len(validation_set)
# avg_diff_prefix_same_anno /= len(validation_set)
# avg_diff_prefix_diff_anno /= len(validation_set)

# print ('[with redirect] avg =  {:10.2f} %'.format( avg_same_condition_same_anno_with_redirect *100))
# print ('[with redirect] avg =  {:10.2f} %'.format( avg_same_condition_diff_anno_with_redirect *100))
# print ('[with ee] avg =  {:10.2f} %'.format( avg_same_condition_same_anno_with_ee *100))
# print ('[with ee] avg =  {:10.2f} %'.format( avg_same_condition_diff_anno_with_ee *100))
# print ('avg_diff_prefix_same_anno =  {:10.2f} %'.format(avg_diff_prefix_same_anno *100))
# print ('avg_diff_prefix_diff_anno =  {:10.2f} %'.format(avg_diff_prefix_diff_anno *100))
# ratio_following_iUNA = avg_same_prefix_diff_anno + avg_diff_prefix_same_anno
# ratio_contradicts_iUNA =  avg_same_prefix_same_anno + avg_diff_prefix_diff_anno
# print ('-----In summary -------')
# print ('following iUNA = {:10.2f} %'.format( ratio_following_iUNA *100))
# print ('contradicts iUNA = {:10.2f} %'.format( ratio_contradicts_iUNA *100))
