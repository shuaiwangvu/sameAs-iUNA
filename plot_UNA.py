
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
from SameAsEqGraph import get_prefix
import collections
from SameAsEqGraph import *


# there are in total 28 entities. 14 each
validate_single = [96073, 712342, 9994282, 18688, 1140988, 25604]
validate_multiple = [6617, 4170, 42616, 39036, 33122, 6927, 11116, 12745]
validation_set = validate_single + validate_multiple

evaluation_single = [9411, 9756, 97757, 99932, 337339, 1133953]
evaluation_multiple = [5723, 14872, 37544, 236350, 240577, 395175, 4635725, 14514123]
evaluation_set = evaluation_single + evaluation_multiple


gs = validation_set + evaluation_set


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
# count_total_error_edges = 0
count_total_unknown_nodes = 0
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

count_error_with_same_label_sources = 0
count_error_with_diff_label_sources = 0
count_error_with_same_comment_sources = 0
count_error_with_diff_comment_sources = 0
count_error_with_same_prefix = 0
count_error_with_same_authority = 0

count_error_about_disambiguation = 0

count_edges_between_dbpedia_multilingual = 0
count_error_between_dbpedia_multilingual = 0

count_edges_between_dbpedia_multilingual_and_dbpedia = 0
count_error_between_dbpedia_multilingual_and_dbpedia = 0

size_of_clusters = []
size_label_source_to_entities = []
size_comment_source_to_entities = []

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
	# print ('there are in total ', count_error_edges, ' errorous edges ')
	# count_total_error_edges += count_error_edges



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

	authority_to_entities = {}
	for n in g.nodes():
		au = get_authority(n)
		if au in authority_to_entities.keys():
			authority_to_entities[au].append(n)
		else:
			authority_to_entities[au] = [n]

	for au in authority_to_entities.keys():
		for s in authority_to_entities[au]:
			for t in authority_to_entities[au]:
				if s != t and get_prefix(s) != get_prefix(t):
					pass
					# print (s, ' and ', t, ' have different prefix despite same authority')

	#
	#
	# # if the edges of disambiguates has higher rate of error than normal
	#
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

	for (s,t) in g.edges():

		if get_authority(s) != 'dbpedia.org' and get_authority(t) != 'dbpedia.org' and 'dbpedia.org' in s and 'dbpedia.org' in t:
			count_edges_between_dbpedia_multilingual += 1
		elif 'dbpedia.org' in s and 'dbpedia.org' in t:
			if get_authority(s) == 'dbpedia.org' and get_authority(t) != 'dbpedia.org':
				count_edges_between_dbpedia_multilingual_and_dbpedia += 1
			elif get_authority(s) != 'dbpedia.org' and get_authority(t) == 'dbpedia.org':
				count_edges_between_dbpedia_multilingual_and_dbpedia += 1

		if (g.nodes[s]['annotation'] != 'unknown'
			and g.nodes[t]['annotation'] != 'unknown'):

			if g.nodes[s]['annotation'] != g.nodes[t]['annotation']:

				# how many of them are about nodes of different label sources
				if len(set(g.nodes[s]['implicit_label_source']).intersection(set(g.nodes[t]['implicit_label_source']))) > 0:
					count_error_with_same_label_sources += 1
				elif len(set(g.nodes[s]['implicit_label_source']).intersection(set(g.nodes[t]['implicit_label_source']))) == 0 and len(set(g.nodes[s]['implicit_label_source']))> 0 and len(set(g.nodes[t]['implicit_label_source']))> 0:
					count_error_with_diff_label_sources += 1

				if len(set(g.nodes[s]['implicit_comment_source']).intersection(set(g.nodes[t]['implicit_comment_source']))) > 0:
					count_error_with_same_comment_sources += 1
				elif len(set(g.nodes[s]['implicit_comment_source']).intersection(set(g.nodes[t]['implicit_comment_source']))) == 0 and len(set(g.nodes[s]['implicit_comment_source']))> 0 and len(set(g.nodes[t]['implicit_comment_source']))> 0:
					count_error_with_diff_comment_sources += 1

				if get_prefix(s) == get_prefix(t):
					count_error_with_same_prefix += 1

				if get_authority(s) == get_authority(t) and get_authority(s) != None and  get_authority(t) != None :
					count_error_with_same_authority += 1

				if g.nodes[s]['comment'] == 'disambiguation' or g.nodes[t]['comment'] == 'disambiguation':
					count_error_about_disambiguation += 1

				if get_authority(s) != 'dbpedia.org' and get_authority(t) != 'dbpedia.org' and 'dbpedia.org' in s and 'dbpedia.org' in t:
					count_error_between_dbpedia_multilingual += 1

				if get_authority(s) == 'dbpedia.org' and get_authority(t) != 'dbpedia.org':
					count_error_between_dbpedia_multilingual_and_dbpedia += 1
				elif get_authority(s) != 'dbpedia.org' and get_authority(t) == 'dbpedia.org':
					count_error_between_dbpedia_multilingual_and_dbpedia += 1


				# if (g.nodes[s]['comment'] != 'disambiguation' and g.nodes[t]['comment'] != 'disambiguation'  ):
				# 	print (s, ' with annoation ', g.nodes[s]['annotation'])
				# 	print ('authority ', get_authority(s))
				# 	print (t, ' with annotation ', g.nodes[t]['annotation'])
				# 	print ('authority ', get_authority(t))
				# 	print ('\n')


	annotation_to_entities = {}
	for n in g.nodes():
		a = g.nodes[n]['annotation']
		if a in annotation_to_entities.keys():
			annotation_to_entities[a].append(n)
		else:
			annotation_to_entities[a] = [n]

	# size_of_clusters = []
	# size_label_source_to_entities = []
	# size_comment_source_to_entities = []
	for a in annotation_to_entities.keys():
		if a != 'unknown':
			# for each source, how many entities are there?
			size_of_clusters.append(len(annotation_to_entities[a]))
			# first, collect all the sources
			label_source = {}
			comment_source = {}
			for n in annotation_to_entities[a]:

				#find its label-like sources

				for ls in g.nodes[n]['implicit_label_source']:
					if ls in label_source.keys():
						label_source[ls].append(n)
					else:
						label_source[ls] = [n]
				#find its comment-like sources
				for cs in g.nodes[n]['implicit_comment_source']:
					if cs in comment_source.keys():
						comment_source[cs].append(n)
					else:
						comment_source[cs] = [n]
			for ls in label_source.keys():
				size_label_source_to_entities.append(len(label_source[ls]))

			for cs in comment_source.keys():
				size_comment_source_to_entities.append(len(comment_source[cs]))


		# for each source, we file how many entities there are in annotation_to_entities[a]

	# count_error_with_same_prefix
	# # How many erronous edges remains after removing unknown and disambiguation nodes
	# collection_disambiguation = set()
	# collection_unknown = set()
	# for n in g.nodes():
	# 	if g.nodes[n]['annotation'] == 'unknown':
	# 		collection_unknown.add(n)
	# 	elif g.nodes[n]['comment'] == 'disambiguation':
	# 		collection_disambiguation.add(n)
	# g.remove_nodes_from(list(collection_disambiguation))
	# g.remove_nodes_from(list(collection_unknown))
	# print ('after removing', len (collection_disambiguation), ' disambiguation entities and ')
	# print ('and ', len (collection_unknown), ' unknown entities')
	# count_total_unknown_nodes += len (collection_unknown)
	# print ('there are ', g.number_of_nodes(), ' nodes')
	# print ('there are ', g.number_of_edges(), ' edges')
	#
	# count_remain_error_edges = 0
	# for (s, t) in g.edges():
	# 	if (g.nodes[s]['annotation'] != 'unknown'
	# 		and g.nodes[t]['annotation'] != 'unknown'
	# 		and g.nodes[s]['annotation'] != g.nodes[t]['annotation']):
	# 		count_remain_error_edges += 1
	#
	# print ('num of count_remain_error_edges = ', count_remain_error_edges)
	# total_count_remain_error_edges += count_remain_error_edges


print ('**Summary**')
# print ('In total, there are ', len(validation_set), 'files for validation\n')
print ('There are in total ', count_total_nodes, ' nodes in the  graphs')
print ('There are in total ', count_total_edges, ' edges in the  graphs\n')
print ('There are ', count_total_unknown_nodes, ' nodes about unknown\n')
print ('There are in total ', count_error_edges, ' error edges in the  graphs\n')
print ('\t {:10.2f} %'.format(100*count_error_edges/count_total_edges))
# print ('There are in total ', count_total_redi_nodes, ' nodes in the redirect graphs')
# print ('There are in total ', count_total_redi_edges, ' edges in the redirect graphs\n')
# print ('There are in total ', count_total_ee_edges, ' edges in the graph of encoding equivalence')

print (count_nodes_with_explicit_source, ' has explicit sources: {:10.2f} %'.format(100*count_nodes_with_explicit_source/count_total_nodes))
print (count_nodes_with_implicit_label_source, ' has implicit label-like sources: {:10.2f} %'.format(100*count_nodes_with_implicit_label_source/count_total_nodes))
print (count_nodes_with_implicit_comment_source, ' has implicit comment-like sources: {:10.2f} %'.format(100*count_nodes_with_implicit_comment_source/count_total_nodes))

prefix_ct_error = Counter()
for (s, t) in collect_error_edges:
	# find where they are from
	prefix_ct_error[get_prefix(s)] += 1
	prefix_ct_error[get_prefix(t)] += 1

prefix_ct = Counter()
for (s, t) in collect_edges:
	# find where they are from
	prefix_ct[get_prefix(s)] += 1
	prefix_ct[get_prefix(t)] += 1

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
	ps = get_prefix(s)
	pt = get_prefix(t)
	if ps > pt :
		prefix_pair_ct[(pt, ps)] += 1
	else:
		prefix_pair_ct[(ps, pt)] += 1

prefix_pair_ct_error = Counter()
for (s, t) in collect_error_edges:
	ps = get_prefix(s)
	pt = get_prefix(t)
	if ps > pt :
		prefix_pair_ct_error[(pt, ps)] += 1
	else:
		prefix_pair_ct_error[(ps, pt)] += 1

prefix_error_rate = {}
for pair in prefix_pair_ct.keys():
	pct = prefix_pair_ct_error[pair] / prefix_pair_ct[pair]
	prefix_error_rate[pair] = pct


prefix_error_rate = {k: v for k, v in sorted(prefix_error_rate.items(), key=lambda item: item[1])}

# for pair in prefix_error_rate.keys():
# 	if prefix_error_rate[pair] >= 0.10: # prefix_pair_ct[pair] >= 5 and
		# print (pair,',')
		# print ('pair = ', pair)
		# print ('prefix pair count ', prefix_pair_ct[pair])
		# print ('prefix pair error count ', prefix_pair_ct_error[pair])
		# print ('error rate = ', prefix_error_rate[pair])
		# print ('\n')


print ('overal total edges: ',count_total_edges, ' = ', len (collect_edges))
print ('overal error edges:', count_error_edges)
print ('overal correct edges:', count_correct_edges)
print ('overall error rate ', count_error_edges / len (collect_edges))
print ('overall correct rate ', count_correct_edges / len (collect_edges))


print ('*'*20)
print ('how many error edges? ', count_error_edges)
print ('---')
print ('How many pct of errors involves disambiguation nodes? ')
print ('about disambiguation ', count_error_about_disambiguation)
print ('in pct among all errors: ', count_edges_involving_disambiguation_error/count_error_edges)
print ('there are total number of edges involving dismbiguation', count_edges_involving_disambiguation)
print ('in pct error over all such edges: ', count_error_about_disambiguation/count_edges_involving_disambiguation)
print ('-> disambiguation error', count_edges_involving_disambiguation_error)
print ('-> disambiguation correct', count_edges_involving_disambiguation_correct)
print ('-> when disambiguation nodes are involved, error rate: ', count_edges_involving_disambiguation_error/ count_edges_involving_disambiguation)
print ('-> when disambiguation nodes are involved, correct rate: ', count_edges_involving_disambiguation_correct/ count_edges_involving_disambiguation)

print ('---')
print ('How many pct of errors is about DBpedia multilingual edges? ')
print ('about errors between multilingual ', count_error_between_dbpedia_multilingual)
print ('in pct: ', count_error_between_dbpedia_multilingual/count_error_edges)
print ('there are edges between dbpedial multilingual', count_edges_between_dbpedia_multilingual)
print ('in pct for all such edges: ', count_error_between_dbpedia_multilingual/count_edges_between_dbpedia_multilingual)
print ('- now between multilingual and dbpedia.org -')
print ('there are ', count_error_between_dbpedia_multilingual)
print ('in pct: ', count_error_between_dbpedia_multilingual/count_error_edges)
print ('there are in total edges between dbpedia multilingual and dbpedia.org', count_edges_between_dbpedia_multilingual_and_dbpedia)
print ('in pct: ', count_error_between_dbpedia_multilingual_and_dbpedia/count_error_edges)
print ('about errors between multilingual and dbpedia.org ', count_error_between_dbpedia_multilingual_and_dbpedia)

# count_error_between_dbpedia_multilingual

print ('----')
print ('How many are due to errors of the same label-source')
print ('there are', count_error_with_same_label_sources)
print ('in pct: ', count_error_with_same_label_sources/count_error_edges)
print ('How many are due to errors of the diff label-source')
print ('there are', count_error_with_diff_label_sources)
print ('in pct: ', count_error_with_diff_label_sources/count_error_edges)

print ('---')
print ('How many are due to errors of the same comment-source')
print ('there are', count_error_with_same_comment_sources)
print ('in pct: ', count_error_with_same_comment_sources/count_error_edges)
print ('How many are due to errors of the diff comment-source')
print ('there are', count_error_with_diff_comment_sources)
print ('in pct: ', count_error_with_diff_comment_sources/count_error_edges)

print ('---')
print ('How many are due to errors of the same prefix')
print ('there are', count_error_with_same_prefix)
print ('in pct: ', count_error_with_same_prefix/count_error_edges)

print ('---')
print ('How many are due to errors of the same authority')
print ('there are', count_error_with_same_authority)
print ('in pct: ', count_error_with_same_authority/count_error_edges)

# print ('the pct of errors after removing disambiguation (and unknown) nodes')
# print (total_count_remain_error_edges/count_error_edges)


#
#
# http://wa.dbpedia.org/resource/
# count edges:  649
# count error edges:  220
#  gives pct error rate  0.3389830508474576


#
# http://www4.wiwiss.fu-berlin.de/flickrwrappr/photos/
# count edges:  76
# count error edges:  11
#  gives pct error rate  0.14473684210526316

# plot the frequency of the following:

# size_of_clusters = []
# size_label_source_to_entities = []
# size_comment_source_to_entities = []


size_of_clusters_counter = collections.Counter(size_of_clusters)
# size_of_clusters_counter = sorted(size_of_clusters_counter)
size_of_clusters_counter_sorted = collections.OrderedDict(sorted(size_of_clusters_counter.items()))
# print (size_of_clusters_counter_sorted)
x = size_of_clusters_counter_sorted.keys()
y = size_of_clusters_counter_sorted.values()

# log
# plt.subplot(311)
# plt.plot(x, y)
# plt.set_size_inches(8.5, 7.5)
# f = plt.figure()
# f.set_figwidth(4)
# f.set_figheight(1.5)
#
# ax = plt.subplot(111)
# #
# ax.bar(x, y, color ='black')
# ax.set_yscale('log')
# #
# print('x = ', x)
# print ('y = ',y)
# plt.xlabel("The size of equivalent classes")
# plt.ylabel("Frequency")
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# # plt.title('Frequency of size of equivalent classes in connected components')
# plt.savefig('size_equi.png', bbox_inches='tight', dpi = 300)
# # plt.show()

size_label_source_to_entities_counter = collections.Counter(size_label_source_to_entities)
# size_of_clusters_counter = sorted(size_of_clusters_counter)
size_label_source_to_entities_counter_sorted = collections.OrderedDict(sorted(size_label_source_to_entities_counter.items()))
# print (size_label_source_to_entities_counter_sorted)
x = size_label_source_to_entities_counter_sorted.keys()
y = size_label_source_to_entities_counter_sorted.values()
b = size_label_source_to_entities_counter[1]
print ('one label-like source ', b)
sum_b = 0
for s in size_label_source_to_entities_counter.keys():
	sum_b += size_label_source_to_entities_counter[s]
print ('sum_b ', sum_b)
print (b /sum_b)
barWidth = 0.49
#
# log
f = plt.figure()
f.set_figwidth(4)
f.set_figheight(2.5)
 # plt.subplot(111)
ax = plt.subplot(111)
# plt.plot(x, y)
ax.bar(x, y, color ='blue', width=barWidth, label = 'label-like source')

ax.set_yscale('log')
# plt.xlabel("Number of entities in each implicit label-like source")
# plt.ylabel("Frequency")
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# plt.title('Frequency of number of entities in label-like sources in equivalence classes')
# plt.grid(True)
# plt.savefig('label_source_frequency.png', bbox_inches='tight', dpi = 300)

size_comment_source_to_entities_counter = collections.Counter(size_comment_source_to_entities)
# size_of_clusters_counter = sorted(size_of_clusters_counter)
size_comment_source_to_entities_counter_sorted = collections.OrderedDict(sorted(size_comment_source_to_entities_counter.items()))
print (size_comment_source_to_entities_counter_sorted)
x = size_comment_source_to_entities_counter_sorted.keys()
y = size_comment_source_to_entities_counter_sorted.values()
b = size_comment_source_to_entities_counter[1]
print ('one comment-like source ', b)
sum_b = 0
for s in size_comment_source_to_entities_counter.keys():
	sum_b += size_comment_source_to_entities_counter[s]
print ('sum_b ', sum_b)
print (b /sum_b)

# log
# plt.subplot(313)
# plt.plot(x, y)
x = [e + barWidth*1 for e in x]
ax.bar(x, y, color ='red', width=barWidth, label = 'comment-like source')

# ax.yscale('log')
plt.xlabel("Number of entities")
plt.ylabel("Frequency")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend()
# plt.title('Frequency of number of entities in implicit sources in equivalence classes')
# plt.grid(True)
# plt.savefig('comment_source_frequency.png', bbox_inches='tight', dpi = 300)
plt.savefig('both_source_frequency.png', bbox_inches='tight', dpi = 300)

# plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)

# plt.show()
# bin_size = max(size_label_source_to_entities)
# print ('bin size = ', bin_size)
# plt.hist(size_label_source_to_entities, bins= bin_size)
# plt.gca().set(title='Frequency Histogram', ylabel='Frequency');
# plt.show()
