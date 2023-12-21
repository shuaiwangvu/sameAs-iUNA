
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



# print ('in the validation dataset, there are ', validation_set, ' files (connected components)')

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

size_label_source_to_entities_nUNA = []
size_label_source_to_entities_qUNA = []
size_label_source_to_entities_iUNA = []

restricted_prefix_list = ["http://dblp.rkbexplorer.com/id/",
"http://dbpedia.org/resource/",
"http://rdf.freebase.com/ns/m/",
"http://sws.geonames.org/",
"http://dbtune.org/musicbrainz/resource/",
"http://bio2rdf.org/uniprot:"]

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

	# load explicit source
	path_to_explicit_source = dir + str(id) + '_explicit_source.hdt'
	load_explicit(path_to_explicit_source, g)


	# load implicit label-like source
	path_to_implicit_label_source = dir + str(id) + '_implicit_label_source.hdt'
	load_implicit_label_source(path_to_implicit_label_source, g)


	# load implicit comment-like source
	path_to_implicit_comment_source = dir + str(id) + '_implicit_comment_source.hdt'
	load_implicit_comment_source(path_to_implicit_comment_source, g)

	# redirection
	in_use_redirect_graph = nx.Graph()
	path_to_redi_graph_nodes = dir + str(id) +'_redirect_nodes.tsv'
	path_to_redi_graph_edges = dir + str(id) +'_redirect_edges.hdt'
	print ('the path to redi graph nodes : ', path_to_redi_graph_nodes)
	print ('the path to redi graph edges : ', path_to_redi_graph_edges)
	redi_graph = load_redi_graph(path_to_redi_graph_nodes, path_to_redi_graph_edges)
	print ('loaded the redi graph with ', redi_graph.number_of_nodes(), 'nodes and ', redi_graph.number_of_edges(), ' edges')

	# encoding equivalence
	in_use_ee_graph = nx.Graph()
	# path_to_ee_graph_nodes = dir + str(id) +'_redirect_nodes.tsv'
	path_to_ee_graph_edges = dir + str(id) +'_encoding_equivalent.hdt'
	ee_graph = load_encoding_equivalence( path_to_ee_graph_edges)
	print ('loaded the ee graph with ', ee_graph.number_of_nodes(), 'nodes and ', redi_graph.number_of_edges(), ' edges')


	annotation_to_entities = {}
	for n in g.nodes():
		a = g.nodes[n]['annotation']
		if a in annotation_to_entities.keys():
			annotation_to_entities[a].append(n)
		else:
			annotation_to_entities[a] = [n]
	# nUNA
	for a in annotation_to_entities.keys():
		if a != 'unknown':
			# for each source, how many entities are there?
			# first, collect all the sources
			label_source_to_entities = {}
			for n in annotation_to_entities[a]:
				#find its label-like sources
				for ls in g.nodes[n]['implicit_label_source']:
					if ls in label_source_to_entities.keys():
						label_source_to_entities[ls].append(n)
					else:
						label_source_to_entities[ls] = [n]

			for ls in label_source_to_entities.keys():
				size_label_source_to_entities_nUNA.append(len(label_source_to_entities[ls]))

	# for UNA-quasi
	for a in annotation_to_entities.keys():
		if a != 'unknown':
			# for each source, how many entities are there?
			# first, collect all the sources
			label_source_to_entities = {}
			for n in annotation_to_entities[a]:
				#find its label-like sources
				for ls in g.nodes[n]['implicit_label_source']:
					if ls in label_source_to_entities.keys():
						label_source_to_entities[ls].append(n)
					else:
						label_source_to_entities[ls] = [n]


			for ls in label_source_to_entities.keys():
				entities = label_source_to_entities[ls]
				to_remove = []
				# filter out the entities that are dead nodes
				for e in entities:
					if e in redi_graph.nodes:
						if redi_graph.nodes[e]['remark'] in ['Error', 'NotFound']:
							to_remove.append(e)

				# filter out the entities that are in the redirect graph
				for e in entities:
					redirect_awareness_flag_e = False
					for p in restricted_prefix_list:
						if p in e:
							redirect_awareness_flag_e = True
					if redirect_awareness_flag_e:
						if e in redi_graph.nodes():
							to_remove.append(e)

				for e in to_remove:
					if e in entities:
						entities.remove(e)
				# update it
				label_source_to_entities[ls] = entities

			for ls in label_source_to_entities.keys():
				if len(label_source_to_entities[ls]) != 0:
					size_label_source_to_entities_qUNA.append(len(label_source_to_entities[ls]))

	# for iUNA
	for a in annotation_to_entities.keys():
		if a != 'unknown':
			# for each source, how many entities are there?
			# first, collect all the sources
			label_source_to_entities = {}
			for n in annotation_to_entities[a]:
				#find its label-like sources
				for ls in g.nodes[n]['implicit_label_source']:
					if ls in label_source_to_entities.keys():
						label_source_to_entities[ls].append(n)
					else:
						label_source_to_entities[ls] = [n]


			for ls in label_source_to_entities.keys():
				entities = label_source_to_entities[ls]
				# print ('there are ', len(entities), ' entities')
				to_remove = []
				# filter out the entities that are dead nodes
				for e in entities:
					if e in redi_graph.nodes:
						if redi_graph.nodes[e]['remark'] in ['Error', 'Timeout', 'NotFound', 'RedirectedUntilTimeout', 'RedirectedUntilError', 'RedirectedUntilNotFound']:
							to_remove.append(e)
				# print ('# to remove after filtering dead nodes', len(to_remove))
				# filter out the entities that are in the redirect graph
				for e in entities:
					if e in redi_graph.nodes():
						to_remove.append(e)
				# print ('# to remove after redirect', len(to_remove))

				for e in to_remove:
					if e in entities:
						entities.remove(e)
				# update it
				label_source_to_entities[ls] = entities

			for ls in label_source_to_entities.keys():
				if len(label_source_to_entities[ls]) != 0:
					size_label_source_to_entities_iUNA.append(len(label_source_to_entities[ls]))

	#
	# # for iUNA
	# for a in annotation_to_entities.keys():
	# 	if a != 'unknown':
	# 		# for each source, how many entities are there?
	# 		# first, collect all the sources
	# 		label_source = {}
	# 		for n in annotation_to_entities[a]:
	#
	# 			#find its label-like sources
	#
	# 			for ls in g.nodes[n]['implicit_label_source']:
	# 				if ls in label_source.keys():
	# 					label_source[ls].append(n)
	# 				else:
	# 					label_source[ls] = [n]
	#
	# 		for ls in label_source.keys():
	# 			entities = label_source[ls]
	# 			# for each ls+prefix :
	# 			ls_prefix = {}
	# 			for e in entities:
	# 				p = get_prefix(e)
	# 				if p in e:
	# 					if ls+p not in ls_prefix.keys():
	# 						ls_prefix[ls+p] = [e]
	# 					else:
	# 						ls_prefix[ls+p].append(e)
	#
	# 			# print ('size with prefix = ', len(ls_prefix.keys()))
	# 			for entities in ls_prefix.values():
	# 				# filter out all the redirect
	# 				redi_pairs = []
	# 				# remove all the entities that are dead
	# 				to_remove = []
	# 				# f = redi_graph.nodes[e]['remark']
	# 				for e in entities:
	# 					if e in redi_graph.nodes():
	# 						if redi_graph.nodes[e]['remark'] in ['Error', 'Timeout', 'NotFound', 'RedirectedUntilTimeout', 'RedirectedUntilError', 'RedirectedUntilNotFound']:
	# 							to_remove.append(e)
	#
	# 				for e in entities:
	# 					if e in ee_graph.nodes():
	# 						to_remove.append(e)
	#
	# 				for e in to_remove:
	# 					if e in entities:
	# 						entities.remove(e)
	#
	# 				# print ('remove ', len (to_remove), ' dead nodes')
	# 				for e in entities:
	# 					if e in redi_graph.nodes():
	# 					# collect all the pairs that one directs to the other
	# 						for f in redi_graph.neighbors(e):
	# 							redi_pairs.append((e, f))
	#
	# 				for (e, f) in redi_pairs:
	# 					if e in redi_graph.nodes() and f in redi_graph.nodes():
	# 						if f in entities:
	# 							entities.remove(e)
	#
	# 				# if len(entities) != len(label_source[ls]):
	# 				# 	print ('before there are ', len(label_source[ls]))
	# 				# 	print ('after there are ', len (entities))
	# 				if len(entities) != 0:
	# 					size_label_source_to_entities_iUNA.append(len(entities))


f = plt.figure()
f.set_figwidth(7)
f.set_figheight(1.6)
barWidth = 0.33
ax = plt.subplot(111)

print ('********* nUNA *********')
size_label_source_to_entities_counter = collections.Counter(size_label_source_to_entities_nUNA)
size_label_source_to_entities_counter_sorted = collections.OrderedDict(sorted(size_label_source_to_entities_counter.items()))
print ('nUNA ', size_label_source_to_entities_counter_sorted)
x1 = size_label_source_to_entities_counter_sorted.keys()
y1 = size_label_source_to_entities_counter_sorted.values()

sum_size_iUNA = sum(size_label_source_to_entities_counter.values())
print ('y1 = ', y1)
y1 = [y/sum_size_iUNA for y in y1]
print ('y1 normalised = ', y1)

y1 = [y *100 for y in y1]
ax.bar(x1, y1, color ='green', width=barWidth, label='nUNA', align='center')
# next, see what it is like for each entity each source
b = size_label_source_to_entities_counter[1]
sum_b = 0
for s in size_label_source_to_entities_counter.keys():
	sum_b += size_label_source_to_entities_counter[s]
print ('sum_one ', sum_b)
if sum_b != 0 :
	print ('proportion of one entity each resource',b /sum_b)

print ('********* qUNA *********')
size_label_source_to_entities_counter = collections.Counter(size_label_source_to_entities_qUNA)
size_label_source_to_entities_counter_sorted = collections.OrderedDict(sorted(size_label_source_to_entities_counter.items()))
print ('qUNA', size_label_source_to_entities_counter_sorted)
x2 = size_label_source_to_entities_counter_sorted.keys()
x2 = [x + barWidth for x in x2]
y2 = size_label_source_to_entities_counter_sorted.values()

sum_size_iUNA = sum(size_label_source_to_entities_counter.values())
print ('y2 = ', y2)
y2 = [y/sum_size_iUNA for y in y2]
print ('y2 normalised = ', y2)
y2 = [y *100 for y in y2]
ax.bar(x2, y2, color ='red', width=barWidth, label='qUNA', align='center')
# next, see what it is like for each entity each source
b = size_label_source_to_entities_counter[1]
sum_b = 0
for s in size_label_source_to_entities_counter.keys():
	sum_b += size_label_source_to_entities_counter[s]
print ('sum_one ', sum_b)
if sum_b != 0 :
	print ('proportion of one entity each resource',b /sum_b)



print ('********* iUNA *********')
size_label_source_to_entities_counter = collections.Counter(size_label_source_to_entities_iUNA)
size_label_source_to_entities_counter_sorted = collections.OrderedDict(sorted(size_label_source_to_entities_counter.items()))
print ('iUNA', size_label_source_to_entities_counter_sorted)
x2 = size_label_source_to_entities_counter_sorted.keys()
x2 = [x + barWidth*2 for x in x2]
y2 = size_label_source_to_entities_counter_sorted.values()

sum_size_iUNA = sum(size_label_source_to_entities_counter.values())
print ('y2 = ', y2)
y2 = [y/sum_size_iUNA for y in y2]
print ('y2 normalised = ', y2)

# next, see what it is like for each entity each source
b = size_label_source_to_entities_counter[1]
sum_b = 0
for s in size_label_source_to_entities_counter.keys():
	sum_b += size_label_source_to_entities_counter[s]
print ('sum_one ', sum_b)
if sum_b != 0 :
	print ('proportion of one entity each resource',b /sum_b)
y2 = [y *100 for y in y2]
ax.bar(x2, y2, color ='blue', width=barWidth, label='iUNA', align='center')




ax.autoscale(tight=True)

ax.legend()


plt.yscale('log')
plt.xscale('log')
plt.xlabel("number of entities")
plt.ylabel("proportion (%)")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# plt.title('Frequency of number of entities in label-like sources in equivalence classes')
# plt.grid(True)
plt.savefig('label_source_frequency.png', bbox_inches='tight', dpi = 300)
plt.show()




# log
# plt.subplot(313)
# plt.plot(x, y)
