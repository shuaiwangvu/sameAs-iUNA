# this file takes the annotated connected components and generate
# the edges of the graph (connected component)
#
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
from rfc3987 import  parse
import urllib.parse
import gzip
from extend_metalink import *
import requests
from requests.exceptions import Timeout
from SameAsEqGraph import get_simp_IRI, get_namespace, get_name



PATH_LOD = "/scratch/wbeek/data/LOD-a-lot/data.hdt"
hdt_lod_a_lot = HDTDocument(PATH_LOD)

PATH_META = "/home/jraad/ssd/data/identity/metalink/metalink-2/metalink-2.hdt"
hdt_metalink = HDTDocument(PATH_META)

PATH_DIS = "sameas_disambiguation_entities.hdt"
hdt_dis = HDTDocument(PATH_DIS)



def load_big_graphs_info (file_name):
	# Index   Size    Entities_without_literals
	nodes_file = open(file_name, 'r')
	reader = csv.DictReader(nodes_file, delimiter='\t',)
	collect_index = {}
	for row in reader:
		s = int(row["Index"])
		n = int(row["Entities_without_literals"])
		collect_index[s] = n
	return collect_index

# load the files
def load_graph (nodes_filename):
	g = nx.Graph()
	nodes_file = open(nodes_filename, 'r')
	reader = csv.DictReader(nodes_file, delimiter='\t',)
	for row in reader:
		s = row["Entity"]
		# a = row["Annotation"]
		# c = row["Comment"]
		g.add_node(s)
	return g


def obtain_edges(g):
	for n in g.nodes():
		(triples, cardi) = hdt_lod_a_lot.search_triples(n, sameas, "")
		for (_,_,o) in triples:
			if o in g.nodes():
				if n != o:
					g.add_edge(n, o)
		(triples, cardi) = hdt_lod_a_lot.search_triples("", sameas, n)
		for (s,_,_) in triples:
			if s in g.nodes():
				if s != n:
					g.add_edge(s, n)
	return g



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



def export_graph_edges (file_name, graph):
	file =  open(file_name, 'w', newline='')
	writer = csv.writer(file, delimiter='\t')
	writer.writerow([ "SUBJECT", "OBJECT", "METALINK_ID"])
	for (l, r) in graph.edges:
		if graph.edges[l, r]['metalink_id'] == None:
			writer.writerow([l, r, 'None'])
		else:
			writer.writerow([l, r, graph.edges[l, r]['metalink_id']])


# sameas_index_to_size_1000.tsv
index_big_graphs = load_big_graphs_info("sameas_index_to_size_1000.tsv")

print ('big graphs are : ', index_big_graphs)
for g_index in list(index_big_graphs.keys())[-1:]:
	print ('index = ', g_index)
	g = load_graph('./big_connected_components/' +str(g_index)+'.tsv')

	bf = g.number_of_nodes()

	g = obtain_edges(g)
	aft = g.number_of_nodes()
	print ('there are ', g.number_of_nodes(), ' edges')
	if aft != bf:
		print ('not the same!')
		print ('before adding edges, there are ', bf)
		print ('before adding edges, there are ', aft)

	# step 2: obtain metalink ID:
	# for (l, r) in g.edges():
	# 	meta_id = find_statement_id(l, r)
	# 	if meta_id != None:
	# 		g[l][r]['metalink_id'] = meta_id
	# 	else:
	# 		g[l][r]['metalink_id'] = None

	# #step 3: export the edges and the metalink ID
	# dir = './big_connected_components/'
	# edges_file_name = dir + str(g_index) + '_edges.tsv'
	# print('the export path for edges = ',edges_file_name )
	# export_graph_edges(edges_file_name, g)

	# step 4
	# test if the grpah is connected
	# print('connected (or not):', nx.is_connected(g))

	# find all the disambiguation_entities
	collect_dis_entities = set()
	for e in g.nodes():
		(_, cardi) = hdt_dis.search_triples(e, "", "")
		if cardi > 0:
			collect_dis_entities.add(e)
	print ('there are in total ', len (collect_dis_entities), ' disambiguation entities')


	print ('after removing the edges, the connnected components are as follows: ')
	# remove them and test the size of connected components
	g.remove_nodes_from(list(collect_dis_entities))
	component_sizes = [len(c) for c in sorted(nx.connected_components(g), key=len, reverse=True)]
	print (component_sizes)
