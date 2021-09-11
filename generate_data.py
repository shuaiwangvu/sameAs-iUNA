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
from urllib.parse import urlparse
import gzip
from extend_metalink import *

PATH_META = "/home/jraad/ssd/data/identity/metalink/metalink-2/metalink-2.hdt"
hdt_metalink = HDTDocument(PATH_META)


hdt_source = HDTDocument("typeA.hdt")
hdt_label = HDTDocument("label_May.hdt")
hdt_comment = HDTDocument("comment_May.hdt")


PATH_LOD = "/scratch/wbeek/data/LOD-a-lot/data.hdt"
hdt_lod_a_lot = HDTDocument(PATH_LOD)


gs = [4170, 5723,6617,6927,9411,9756,11116,12745,14872,18688,25604,33122,37544,
39036, 42616,96073,97757,99932,236350,240577,337339,395175,712342,1133953,
1140988,4635725,9994282,14514123]

single = [9411, 9756, 18688, 25604, 96073, 97757, 99932, 337339,
712342, 1133953, 1140988, 9994282]

related = [4170, 5723, 11116]

multiple = [6617, 6927, 12745, 14872, 33122, 37544, 39036, 42616, 236350, 240577,
395175, 4635725, 14514123]


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

def read_file (file_name):
	pairs = []
	eq_file = open(file_name, 'r')
	reader = csv.DictReader(eq_file, delimiter='\t',)
	for row in reader:
		s = row["Entity"]
		o = row["Annotation"]
		pairs.append((s,o))
	return pairs

def obtain_edges(g):
	for n in g.nodes():
		(triples, cardi) = hdt_lod_a_lot.search_triples(n, sameas, "")
		for (_,_,o) in triples:
			if o in g.nodes():
				g.add_edge(n, o)
		(triples, cardi) = hdt_lod_a_lot.search_triples("", sameas, n)
		for (s,_,_) in triples:
			if s in g.nodes():
				g.add_edge(s, n)
	return g


def export_graph_edges (file_name, graph):
	file =  open(file_name, 'w', newline='')
	writer = csv.writer(file)
	writer.writerow([ "SUBJECT", "OBJECT", "METALINK_ID"])
	for (l, r) in graph.edges:
		if graph.edges[l, r]['metalink_id'] == None:
			writer.writerow([l, r, 'None'])
		else:
			writer.writerow([l, r, graph.edges[l, r]['metalink_id']])

# type A: explicit sources
def export_explicit_source (file_name, graph):
	count_A = 0
	with open(file_name, 'w') as output:
		for n in graph.nodes:
			triples, cardinality = hdt_source.search_triples(n, "", "")
			for (_, predicate, file) in triples:
				line = '<' + n + '> '
				line += '<' + predicate + '> '
				if str(file)[0] == '"':
					line += '' + file + ' .\n'
				else:
					line += '<' + file + '>. \n'

				output.write(str(line))
				count_A += 1
	print ('count A ', count_A)

# type B: implicit label sources
def export_implicit_label_source (file_name, graph):
	# type B
	count_B = 0
	with open( file_name, 'w') as output:
		for n in graph.nodes:
			triples, cardinality = hdt_label.search_triples(n, "", "")
			for (_, predicate, file) in triples:
				line = '<' + n + '> '
				line += '<' + predicate + '> '
				line += '<' + file + '>. \n'
				output.write(str(line))
				count_B += 1
	print ('count B ', count_B)


# type C: implicit comment sources
def export_implicit_comment_source (file_name, graph):
	count_C = 0
	with open( file_name, 'w') as output:
		for n in graph.nodes():
			triples, cardinality = hdt_comment.search_triples(n, "", "")
			for (_, predicate, file) in triples:
				line = '<' + n + '> '
				line += '<' + predicate + '> '
				line += '<' + file + '>. \n'
				output.write(str(line))
				count_C += 1
	print ('count C ', count_C)


sum_num_entities = 0
total_num_unknown = 0
prefix_ct = Counter()
prefix_ct_unknown = Counter()
for id in gs:
	print ('\n***************\n')
	dir = './gold/'
	filename = dir + str(id) +'.tsv'
	pairs = read_file(filename)

	print ('Single: reading ', id)

	sum_num_entities += len (pairs)
	g = nx.DiGraph()

	for (e, a) in pairs:
		g.add_node(e, annotation = a)
	# step 1: obtain the whole graph
	obtain_edges(g)
	print ('There are ', g.number_of_nodes(), ' nodes')
	print ('There are ', g.number_of_edges(), ' edges')

	# step 2: obtain metalink ID:
	for (l, r) in g.edges():
		meta_id = find_statement_id(l, r)
		if meta_id != None:
			g[l][r]['metalink_id'] = meta_id
		else:
			g[l][r]['metalink_id'] = None

	#step 3: export the edges and the metalink ID
	edges_file_name = dir + str(id) +'_edges.tsv'
	# export_graph_edges(edges_file_name, g)

	# step 4: export the sources: Type A B C
	explicit_file_path = dir + str(id) + '_explicit_source.nt'
	export_explicit_source(explicit_file_path, g)

	label_file_path = dir + str(id) + '_implicit_label_source.nt'
	export_implicit_label_source(label_file_path, g)

	comment_file_path = dir +  str(id) + '_implicit_comment_source.nt'
	export_implicit_comment_source(comment_file_path, g)
