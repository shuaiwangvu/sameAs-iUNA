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

PATH_META = "/home/jraad/ssd/data/identity/metalink/metalink-2/metalink-2.hdt"
hdt_metalink = HDTDocument(PATH_META)

hdt_source = HDTDocument("typeA.hdt")
hdt_label = HDTDocument("label_May.hdt")
hdt_comment = HDTDocument("comment_May.hdt")

sameas = "http://www.w3.org/2002/07/owl#sameAs"
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

# task one, generate the edges

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


sum_num_entities = 0
total_num_unknown = 0
prefix_ct = Counter()
prefix_ct_unknown = Counter()
for id in gs:
	print ('\n***************\n')
	dir = './gold/'
	filename = dir + str(id) +'.tsv'
	pairs = read_file(filename)
	if id in single:
		print ('Single: reading ', id)
	elif id in related:
		print ('Related: reading ', id)
	elif id in multiple:
		print ('Multiple: reading ', id)
	else:
		print ('ERROR!')

	sum_num_entities += len (pairs)
	g = nx.DiGraph()
	for (e, a) in pairs:
		g.add_node(e, annotation = a)
	# obtain the whole graph
	obtain_edges(g)
	print ('There are ',len(g.edges()), ' edges')

if gs == (single + related + multiple):
	print ('Correct')
else:
	for g in single:
		if g not in gs:
			print ('single: ', g)
	for g in related:
		if g not in gs:
			print ('single: ', g)
	for g in multiple:
		if g not in gs:
			print ('single: ', g)
