

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
# from extend_metalink import *
import requests
from requests.exceptions import Timeout

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



def read_file (file_name):
	pairs = []
	eq_file = open(file_name, 'r')
	reader = csv.DictReader(eq_file, delimiter='\t',)
	for row in reader:
		s = row["Entity"]
		o = row["Annotation"]
		c = row["Comment"]
		pairs.append([s,o,c])
	return pairs


gs = [4170, 5723,6617,6927,9411,9756,11116,12745,14872,18688,25604,33122,37544,
39036, 42616,96073,97757,99932,236350,240577,337339,395175,712342,1133953,
1140988,4635725,9994282,14514123]

single = []
multiple = []

PATH_DIS = "../sameas_disambiguation_entities_Nov.hdt"
hdt_dis = HDTDocument(PATH_DIS)


sum_num_entities = 0
total_num_unknown = 0

total_annotated_disambiguation = 0
total_typed_disambiguation = 0

total_collect_annotated_disambiguation = set()
total_collect_typed_disambiguation = set()

prefix_ct = Counter()
prefix_ct_unknown = Counter()
annotation = {}
for id in gs:
	# print ('reading ', id)
	filename = str(id) +'.tsv'
	entries = read_file(filename)
	sum_num_entities += len (entries)
	print ('\n***********************\n', id, ' has ', len (entries), ' entities')
	count_unknown = 0
	collect_annotated_disambiguation = set()
	collect_typed_disambiguation = set()
	for row in entries:
		e = row[0]
		a = row[1]
		c = row[2] # comment
		annotation [e] = a

		if a == 'unknown':
			count_unknown += 1
		if c == 'disambiguation':
			collect_annotated_disambiguation.add(e)

		triples, cardinality = hdt_dis.search_triples(e, "", "")
		if cardinality > 0:
			collect_typed_disambiguation.add(e)

	print ('it has ', len(collect_annotated_disambiguation), ' annotated disambiguation entities')
	print ('it has ', len(collect_typed_disambiguation), ' typed disambiguation entities')

	total_num_unknown += count_unknown
	total_collect_annotated_disambiguation = total_collect_annotated_disambiguation.union(collect_annotated_disambiguation)
	total_collect_typed_disambiguation = total_collect_typed_disambiguation.union(collect_typed_disambiguation)

	anno_only = collect_annotated_disambiguation.difference(collect_typed_disambiguation)
	print('\tannotated only ', len(anno_only))

	# for t in anno_only:
	# 	print ('anno only ',t, ' with annotation ', annotation[t])

	type_only = collect_typed_disambiguation.difference(collect_annotated_disambiguation)
	# for t in type_only:
	# 	print (t, ' with annotation ', annotation[t])

	print('\ttyped only ', len(type_only))
	for t in type_only:
		print ('typed only', t, ' with annotation ', annotation[t])

	print ('\tmutual :', len (collect_typed_disambiguation.intersection(collect_annotated_disambiguation)))

print ('there are ', sum_num_entities , ' entities in 28 files')
print ('there are in total ', total_num_unknown, ' unknown entities (by annotation)')

total_annotated_disambiguation = len(total_collect_annotated_disambiguation)
total_typed_disambiguation = len(total_collect_typed_disambiguation)


print ('there are in total ', total_annotated_disambiguation, ' disambiguation entities (by annotation)')
print ('there are in total ', total_typed_disambiguation, ' disambiguation entities (by type)')

all_typed_only =total_collect_typed_disambiguation.difference(total_collect_annotated_disambiguation)
print('in total ',len (all_typed_only), ' typed only')

all_annotated_only =total_collect_annotated_disambiguation.difference(total_collect_typed_disambiguation)
print('in total ',len (all_annotated_only), ' annotated only')

all_mutual =total_collect_annotated_disambiguation.intersection(total_collect_typed_disambiguation)
print ('mutual: ', len (all_mutual))
