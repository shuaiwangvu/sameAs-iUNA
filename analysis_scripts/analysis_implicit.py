

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

PATH_B = "typeB_Sep20_metalink_only.hdt"
hdt_B = HDTDocument(PATH_B)

PATH_C = "typeC_Sep20_metalink_only.hdt"
hdt_C = HDTDocument(PATH_C)

#
# PATH_B = "label_May.hdt"
# hdt_B = HDTDocument(PATH_B)
#
# PATH_C = "comment_May.hdt"
# hdt_C = HDTDocument(PATH_C)
#

PATH_A = "typeA.hdt"
hdt_A = HDTDocument(PATH_A)


_, cardinality = hdt_A.search_triples("", "", "")
print ('Type A: there are ', cardinality, ' triples')

_, cardinality = hdt_B.search_triples("", "", "")
print ('Type B: there are ', cardinality, ' triples')

_, cardinality = hdt_C.search_triples("", "", "")
print ('Type C: there are ', cardinality, ' triples')

sum_num_entities = 0
sum_has_label_entities = 0
sum_has_comment_entities = 0
annotation = {}
for id in gs:
	# print ('reading ', id)
	filename = './gold/'+str(id) +'.tsv'
	entries = read_file(filename)
	sum_num_entities += len (entries)
	print ('\n***********************\n', id, ' has ', len (entries), ' entities')
	count_B = 0
	count_C = 0
	for row in entries:
		e = row[0]
		a = row[1]
		c = row[2] # comment
		annotation [e] = a

		_, cardinality = hdt_B.search_triples(e, "", "")

		if cardinality >0:
			sum_has_label_entities += 1

		_, cardinality = hdt_C.search_triples(e, "", "")
		if cardinality >0:
			sum_has_comment_entities += 1
print ('total entities in the gold standard ', sum_num_entities)
print ('B ', sum_has_label_entities, '{:10.2f}'.format(100*sum_has_label_entities/sum_num_entities))
print ('C ', sum_has_comment_entities, '{:10.2f}'.format(100*sum_has_comment_entities/sum_num_entities))
