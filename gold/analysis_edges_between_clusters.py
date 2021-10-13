

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


rdf_type = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

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

PATH_DIS = "../sameas_disambiguation_entities.hdt"
hdt_dis = HDTDocument(PATH_DIS)

PATH_LOD = "/scratch/wbeek/data/LOD-a-lot/data.hdt"
hdt_lod_a_lot = HDTDocument(PATH_LOD)

sum_num_entities = 0
total_num_unknown = 0

total_annotated_disambiguation = 0
total_typed_disambiguation = 0

total_collect_annotated_disambiguation = set()
total_collect_annotated_disambiguation_or_unknown = set()
total_collect_typed_disambiguation = set()
total_collect_entities = set()

use_template = "http://dbpedia.org/property/wikiPageUsesTemplate"
dbr_disambig = "http://dbpedia.org/resource/Template:Disambig" #105043
dbr_disambiguation = "http://dbpedia.org/resource/Template:Disambiguation" # 54271



pairs_dis = [
("http://it.dbpedia.org/property/wikiPageUsesTemplate","http://it.dbpedia.org/resource/Template:Disambigua"),
("http://es.dbpedia.org/property/wikiPageUsesTemplate","http://es.dbpedia.org/resource/Plantilla:Desambiguación"),
("http://ru.dbpedia.org/property/wikiPageUsesTemplate","http://ru.dbpedia.org/resource/Шаблон:Неоднозначность"),
("http://sr.dbpedia.org/property/wikiPageUsesTemplate","http://sr.dbpedia.org/resource/Шаблон:Вишезначна_одредница"),
("http://sco.dbpedia.org/property/wikiPageUsesTemplate","http://sco.dbpedia.org/resource/Template:Disambig"),
("http://fa.dbpedia.org/property/wikiPageUsesTemplate","http://fa.dbpedia.org/resource/الگو:ابهام_زدایی"),
("http://fr.dbpedia.org/property/wikiPageUsesTemplate","http://fr.dbpedia.org/resource/Modèle:Homonymie"),
("http://de.dbpedia.org/property/wikiPageUsesTemplate","http://de.dbpedia.org/resource/Vorlage:Begriffsklärung"),
("http://no.dbpedia.org/property/wikiPageUsesTemplate","http://no.dbpedia.org/resource/Mal:Pekerside"),
("http://lt.dbpedia.org/property/wikiPageUsesTemplate","http://lt.dbpedia.org/resource/Šablonas:Disambig"),
("http://sv.dbpedia.org/property/wikiPageUsesTemplate","http://sv.dbpedia.org/resource/Mall:Gren"),
("http://ja.dbpedia.org/property/wikiPageUsesTemplate","http://ja.dbpedia.org/resource/Template:Aimai"),
("http://pt.dbpedia.org/property/wikiPageUsesTemplate","http://pt.dbpedia.org/resource/Predefinição:Disambig"),
("http://pl.dbpedia.org/property/wikiPageUsesTemplate","http://pl.dbpedia.org/resource/Szablon:Ujednoznacznienie"),
("http://hy.dbpedia.org/property/wikiPageUsesTemplate","http://hy.dbpedia.org/resource/Կաղապար:Բի"),
("http://uk.dbpedia.org/property/wikiPageUsesTemplate","http://uk.dbpedia.org/resource/Шаблон:DisambigG"),
("http://ca.dbpedia.org/property/wikiPageUsesTemplate","http://ca.dbpedia.org/resource/Plantilla:Desambiguació")
]


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
	annotation_to_nodes = {}
	for row in entries:
		e = row[0]
		a = row[1]
		c = row[2] # comment
		annotation [e] = a
		total_collect_entities.add(e)
		if a not in annotation_to_nodes.keys():
			annotation_to_nodes[a] = [e]
		else:
			annotation_to_nodes[a].append(e)

	# find out how many annotations > 3 entities
	for k in annotation_to_nodes.keys():
		print (k ,' has nodes ', len(annotation_to_nodes[k]))
