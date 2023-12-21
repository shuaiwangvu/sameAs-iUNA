# this script converts the sources to its weights.

import networkx as nx
from SameAsEqGraph import *
from pyvis.network import Network
import community
import collections
import matplotlib.pyplot as plt
from random import randint
import pandas as pd
import requests
from collections import Counter
from rfc3987 import  parse
import urllib.parse
from hdt import HDTDocument, IdentifierPosition
# from z3 import *
from rdflib.namespace import XSD
import csv
from extend_metalink import *
import time

UNKNOWN = 0
REMOVE = 1
KEEP = 2

hdt_source = None
hdt_label = None
hdt_comment = None


PATH_SAMEAS_SOURCE = "./sameas_laundromat_metalink_Oct18.hdt"
hdt_source = HDTDocument(PATH_SAMEAS_SOURCE)


def find_weight (id):
	cardinality = 0
	try:
		triples, cardinality = hdt_source.search_triples(id, my_exist_in_file, "")
	except:
		pass

	return cardinality



ct = Counter ()
count_statements_processed = 0
log_writer = open('sameas_laundromat_metalink_sum_weight_Oct.log', 'w')
distribution_writer = open('sameas_laundromat_metalink_sum_weight_Oct_distribution.log', 'w')
start = time.time()
with open('sameas_laundromat_metalink_sum_weight_Oct.nt', 'w') as writer:
	visited = set()
	triples, cardinality = hdt_source.search_triples("", my_exist_in_file, "")
	for (id, _, _) in triples:
		try:
			if id not in visited:
				weight = find_weight(id)
				ct [weight] += 1
				visited.add(id)
				line = '<' + id + '> '
				line += '<' + my_has_num_occurences_in_files + '> '
				line += '"'+str(weight)+'"^^<' + str(XSD.integer) + '> . \n'
				writer.write(str(line))
				count_statements_processed += 1

			else:
				continue
		except Exception as inst:
			print ('An exception happened!')
		else:
			pass



		if count_statements_processed %10000 == 0:

			log_writer.write('total statements processed: ' + str(count_statements_processed) +'\n')
			end = time.time()
			hours, rem = divmod(end-start, 3600)
			minutes, seconds = divmod(rem, 60)
			time_formated = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)
			log_writer.write ('time taken: ' + time_formated)
			log_writer.flush()
			# break



for c in ct:
	line = str(c) + ' : ' + str(ct[c]) +',\n'
	distribution_writer.write(str(line))
