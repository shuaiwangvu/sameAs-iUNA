# load the local test.nt file and convert it to a format ready to be handled.
# when a string is there, conver it to ""^^<http://www.w3.org/2001/XMLSchema#string>

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


PATH_META = "/home/jraad/ssd/data/identity/metalink/metalink-2/metalink-2.hdt"
hdt_metalink = HDTDocument(PATH_META)

file1 = open('typeB_Sep16.nt', 'r')
file2 = open('typeB_Sep20_metalink_only.nt', 'w')

rdf_subject = "http://www.w3.org/1999/02/22-rdf-syntax-ns#subject"
rdf_object = "http://www.w3.org/1999/02/22-rdf-syntax-ns#object"
rdf_predicate = "http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate"
count  = 0

file_no_metalink  = open( "convert_typeB_progress.tsv", 'w')
no_metalink_writer = csv.writer(file_no_metalink, delimiter='\t')
no_metalink_writer.writerow(['Count', 'Time'])

start = time.time()

while True:
	# count += 1

	# if count > 300:
	# 	break
	# Get next line from file
	l = file1.readline()
	if not l:
		break
	# else:
	# 	print ('reading line: ', l)

	splited = l.split(' ')
	s = splited[0][1:-1]
	# print (s)
	_, cardinality = hdt_metalink.search_triples("", rdf_subject, s)
	_, cardinality2 = hdt_metalink.search_triples("", rdf_object, s)

	if cardinality2 >0 or cardinality >0 :
		file2.write(l)
		count += 1
		# print (count)
		if count %100000 == 0:
			end = time.time()
			hours, rem = divmod(end-start, 3600)
			minutes, seconds = divmod(rem, 60)

			time_formated = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)
			no_metalink_writer.writerow([count, time_formated])
			file_no_metalink.flush()

file1.close()
file2.close()

#
# for l in file1.readlines():
# 	print (count)
# 	if count > 100:
# 		break
# 	splited = l.split(' ')
# 	s = splited[0][1:-1]
#
# 	_, cardinality = hdt_metalink.search_triples("", rdf_subject, s)
# 	_, cardinality2 = hdt_metalink.search_triples("", rdf_object, s)
#
# 	if cardinality2 >0 or cardinality >0 :
# 		file2.writelines(l)
# 		count += 1
