# This is the script we used to perform data analysis (Table 3)
# The script samples some pairs and examines the error rates
# regarding various definitions of UNAs. 

import networkx as nx
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
from z3 import *
import csv
from rdflib import Literal, XSD
from networkx.algorithms.connectivity import is_locally_k_edge_connected
from SameAsEqGraph import *
from extend_metalink import *
import csv
import random
import time
# import copy
from math import exp

import numpy as np
from GraphSolver import GraphSolver

SMT_UNKNOWN = 0

GENERAL = 0
EFFECIENT = 1
FINETUNNED = 2


MODE = EFFECIENT

WITHWEIGHT = False

# source_switch =  'implicit_comment_source'
source_switch =  'implicit_label_source'

# ===================

UNKNOWN = 0
REMOVE = 1
KEEP = 2

hdt_source = None
hdt_label = None
hdt_comment = None
hdt_disambiguation = None

NOTFOUND = 1
NOREDIRECT = 2
ERROR = 3
TIMEOUT = 4
REDIRECT = 5


# there are in total 28 entities. 14 each
validate_single = [96073, 712342, 9994282, 18688, 1140988, 25604]
validate_multiple = [33122, 11116,   12745, 6617,4170, 42616, 6927, 39036]
validation_set = validate_single + validate_multiple

evaluation_single = [9411, 9756, 97757, 99932, 337339, 1133953]
evaluation_multiple = [5723, 14872, 37544, 236350, 240577, 395175, 4635725, 14514123]
evaluation_set = evaluation_single + evaluation_multiple


gs = validation_set + evaluation_set

restricted_prefix_list = ["http://dblp.rkbexplorer.com/id/",
"http://dbpedia.org/resource/",
"http://rdf.freebase.com/ns/m/",
"http://sws.geonames.org/",
"http://dbtune.org/musicbrainz/resource/",
"http://bio2rdf.org/uniprot:"]


avg_precision = 0
avg_recall = 0
num_edges_removed = 0

count_valid_result = 0
count_invalid_result = 0

hard_graph_ids = [39036, 33122, 11116, 6927]
# graph_ids = hard_graph_ids
# graph_ids = evaluation_multiple
graph_ids = gs
# hard_graph_ids
start = time.time()

# part 1: understand the baseline
sample_total_edges = 0
sample_total_correct_edges = 0
sample_total_error_edges = 0
total_entities = 0
total_edges = 0

for graph_id in graph_ids:
	# print ('\n\n\ngraph id = ', str(graph_id))
	dir = './gold/'
	gsolver = GraphSolver(dir, graph_id)
	gtotal, gcorrect, gerror = gsolver.random_sample_error_rate()
	sample_total_edges += gtotal
	sample_total_correct_edges += gcorrect
	sample_total_error_edges += gerror
	total_entities += gsolver.input_graph.number_of_nodes()
	total_edges += gsolver.input_graph.number_of_edges()

print ('number of files examined: ', len (graph_ids))
print ('number of entities: ', total_entities)
print ('number of edges: ', total_edges)
print ('total edges generated: ', sample_total_edges)
print ('total correct edges: ', sample_total_correct_edges)
print ('total error edges: ', sample_total_error_edges)
print ('\ncorrect rate{:10.4f}'.format(sample_total_correct_edges/sample_total_edges))
print ('error rate{:10.4f}'.format(sample_total_error_edges/sample_total_edges))
print ('the error rate is therefore between{:10.4f}'.format(sample_total_error_edges/sample_total_edges))
print('and {:10.4f}'.format(1- sample_total_correct_edges/sample_total_edges))

# part 2: understand how violation of UNA can be used for identifying errors
sample_total_edges = 0
sample_total_vio_edges_nUNA = 0
sample_total_correct_edges_nUNA = 0
sample_total_error_edges_nUNA = 0

sample_total_vio_edges_qUNA = 0
sample_total_correct_edges_qUNA = 0
sample_total_error_edges_qUNA = 0

sample_total_vio_edges_iUNA = 0
sample_total_correct_edges_iUNA = 0
sample_total_error_edges_iUNA = 0

for gid in graph_ids:
	# print ('\n\n\ngraph id = ', str(graph_id))
	dir = './gold/'
	gsolver = GraphSolver(dir, graph_id=gid)
	gsolver.source_switch = source_switch
	gtotal, gvio_nUNA, gcorrect_nUNA, gerror_nUNA, gvio_qUNA, gcorrect_qUNA, gerror_qUNA, gvio_iUNA, gcorrect_iUNA, gerror_iUNA = gsolver.random_sample_error_rate_UNA()
	sample_total_edges += gtotal

	sample_total_vio_edges_nUNA += gvio_nUNA
	sample_total_correct_edges_nUNA += gcorrect_nUNA
	sample_total_error_edges_nUNA += gerror_nUNA

	sample_total_vio_edges_qUNA += gvio_qUNA
	sample_total_correct_edges_qUNA += gcorrect_qUNA
	sample_total_error_edges_qUNA += gerror_qUNA

	sample_total_vio_edges_iUNA += gvio_iUNA
	sample_total_correct_edges_iUNA += gcorrect_iUNA
	sample_total_error_edges_iUNA += gerror_iUNA

print ('\n\n nUNA \n')

print ('total edges generated: ', sample_total_edges)
print ('total edges generated that violates nUNA: ', sample_total_vio_edges_nUNA)
print ('total correct edges: ', sample_total_correct_edges_nUNA)
print ('total error edges: ', sample_total_error_edges_nUNA)
print ('\ncorrect ratio{:10.4f}'.format(sample_total_correct_edges_nUNA/sample_total_vio_edges_nUNA))
print ('error ratio {:10.4f}'.format(sample_total_error_edges_nUNA/sample_total_vio_edges_nUNA))
print ('the error ratio is therefore between{:10.4f}'.format(sample_total_error_edges_nUNA/sample_total_vio_edges_nUNA))
print('and {:10.4f}'.format(1- sample_total_correct_edges_nUNA/sample_total_vio_edges_nUNA))

print ('\n\n qUNA \n')

print ('total edges generated: ', sample_total_edges)
print ('total edges generated that violates qUNA: ', sample_total_vio_edges_qUNA)
print ('total correct edges: ', sample_total_correct_edges_qUNA)
print ('total error edges: ', sample_total_error_edges_qUNA)
print ('\ncorrect ratio{:10.4f}'.format(sample_total_correct_edges_qUNA/sample_total_vio_edges_qUNA))
print ('error ratio{:10.4f}'.format(sample_total_error_edges_qUNA/sample_total_vio_edges_qUNA))
print ('the error ratio is therefore between{:10.4f}'.format(sample_total_error_edges_qUNA/sample_total_vio_edges_qUNA))
print('and {:10.4f}'.format(1- sample_total_correct_edges_qUNA/sample_total_vio_edges_qUNA))


print ('\n\n iUNA \n')


print ('total edges generated: ', sample_total_edges)
print ('total edges generated that violates iUNA: ', sample_total_vio_edges_iUNA)
print ('total correct edges: ', sample_total_correct_edges_iUNA)
print ('total error edges: ', sample_total_error_edges_iUNA)
print ('\ncorrect ratio{:10.4f}'.format(sample_total_correct_edges_iUNA/sample_total_vio_edges_iUNA))
print ('error ratio{:10.4f}'.format(sample_total_error_edges_iUNA/sample_total_vio_edges_iUNA))
print ('the error ratio is therefore between{:10.4f}'.format(sample_total_error_edges_iUNA/sample_total_vio_edges_iUNA))
print('and {:10.4f}'.format(1- sample_total_correct_edges_iUNA/sample_total_vio_edges_iUNA))




# end = time.time()
# hours, rem = divmod(end-start, 3600)
# minutes, seconds = divmod(rem, 60)
# time_formated = "{:0>2}:{:0>2}:{:05.f}".format(int(hours), int(minutes), seconds)
# print ('time taken: ' + time_formated)
