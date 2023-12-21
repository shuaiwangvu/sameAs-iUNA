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


# path = '/Users/sw-works/Documents/backbone/Final-SameAs-goldstandard/sameAs_data/id2terms_0-99.csv'
path = '/home/jraad/ssd/data/identity-data/identity-data/id2terms_original.csv'

biggest = [4073]
biggest_two = [4073, 142063] # and another one I don't remember
large = []

# Entity	Annotation	Comment
def export_graph_csv (file_name, entities):
	# pass
	file =  open(file_name, 'w', newline='')
	writer = csv.writer(file, delimiter='\t')
	writer.writerow(['Entity', 'Annotation', 'Comment'])
	count_real_entities = 0
	for e in entities:
		if e[0] == '"':
			# writer.writerow([e, 'unknown', 'unknown'])
			pass
			# print ('here:', e)
		elif e[0] == '<':
			writer.writerow([e[1:-1], 'unknown', 'unknown'])
			count_real_entities += 1
		else:
			pass
			# writer.writerow([e, 'unknown', 'unknown'])
			# print ('there:', e)
	print ('# real entities ', count_real_entities)
	return count_real_entities

def export_index_and_size (file_name, index_to_size, index_to_num_real_entities = None):
	file =  open(file_name, 'w', newline='')
	writer = csv.writer(file, delimiter='\t')

	if index_to_num_real_entities == None:
		writer.writerow(['Index', 'Size'])
		for index in index_to_size.keys():
			writer.writerow([index, index_to_size[index]])
	else:
		writer.writerow(['Index', 'Size', 'Entities_without_literals'])
		for index in index_to_size.keys():
			writer.writerow([index, index_to_size[index], index_to_num_real_entities[index]])


# count = 0
with open(path) as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=' ')
	index_to_size = {}
	index_to_size_1000 = {}
	index_to_size_1000_real = {}
	entities = []
	for row in csv_reader:
		# if count > 10000000: # total lines :44676381
		# 	break
		# count += 1
		index = int(row[0])
		size = len (row) - 1
		index_to_size[index] = size

		if size >= 1000:
			index_to_size_1000 [index] = size
			entities = row[1:]
			path_to_file = './big_connected_components/' +str(index)+'.tsv'
			count_real_entities = export_graph_csv(path_to_file, entities)
			index_to_size_1000_real [index] = count_real_entities
		# print ('index = ', index)
		# print ('size = ', size)
	# sort the dictionary by values
	# print(index_to_size)
	sorted_index_to_size = {k: v for k, v in sorted(index_to_size.items(), key=lambda item: item[1])}
	sorted_index_to_size_1000 = {k: v for k, v in sorted(index_to_size_1000.items(), key=lambda item: item[1])}
	# print ('count lines = ', count)
	# for index in sorted_index_to_size.keys():
	# 	print (index, ' has ', sorted_index_to_size[index],' entities')

	export_index_and_size('sameas_index_to_size.tsv', sorted_index_to_size)
	export_index_and_size('sameas_index_to_size_1000.tsv', sorted_index_to_size_1000, index_to_size_1000_real)
