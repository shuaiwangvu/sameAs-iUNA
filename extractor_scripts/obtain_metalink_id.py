# this script extracts the weight from
# sameas_laundromat_metalink_sum_weight.hdt
#

import networkx as nx
from SameAsEqGraph import get_simp_IRI, get_namespace, get_name
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
from rdflib.namespace import XSD
import csv

PATH_META = "/home/jraad/ssd/data/identity/metalink/metalink.hdt"
hdt_metalink = HDTDocument(PATH_META)

PATH_SAMEAS_SOURCE = "./sameas_laundromat_metalink.hdt"
hdt_source = HDTDocument(PATH_SAMEAS_SOURCE)

my_has_label_in_file = "https://krr.triply.cc/krr/metalink/def/hasLabelInFile" # a relation
my_has_comment_in_file = "https://krr.triply.cc/krr/metalink/def/hasCommentInFile" # a relation
rdfs_label = "http://www.w3.org/2000/01/rdf-schema#label"
rdfs_comment = 'http://www.w3.org/2000/01/rdf-schema#comment'
my_file_IRI_prefix = "https://krr.triply.cc/krr/metalink/fileMD5/" # followed by the MD5 of the data
my_file = "https://krr.triply.cc/krr/metalink/def/File"
my_exist_in_file = "https://krr.triply.cc/krr/metalink/def/existsInFile" # a relation
my_has_num_occurences_in_files = "https://krr.triply.cc/krr/metalink/def/numOccurences" #


meta_eqSet = "https://krr.triply.cc/krr/metalink/def/equivalenceSet"
meta_comm = "https://krr.triply.cc/krr/metalink/def/Community"
meta_identity_statement = "https://krr.triply.cc/krr/metalink/def/IdentityStatement"
rdf_statement = "http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement"

rdf_subject = "http://www.w3.org/1999/02/22-rdf-syntax-ns#subject"
rdf_object = "http://www.w3.org/1999/02/22-rdf-syntax-ns#object"
rdf_predicate = "http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate"

rdf_type = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"



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
	if len (inter_section) >= 1:
		return list(inter_section)[0] #
	else:
		return None

def find_weight (id):
	cardinality = 0
	try:
	# print ('trying ', id)
		triples, cardinality = hdt_source.search_triples(id, my_exist_in_file, "")
	except:
		pass
		# print ('cannot find the id in source file')

	return cardinality


graph_ids = [11116, 240577, 395175, 14514123]
# graph_ids = [11116]

	# writer = csv.writer(output, delimiter=' ')
for graph_id in graph_ids:
	ct = Counter ()
	with open( str(graph_id) + '_metalink_id.csv', 'w') as writer:
		csv_writer = csv.writer(writer, delimiter=' ')

		path_to_input_graph = './Evaluate_May/' + str(graph_id) + '_edges_original.csv'

		input_graph_data = pd.read_csv(path_to_input_graph)

		sources = input_graph_data['SUBJECT']
		targets = input_graph_data['OBJECT']

		edge_data = zip(sources, targets)

		for (s,t) in edge_data:
			id = find_statement_id(s,t)
			if id == None:
				pass
			else:
				csv_writer.writerow([s, t, id])
