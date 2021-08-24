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



hdt_source = HDTDocument("typeA.hdt")
hdt_label = HDTDocument("label_May.hdt")
hdt_comment = HDTDocument("comment_May.hdt")

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



graph_ids = [11116, 240577, 395175, 14514123]
# graph_ids = [11116]
for graph_id in graph_ids:

	path_to_input_graph = './Evaluate_May/' + str(graph_id) + '_edges_original.csv'

	input_graph_data = pd.read_csv(path_to_input_graph)

	sources = input_graph_data['SUBJECT']
	targets = input_graph_data['OBJECT']
	all_nodes = list(set (sources).union (set(targets)))
	# output the source of each node
	# type A
	count_A = 0
	with open( str(graph_id) + '_explicit_source.nt', 'w') as output:
		for n in all_nodes:
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
	# type B
	count_B = 0
	with open( str(graph_id) + '_implicit_label_source.nt', 'w') as output:
		for n in all_nodes:
			triples, cardinality = hdt_label.search_triples(n, "", "")
			for (_, predicate, file) in triples:

				line = '<' + n + '> '
				line += '<' + predicate + '> '
				line += '<' + file + '>. \n'
				output.write(str(line))
				count_B += 1
	print ('count B ', count_B)
	# type C
	count_C = 0
	with open( str(graph_id) + '_implicit_comment_source.nt', 'w') as output:
		for n in all_nodes:
			triples, cardinality = hdt_comment.search_triples(n, "", "")
			for (_, predicate, file) in triples:

				line = '<' + n + '> '
				line += '<' + predicate + '> '
				line += '<' + file + '>. \n'
				output.write(str(line))
				count_C += 1
	print ('count C ', count_C)
