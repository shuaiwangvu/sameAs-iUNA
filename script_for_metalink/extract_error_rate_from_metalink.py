
import numpy as np
import datetime
import pickle
import time
import networkx as nx
import sys
import csv
import json
import random
from collections import Counter
from hdt import HDTDocument, IdentifierPosition
from urllib.parse import urlparse
import gzip

from SameAsEqGraph import *


PATH_META = "/home/to/data/identity/metalink/metalink.hdt"
hdt_metalink = HDTDocument(PATH_META)

PATH_META2 = "/home/to/ssd/data/identity/metalink/metalink-2/metalink-2.hdt"
hdt_metalink2 = HDTDocument(PATH_META)

meta_eqSet = "https://krr.triply.cc/krr/metalink/def/equivalenceSet"
meta_comm = "https://krr.triply.cc/krr/metalink/def/Community"
meta_identity_statement = "https://krr.triply.cc/krr/metalink/def/IdentityStatement"
rdf_statement = "http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement"
meta_error = "https://krr.triply.cc/krr/metalink/def/error"

rdf_subject = "http://www.w3.org/1999/02/22-rdf-syntax-ns#subject"
rdf_object = "http://www.w3.org/1999/02/22-rdf-syntax-ns#object"
rdf_predicate = "http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate"

rdf_type = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

# my extension:
# https://krr.triply.cc/krr/metalink/fileMD5/<file MD5>
my_file_IRI_prefix = "https://krr.triply.cc/krr/metalink/fileMD5/" # followed by the MD5 of the data
my_file = "https://krr.triply.cc/krr/metalink/def/File"
my_exist_in_file = "https://krr.triply.cc/krr/metalink/def/existsInFile" # a relation
my_has_num_occurences_in_files = "https://krr.triply.cc/krr/metalink/def/numOccurences" #
my_redirect = "https://krr.triply.cc/krr/metalink/def/redirectedTo" # a relation


sameas = "http://www.w3.org/2002/07/owl#sameAs"

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

	# do it the reverse way: (object, predicate, subject)
	triples, cardinality = hdt_metalink.search_triples("", rdf_object, subject)
	collect_statement_id_regarding_subject = set()

	for (s,p,o) in triples:
		collect_statement_id_regarding_subject.add(str(s))

	triples, cardinality = hdt_metalink.search_triples("", rdf_subject, object)

	collect_statement_id_regarding_object = set()

	for (s,p,o) in triples:
		collect_statement_id_regarding_object.add(str(s))

	inter_section2 = collect_statement_id_regarding_object.intersection(collect_statement_id_regarding_subject)

	if len (inter_section) >= 1:
		return list(inter_section)[0] #
	elif len (inter_section2) >= 1:
		# print ('\nfound one in reverse!: \n', subject, '\t', object)
		return list(inter_section2)[0] #:
	else:
		return None



def find_statement_id2(subject, object):

	triples, cardinality = hdt_metalink2.search_triples("", rdf_subject, subject)
	collect_statement_id_regarding_subject = set()

	for (s,p,o) in triples:
		collect_statement_id_regarding_subject.add(str(s))

	triples, cardinality = hdt_metalink2.search_triples("", rdf_object, object)

	collect_statement_id_regarding_object = set()

	for (s,p,o) in triples:
		collect_statement_id_regarding_object.add(str(s))

	inter_section = collect_statement_id_regarding_object.intersection(collect_statement_id_regarding_subject)

	# do it the reverse way: (object, predicate, subject)
	triples, cardinality = hdt_metalink2.search_triples("", rdf_object, subject)
	collect_statement_id_regarding_subject = set()

	for (s,p,o) in triples:
		collect_statement_id_regarding_subject.add(str(s))

	triples, cardinality = hdt_metalink2.search_triples("", rdf_subject, object)

	collect_statement_id_regarding_object = set()

	for (s,p,o) in triples:
		collect_statement_id_regarding_object.add(str(s))

	inter_section2 = collect_statement_id_regarding_object.intersection(collect_statement_id_regarding_subject)

	if len (inter_section) >= 1:
		return list(inter_section)[0] #
	elif len (inter_section2) >= 1:
		# print ('\nfound one in reverse!: \n', subject, '\t', object)
		return list(inter_section2)[0] #:
	else:
		return None


def decode_utf8 (b_subject, b_object):
	subject = None
	object = None
	try:
		subject = b_subject.decode('utf-8') [1:-1]
		object = b_object.decode('utf-8') [1:-1]
	except Exception as e:
		return (None, None)
	else:
		return (subject, object)

def decode_latin1 (b_subject, b_object):
	subject = None
	object = None
	try:
		subject = b_subject.decode('latin-1') [1:-1]
		object = b_object.decode('latin-1') [1:-1]
	except Exception as e:
		return (None, None)
	else:
		return (subject, object)

# cp1252 : Windows-1252(cp1252)
def decode_cp1252 (b_subject, b_object):
	subject = None
	object = None
	try:
		subject = b_subject.decode('cp1252') [1:-1]
		object = b_object.decode('cp1252') [1:-1]
	except Exception as e:
		return (None, None)
	else:
		return (subject, object)

def decode_pair(b_subject, b_object):
	subject = None
	object = None

	(subject, object) = decode_utf8(b_subject, b_object)
	if subject != None and object != None:
		id = find_statement_id(subject, object)
		if id != None:
			# print ('found id when decoding using utf8')
			return (subject, object, id, 'utf8')
		else:
			(subject, object) = decode_latin1(b_subject, b_object)
			if subject != None and object != None:
				id = find_statement_id(subject, object)
				if id != None:
					# print ('found id when decoding using latin-1')
					return (subject, object, id, 'latin1')
				else:
					(subject, object) = decode_cp1252(b_subject, b_object)
					id = find_statement_id(subject, object)
					if id != None:
						# print ('found id when decoding using cp1252')
						return (subject, object, id, 'cp1252')
					else:
						return None
						# print ('not found after all trying: ', subject, ' -> ', object)
	return None


count_short = 0

count_sameAs_statement = 0
count_sameAs_statement_with_metalinkID = 0


def get_error_rate (edge_id):
	# meta_error
	triples, cardinality = hdt_metalink.search_triples(edge_id, meta_error, "")
	if cardinality == 1:
		for (s,p,o) in triples:
			return o
	else:
		print ('this edge does not have an id in Metalink')
		return None

def get_error_rate2 (edge_id):
	# meta_error
	triples, cardinality = hdt_metalink2.search_triples(edge_id, meta_error, "")
	if cardinality == 1:
		for (s,p,o) in triples:
			return o
	else:
		print ('this edge does not have an id in Metalink')
		return None


# load a file
validation_single = [96073, 712342, 9994282, 18688, 1140988, 25604]
validation_multiple = [33122, 11116, 12745, 6617,4170, 42616, 6927, 39036]
validation_set = validation_single + validation_multiple
# the evaluation set
evaluation_single = [9411, 9756, 97757, 99932, 337339, 1133953]
evaluation_multiple = [5723, 14872, 37544, 236350, 240577, 395175, 4635725, 14514123]
evaluation_set = evaluation_single + evaluation_multiple


gs = validation_set + evaluation_set


for graph_id in gs:
	print ('\n\n\ngraph id = ', str(graph_id))
	dir = './gold/'
	path_to_nodes = dir + str(graph_id) +'.tsv'
	path_to_edges = dir + str(graph_id) +'_edges.tsv'
	g = load_graph(path_to_nodes, path_to_edges)
	g = nx.Graph(g)

	file = open( dir + str(graph_id)+"_edges_with_Metalink_edge_id_and_error_degree.tsv", 'w')
	file_writer = csv.writer(file, delimiter='\t')
	file_writer.writerow(["SUBJECT", "OBJECT", "METALINK_ID", "ERROR_DEGREE"])
	for (s, t) in g.edges():
		edge_id = find_statement_id(s,t)
		edge_error = None
		if edge_id == None:
			print ('found error: no edge id found')
		else:
			# print (s, t, 'has edge id: ', edge_id)
			edge_error = get_error_rate(str(edge_id))
			if edge_error == None :
				print ('no error rate found')
			else:
				pass
				# print (s, t, 'has error rate: ', edge_error)

		edge_id2 = find_statement_id(s,t)
		edge_error2 = None
		if edge_id2 == None:
			print ('found error: no edge id found')
		else:
			# print (s, t, 'has edge id: ', edge_id)
			edge_error2 = get_error_rate(str(edge_id))
			if edge_error2 == None :
				print ('no error rate found')
			else:
				pass
				# print (s, t, 'has error rate: ', edge_error)
		if edge_id != edge_id2 :
			print ('not the same id!')

		if edge_error != edge_error2 :
			print ('not the same error rate!')


		file_writer.writerow([s, t, edge_id, edge_error])
