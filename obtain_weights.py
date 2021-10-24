# this script extracts the weight from
# sameas_laundromat_metalink_sum_weight.hdt
#

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
from z3 import *
from rdflib.namespace import XSD
import csv
from extend_metalink import *
# PATH_META = "/home/jraad/ssd/data/identity/metalink/metalink.hdt"
# /home/jraad/ssd/data/identity/metalink/metalink-2/metalink-2.hdt


PATH_LOD = "/scratch/wbeek/data/LOD-a-lot/data.hdt"
hdt_lod_a_lot = HDTDocument(PATH_LOD)


PATH_META = "/home/jraad/ssd/data/identity/metalink/metalink-2/metalink-2.hdt"
hdt_metalink = HDTDocument(PATH_META)

PATH_SAMEAS_SOURCE = "./sameas_laundromat_metalink_Oct18.hdt"
hdt_source = HDTDocument(PATH_SAMEAS_SOURCE)
#
# my_has_label_in_file = "https://krr.triply.cc/krr/metalink/def/hasLabelInFile" # a relation
# my_has_comment_in_file = "https://krr.triply.cc/krr/metalink/def/hasCommentInFile" # a relation
# rdfs_label = "http://www.w3.org/2000/01/rdf-schema#label"
# rdfs_comment = 'http://www.w3.org/2000/01/rdf-schema#comment'
# my_file_IRI_prefix = "https://krr.triply.cc/krr/metalink/fileMD5/" # followed by the MD5 of the data
# my_file = "https://krr.triply.cc/krr/metalink/def/File"
# my_exist_in_file = "https://krr.triply.cc/krr/metalink/def/existsInFile" # a relation
# my_has_num_occurences_in_files = "https://krr.triply.cc/krr/metalink/def/numOccurences" #
#
#
# meta_eqSet = "https://krr.triply.cc/krr/metalink/def/equivalenceSet"
# meta_comm = "https://krr.triply.cc/krr/metalink/def/Community"
# meta_identity_statement = "https://krr.triply.cc/krr/metalink/def/IdentityStatement"
# rdf_statement = "http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement"
#
# rdf_subject = "http://www.w3.org/1999/02/22-rdf-syntax-ns#subject"
# rdf_object = "http://www.w3.org/1999/02/22-rdf-syntax-ns#object"
# rdf_predicate = "http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate"
#
# rdf_type = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"


# there are in total 28 entities. 14 each
validate_single = [96073, 712342, 9994282, 18688, 1140988, 25604]
validate_multiple = [6617, 4170, 42616, 39036, 33122, 6927, 11116, 12745]
validation_set = validate_single + validate_multiple

evaluation_single = [9411, 9756, 97757, 99932, 337339, 1133953]
evaluation_multiple = [5723, 14872, 37544, 236350, 240577, 395175, 4635725, 14514123]
evaluation_set = evaluation_single + evaluation_multiple


gs = validation_set + evaluation_set



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

def find_weight (id):
	cardinality = 0
	try:
	# print ('trying ', id)
		triples, cardinality = hdt_source.search_triples(id, my_exist_in_file, "")
	except:
		pass
		# print ('cannot find the id in source file')

	return cardinality


def obtain_edges(g):
	for n in g.nodes():
		(triples, cardi) = hdt_lod_a_lot.search_triples(n, sameas, "")
		for (_,_,o) in triples:
			if o in g.nodes():
				if n != o:
					g.add_edge(n, o)
		(triples, cardi) = hdt_lod_a_lot.search_triples("", sameas, n)
		for (s,_,_) in triples:
			if s in g.nodes():
				if s != n:
					g.add_edge(s, n)
	return g


# graph_ids = [11116, 240577, 395175, 14514123]
graph_ids = [11116]

	# writer = csv.writer(output, delimiter=' ')


def read_file (file_name):
	pairs = []
	eq_file = open(file_name, 'r')
	reader = csv.DictReader(eq_file, delimiter='\t',)
	for row in reader:
		s = row["Entity"]
		o = row["Annotation"]
		pairs.append((s,o))
	return pairs

count_missing_id = 0
count_total_edges = 0
count_weight_distribution = Counter ()


for id in gs:
	print ('\n\n This is graph ', id)
	count_no_metalink_id = 0
	count_reflexive = 0
	dir = './gold/'
	weight_filename = dir + str(id) +'_sum_weight.nt'
	ct = Counter()
	with open(weight_filename, 'w') as writer:
		print ('\n***************\n')

		filename = dir + str(id) +'.tsv'
		pairs = read_file(filename)

		g = nx.DiGraph()

		for (e, a) in pairs:
			g.add_node(e, annotation = a)

		# step 1: obtain the whole graph
		obtain_edges(g)
		print ('There are ', g.number_of_nodes(), ' nodes')
		print ('There are ', g.number_of_edges(), ' edges')

		for (s,t) in g.edges():
			count_total_edges += 1
			if s == t :
				count_reflexive += 1
				print ('Reflexive edges!!!!!!\n\n\n\n')
			else:
				id = find_statement_id(s,t)
				if id == None:
					# pass
					line = s + '\t' + t + ' \n'
					# line += '<' + my_has_num_occurences_in_files + '> '
					# line += '"'+str(weight)+'"^^<' + str(XSD.integer) +
					# print (line)
					# writer_not_found.write(str(line))
					# print('not fou')
					count_no_metalink_id += 1
				else:
					weight = find_weight(id)
					# if weight == 0:
					# 	pass
						# print (s, ' -> ', t, ' has no weight')
					ct[weight] += 1
					# export the weight
					# print (XSD.integer)
					line = '<' + str(id) + '> '
					line += '<' + my_has_num_occurences_in_files + '> '
					line += '"'+str(weight)+'"^^<' + str(XSD.integer) + '> . \n'
					# print (line)
					writer.write(str(line))
		print ('# not found metalink id = ', count_no_metalink_id)
		print ('count reflexive = ', count_reflexive)

	count_missing_id += count_no_metalink_id

	for w in ct.keys():
		count_weight_distribution[w] += ct[w]
		print (ct[w], ' has weight ', w)

print ('total edges: ', count_total_edges)
print ('edges without id ', count_missing_id, ' -> ', count_missing_id/count_total_edges)
print ('edges without weight ', count_weight_distribution[0], ' -> ', count_weight_distribution[0]/count_total_edges )
print ('weight distibution ', count_weight_distribution)

# for id in graph_ids:
# 	dir = './gold/'
# 	weight_filename = dir + str(id) +'_sum_weight.nt'
# 	print (weight_filename)
