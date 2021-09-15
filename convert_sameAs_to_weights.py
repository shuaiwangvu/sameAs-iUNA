# this script converts the sources to its weights.

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
from extend_metalink import *

UNKNOWN = 0
REMOVE = 1
KEEP = 2

hdt_source = None
hdt_label = None
hdt_comment = None

PATH_LOD = "/scratch/wbeek/data/LOD-a-lot/data.hdt"
hdt_lod = HDTDocument(PATH_LOD)

PATH_META = "/home/jraad/ssd/data/identity/metalink/metalink-2/metalink-2.hdt"
hdt_metalink = HDTDocument(PATH_META)

PATH_SAMEAS_SOURCE = "./sameas_laundromat_metalink_Sep.hdt"
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
#
# xsd= "http://www.w3.org/2001/XMLSchema#"
#
# sameas = 'http://www.w3.org/2002/07/owl#sameAs'
# load LOD-a-lot
# for each sameAs link, compute its weight
# this is the class where SameAsEqSolver is defined



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



ct = Counter ()
count_sameas_triples_in_lod_a_lot = 0
sameAs_satement_without_metalink_id = 0
log_writer = open('sameas_laundromat_metalink_sum_weight_sep.log', 'w')

with open('sameas_laundromat_metalink_sum_weight_sep.nt', 'w') as writer:
	# writer = csv.writer(output, delimiter=' ')

	triples, cardinality = hdt_lod.search_triples("", sameas, "")
	for (s, sameas, t) in triples:
		# if count_sameas_triples_in_lod_a_lot>=10000:
		# 	break

		if s != t:
			try :
				id = find_statement_id(s, t)
				if id == None:
					sameAs_satement_without_metalink_id += 1
					line = s +' -> ' + t + ' has no id in metalink'
					log_writer.write(str(line))
				else:
					weight = find_weight(id)
					# if weight == 0:
					# 	pass
						# print (s, ' -> ', t, ' has no weight')
					ct[weight] += 1
					# export the weight
					# print (XSD.integer)
					line = '<' + id + '> '
					line += '<' + my_has_num_occurences_in_files + '> '
					line += '"'+str(weight)+'"^^<' + str(XSD.integer) + '> . \n'
					# print (line)
					writer.write(str(line))
			except Exception as inst:
				print (s, t)
				print (inst)

		count_sameas_triples_in_lod_a_lot += 1


print ('total statements processed: ', count_sameas_triples_in_lod_a_lot)
print ('statements without metalink id: ', sameAs_satement_without_metalink_id)
print (ct)

log_writer.write('total statements processed: ' + str(count_sameas_triples_in_lod_a_lot) +'\n')
log_writer.write('statements without metalink id: ' + str(sameAs_satement_without_metalink_id)+'\n')

for c in ct:
	line = str(c) + ' : ' + str(ct[c]) +'\n'
	log_writer.write(str(line))
