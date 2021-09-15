
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
from tarjan import tarjan
from collections import Counter
from hdt import HDTDocument, IdentifierPosition
import glob
from urllib.parse import urlparse
import gzip

subPropertyOf = 'http://www.w3.org/2000/01/rdf-schema#subPropertyOf'
sameas = 'http://www.w3.org/2002/07/owl#sameAs'
owleqP = 'http://www.w3.org/2002/07/owl#equivalentProperty'


rdfs_label = "http://www.w3.org/2000/01/rdf-schema#label"
rdfs_comment = 'http://www.w3.org/2000/01/rdf-schema#comment'
rdfs_isDefinedBy = "http://www.w3.org/2000/01/rdf-schema#isDefinedBy"
# skos_inScheme = "http://www.w3.org/2004/02/skos/core#inScheme"
# mads_scheme = "http://www.loc.gov/mads/rdf/v1#isMemberOfMADSScheme"
# skos_topConceptOf = "http://www.w3.org/2004/02/skos/core#topConceptOf"


meta_eqSet = "https://krr.triply.cc/krr/metalink/def/equivalenceSet"
meta_comm = "https://krr.triply.cc/krr/metalink/def/Community"
meta_identity_statement = "https://krr.triply.cc/krr/metalink/def/IdentityStatement"
rdf_statement = "http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement"

rdf_subject = "http://www.w3.org/1999/02/22-rdf-syntax-ns#subject"
rdf_object = "http://www.w3.org/1999/02/22-rdf-syntax-ns#object"
rdf_predicate = "http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate"

rdf_type = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

# my extension:
# https://krr.triply.cc/krr/metalink/file_MD5/<file MD5>

my_file_IRI_prefix = "https://krr.triply.cc/krr/metalink/fileMD5/" # followed by the MD5 of the data
my_file = "https://krr.triply.cc/krr/metalink/def/file"
my_exist_in_file = "https://krr.triply.cc/krr/metalink/def/existsInFile" # a relation
my_has_label_in_file = "https://krr.triply.cc/krr/metalink/def/hasLabelInFile" # a relation
my_has_comment_in_file = "https://krr.triply.cc/krr/metalink/def/hasCommentInFile" # a relation
my_has_num_occurences_in_files = "https://krr.triply.cc/krr/metalink/def/numOccurences" #


definition_in_relations = set([rdfs_isDefinedBy])
# source_relations = set([rdfs_isDefinedBy, skos_inScheme, mads_scheme, skos_topConceptOf])
label_relations = set([rdfs_label])
comment_relations = set([rdfs_comment])

PATH_LOD = "/scratch/wbeek/data/LOD-a-lot/data.hdt"
hdt_lod = HDTDocument(PATH_LOD)



# find all the transitive closure of subPropertyOf, sameAs:
def find_subPropertyOf_eqPropertyOf_closure(source_relations, sizebound):
	size = 0
	while size != len (source_relations):
		# update size record
		size = len (source_relations)

		# find all the new relations
		new_relations = set()
		for r in source_relations:
			triples, cardinality = hdt_lod.search_triples("", subPropertyOf, r)
			for s, _, _  in triples:
				new_relations.add(s)

			triples, cardinality = hdt_lod.search_triples("", owleqP, r)
			for s, _, _  in triples:
				new_relations.add(s)

			triples, cardinality = hdt_lod.search_triples(r, owleqP, "")
			for s, _, _  in triples:
				new_relations.add(s)

		source_relations = source_relations.union(new_relations)

	print ('After computing the closure (under subPropertyOf and equivalentProperty), we found ', len (source_relations), ' relations!')
	count = 0
	to_return = []
	for s in source_relations:
		s_triples, s_cardinality = hdt_lod.search_triples("", s, "")
		if s_cardinality >= sizebound:
			to_return.append(s)
			count += 1
			print ('\t',s)
			print ("\t\tWith", s_cardinality)
	print ("There are ", count, " properties (with more than", sizebound ," entries) above")
	return to_return



definition_in_relations = find_subPropertyOf_eqPropertyOf_closure(definition_in_relations, 100000)
print ("\n"*3)
label_relations = find_subPropertyOf_eqPropertyOf_closure(label_relations, 100000)
print ("\n"*3)
comment_relations = find_subPropertyOf_eqPropertyOf_closure(comment_relations, 100000)

# should export these relations
# labels
file =  open('typeB_relations.csv', 'w', newline='')
writer = csv.writer(file,  delimiter='\t')
writer.writerow(["Relation", "NumOfTriples"])
for l in label_relations:
	_, cardinality = hdt_lod.search_triples("", l, "")
	writer.writerow([l, cardinality])
# comments
file =  open('typeC_relations.csv', 'w', newline='')
writer = csv.writer(file,  delimiter='\t')
writer.writerow(["Relation", "NumOfTriples"])
for l in comment_relations:
	_, cardinality = hdt_lod.search_triples("", l, "")
	writer.writerow([l, cardinality])


#
# start = time.time()
#
# # total_unique_entities = 100000
#
# triples, cardinality = hdt_lod.search_triples("", sameas, "")
# count = 0
# collect_entities = set()
# for (s, _, o) in triples:
# 	# if count %1000 == 0:
# 	# 	print (count)
# 	count += 1
#
# 	collect_entities.add(s)
# 	collect_entities.add(o)
# 	# if len (collect_entities) >= total_unique_entities:
# 	# 	break
# print ('found ', len (collect_entities), ' unique entities in ', count, ' sameas triples')
#
# total_unique_entities = len (collect_entities)
#
# end = time.time()
# hours, rem = divmod(end-start, 3600)
# minutes, seconds = divmod(rem, 60)
# print("Time taken: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))


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


def decode_pair(predicate, b_subject, b_object):
	subject = None
	object = None

	(subject, object) = decode_utf8(b_subject, b_object)
	if subject != None and object != None:
		# print ('subject = ', subject)
		# print ('object = ', object)
		# print ('predicate = ', predicate)
		triples, cardinality = hdt_lod.search_triples(subject, predicate, object)
		if cardinality != 0:
			return (subject, object, 'utf8')
		else:
			(subject, object) = decode_latin1(b_subject, b_object)
			if subject != None and object != None:
				triples, cardinality = hdt_lod.search_triples(subject, predicate, object)
				if cardinality != 0:
					return (subject, object, 'latin1')
				else:
					(subject, object) = decode_cp1252(b_subject, b_object)
					if subject != None and object != None:
						triples, cardinality = hdt_lod.search_triples(subject, predicate, object)
						if cardinality != 0:
							return (subject, object,  'cp1252')
						else:
							return None
							print ('not found after all trying: ', subject, ' -> ', object)
	return None



which = 'sameas'

full_URI = ''
short_URI = ''

if which == 'broader':
	full_URI = 'http://www.w3.org/2004/02/skos/core#broader'
	short_URI = 'skos:broader'
elif which == 'subclass':
	full_URI = 'http://www.w3.org/2000/01/rdf-schema#subClassOf'
	short_URI = 'rdfs:subClassOf'
elif which == 'sameas':
	full_URI = 'http://www.w3.org/2002/07/owl#sameAs'
	short_URI = 'owl:sameAs'


# The location of the data
# ZIPFILES = '/scratch/wbeek/data/LOD-Laundromat/**/**/data.nq.gz'
ZIPFILES_PATH = '/scratch/wbeek/data/LOD-Laundromat/'

# top_dir = list(ct.keys())
top_dir = []
lst = ['0', '1','2','3','4','5','6','7','8','9','a','b','c','d','e','f']
for l in lst:
	for r in lst:
		top_dir.append(l+r)

# top_dir = top_dir[2:10]

file_name_B = 'typeB_Sep15.nt'
file_B =  open(file_name_B, 'w', newline='')
writer_B = csv.writer(file_B, delimiter=' ')


file_name_C = 'typeC_Sep15.nt'
file_C = open(file_name_C, 'w', newline='')
writer_C = csv.writer(file_C, delimiter=' ')


log_file = open("TypeBC_log.tsv", 'w')
log_file_writer = csv.writer(log_file, delimiter='\t')
log_file_writer.writerow(['B', 'foundB', 'C', 'foundC', 'Time'])


count_total_B = 0
count_found_B = 0
count_total_C = 0
count_found_C = 0

for t in top_dir:

	print ('\n\n ************\nNOW let us deal with the dir ', t)
	# print ('it has ', ct[t], 'identified objects that are not URL from the data by Joe')
	ZIPFILES = ZIPFILES_PATH + t + '/**/data.nq.gz'
	# ZIPFILES = ZIPFILES_PATH + t + '/**/data.nq.gz'
	filelist = glob.glob(ZIPFILES)
	file_path=""
	print ('This directory has ', len(filelist), ' files')
	count_processed_targeting_predicate = 0

	start = time.time()
	total_files_processed = 0
	for gzfile in filelist: # may skip the first 1000 , there is no decoding error

		# print ('now working on ', gzfile)
		# if total_files_processed % 1000 == 0:
		# 	print ('processing ...', int (total_files_processed/1000), 'k')
		# 	print ('now the path is ', gzfile)
		# total_files_processed += 1
		# if total_files_processed >= 100:
		# 	break
		file_path = gzfile
		# special_file_path = ZIPFILES_PATH + 'ac/' + 'ac878d93b26a21c24114631bee123bb7/data.nq.gz'
		# file_path = special_file_path

		# print the files' name
		# print("#Starting special file at : " + file_path)

		#The name of the dataset's folder
		folder = gzfile.split("/",8)
		f = gzip.open( gzfile, 'rb') # latin-1 ,
								# encoding= 'utf-8'
								# changed the mode from rt to rb (text mode to binary mode)

		while True:
			bline = ''
			line = ''
			last = ''
			predicate = None
			try:
				bline = next(f)
				bline_split = bline.split(b' ')
				try:
					predicate = bline_split[1].decode('utf-8') [1:-1]
				except Exception as e:
					predicate = bline_split[1].decode('latin-1') [1:-1]
				else:
					pass

				md5 = folder[6]
				# subject = bline_split[0].decode('latin-1') [1:-1]
				# object = bline_split[2].decode('latin-1') [1:-1]

				if predicate in label_relations:
					count_total_B += 1

					result = decode_pair (predicate, bline_split[0], bline_split[2])
					if result != None:
						subject, object, decoding_method = result
						writer_B.writerow(['<'+subject+'>', '<'+my_has_label_in_file+'>', '<'+my_file_IRI_prefix+md5+'>', '.'])
						writer_B.writerow(['<'+my_file_IRI_prefix+md5+'>', '<'+rdf_type+'>', '<'+my_file+'>', '.'])
						count_found_B += 1
				elif predicate in comment_relations:
					count_total_C += 1
					result = decode_pair (predicate, bline_split[0], bline_split[2])
					if result != None:
						subject, object, decoding_method = result
						writer_C.writerow(['<'+subject+'>', '<'+my_has_comment_in_file+'>', '<'+my_file_IRI_prefix+md5+'>', '.'])
						writer_C.writerow(['<'+my_file_IRI_prefix+md5+'>', '<'+rdf_type+'>', '<'+my_file+'>', '.'])
						count_found_C += 1
				else:
					pass


				# print ('predicate decoded to ', predicate)

				# if predicate == full_URI:
				# 	writer.writerow([subject, object, my_file_IRI_prefix+md5])

				# if predicate in label_relations:
				# 	writer_B.writerow(['<'+subject+'>', '<'+my_has_label_in_file+'>', '<'+my_file_IRI_prefix+md5+'>', '.'])
				# 	writer_B.writerow(['<'+my_file_IRI_prefix+md5+'>', '<'+rdf_type+'>', '<'+my_file+'>', '.'])
				#
				# if predicate in comment_relations:
				# 	writer_C.writerow(['<'+subject+'>', '<'+my_has_comment_in_file+'>', '<'+my_file_IRI_prefix+md5+'>', '.'])
				# 	writer_C.writerow(['<'+my_file_IRI_prefix+md5+'>', '<'+rdf_type+'>', '<'+my_file+'>', '.'])

			except StopIteration:
				break
			except Exception as err:
				print ('error found : ', err)
				with open(which+"_exception.txt", "a") as error:
					error.write('\n\nFile path = ' +str(file_path) + '\n')
					error.write('\n\nLine = ' +str(line) + '\n')
					error.write(" Error: {}".format(err))

	end = time.time()
	hours, rem = divmod(end-start, 3600)
	minutes, seconds = divmod(rem, 60)
	time_formated = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)
	print("Time taken: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
	log_file_writer.writerow([count_total_B,count_found_B, count_total_C, count_found_C, time_formated])
	log_file.flush()
