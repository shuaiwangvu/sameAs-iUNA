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


PATH_META = "/home/jraad/ssd/data/identity/metalink/metalink.hdt"
hdt = HDTDocument(PATH_META)

which = 'sameas'

full_IRI = ''
short_URI = ''

if which == 'broader':
	full_IRI = 'http://www.w3.org/2004/02/skos/core#broader'
	short_URI = 'skos:broader'
elif which == 'subclass':
	full_IRI = 'http://www.w3.org/2000/01/rdf-schema#subClassOf'
	short_URI = 'rdfs:subClassOf'
elif which == 'sameas':
	full_IRI = 'http://www.w3.org/2002/07/owl#sameAs'
	short_URI = 'owl:sameAs'


# ct = {'bd': 423659, '34': 396655, 'af': 371122, 'f6': 276836, '95': 257863, 'd3': 224435, '67': 200423, '0f': 198868, '2f': 194825, 'e6': 188076, 'ff': 159546, '7d': 134168, '42': 106602, 'e8': 96957, '0d': 96086, 'a4': 93918, 'be': 89011, '79': 77519, 'a8': 64511, '50': 63639, 'e7': 62502, '46': 60773, '63': 55764, '61': 54232, '17': 52460, 'df': 52292, '76': 51586, '78': 50286, 'f1': 48343, 'ac': 46196, 'ad': 45772, '29': 42244, '03': 40617, 'ab': 40025, '73': 39140, '48': 38998, '4d': 37676, '56': 32671, '7c': 31508, '6d': 30687, '85': 28662, '77': 27172, 'bb': 27125, 'b0': 24974, 'c6': 24626, 'b5': 22933, '2b': 22208, '70': 21371, '1e': 19887, 'ea': 18756, 'fc': 18611, '35': 16274, 'f8': 15521, 'ca': 15273, '4e': 15172, '54': 14541, 'fa': 14516, '6a': 14498, '01': 14294, 'd8': 14123, 'a7': 13652, '1b': 12419, 'cb': 12276, 'b8': 12054, '8e': 11652, '94': 11158, 'ht': 10882, '7e': 10858, '1c': 10406, '7a': 10355, 'ce': 10295, '5b': 10274, '16': 9740, '3e': 9730, 'e2': 7717, '3f': 6749, 'd5': 5205, '8b': 4981, '9b': 4928, '44': 4831, '45': 4783, 'c2': 4763, '57': 4343, '55': 3647, '37': 3619, '8a': 3275, '36': 2827, '19': 2812, 'ef': 2112, 'f0': 1885, 'b3': 1866, 'ed': 1856, 'bf': 1806, 'dd': 1714, '25': 1207, '4c': 1097, '4f': 1061, 'cc': 1056, '71': 914, '6f': 896, '68': 874, 'c1': 794, '31': 753, 'aa': 712, '18': 686, '43': 664, 'eb': 659, 'd2': 510, 'cf': 498, '64': 437, '2d': 218, 'd4': 189, '6c': 186, 'c5': 172, 'e4': 171, '87': 142, 'a2': 138, '08': 132, '65': 131, '2e': 123, '5f': 73, '5a': 48, 'e3': 39, 'c4': 34, '49': 33, '97': 33, '13': 27, '69': 26, '5e': 25, 'b9': 24, '7f': 21, '2a': 18, '86': 16, '38': 15, '4b': 15, '': 12, '81': 11, '80': 11, 'fd': 10, '3d': 10, '3b': 10, '3a': 10, '06': 9, 'c3': 8, 'f7': 8, '8c': 6, 'hs': 5, 'ra': 3, '51': 3, '30': 2, '40': 2, 'n': 2, '7b': 2, '22': 2, 'yp': 2, 'es': 2, 'ou': 2, '0c': 1, 'ar': 1, 'ae': 1, '32': 1, '99': 1, '9e': 1, '90': 1, 'c0': 1}



meta_eqSet = "https://krr.triply.cc/krr/metalink/def/equivalenceSet"
meta_comm = "https://krr.triply.cc/krr/metalink/def/Community"
meta_identity_statement = "https://krr.triply.cc/krr/metalink/def/IdentityStatement"
rdf_statement = "http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement"

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



# The location of the data
# ZIPFILES = '/scratch/wbeek/data/LOD-Laundromat/**/**/data.nq.gz'
ZIPFILES_PATH = '/scratch/wbeek/data/LOD-Laundromat/'

# top_dir = list(ct.keys())
top_dir = []
# lst = ['0', '1','2','3','4','5','6','7','8','9','a','b','c','d','e','f']
lst = ['0', '1', '2']
for l in lst:
	for r in lst:
		top_dir.append(l+r)





def find_statement_id(subject, object):

	triples, cardinality = hdt.search_triples("", rdf_subject, subject)
	collect_statement_id_regarding_subject = set()

	for (s,p,o) in triples:
		collect_statement_id_regarding_subject.add(str(s))

	triples, cardinality = hdt.search_triples("", rdf_object, object)

	collect_statement_id_regarding_object = set()

	for (s,p,o) in triples:
		collect_statement_id_regarding_object.add(str(s))

	inter_section = collect_statement_id_regarding_object.intersection(collect_statement_id_regarding_subject)
	if len (inter_section) >= 1:
		return list(inter_section)[0] #
	else:
		return None



count_short = 0


with open( which + "_laundromat_metalink.nt", 'w') as output:
	writer = csv.writer(output, delimiter=' ')

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
		# total_files_processed = 0
		for gzfile in filelist: # may skip the first 1000 , there is no decoding error

			# print ('now working on ', gzfile)
			# if total_files_processed % 1000 == 0:
			# 	print ('processing ...', int (total_files_processed/1000), 'k')
			# 	print ('now the path is ', gzfile)
			# total_files_processed += 1
			# if total_files_processed >= 3000:
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
				try:
					bline = next(f)
					bline_split = bline.split(b' ')

					if len (bline_split) == 4:
						object = bline_split[2].decode('latin-1')
						if object[0] != '"':
							predicate = bline_split[1].decode('latin-1') [1:-1]
							if predicate == full_IRI:
								subject = bline_split[0].decode('latin-1') [1:-1]
								# now we have the subject, predicate, object

								statementID = find_statement_id (subject, object)
								if statementID != None:
									# out put two lines
									# 1) metalink_id has a source file+MD5
									writer.writerow(['<'+statementID+'>', '<'+my_exist_in_file+'>', '<'+my_file_IRI_prefix+md5+'>', '.'])
									# 2) source file+MD5 is a of type file
									writer.writerow(['<'+my_file_IRI_prefix+md5+'>', '<'+rdf_type+'>', '<'+my_file+'>', '.'])

								output.write(str(line))

				except StopIteration:
					break
				except Exception as err:
					print ('error found : ', err)
					with open(which+"_exception.txt", "a") as error:
						error.write('\n\nFile path = ' +str(file_path) + '\n')
						error.write('\n\nLine = ' +str(line) + '\n')
						error.write(" Error: {}".format(err))
				# except UnicodeDecodeError:
				# 	print ('\n\ndecoding error, ', last.decode('latin-1'))
				# 	tec  = chardet.detect(last)
				# 	print (tec)
				# 	if tec['encoding'] != None and tec['encoding'] != "utf-8" and tec['encoding'] != "utf8":
				# 		print ('decode = ', last.decode(tec['encoding']))
				# 	else:
				# 		print ('decoding as latin-1 = ', last.decode('latin-1'))
				# 		print ('decoding as ISO-8859-9 = ', last.decode('ISO-8859-9'))
				# except Exception as err:
				# 		print ('error found : ', err)
					# if tec['encoding'] == 'ISO-8859-1':
					# 	print ('decoding as ISO-8859-1: ', last.decode('iso-8859-1'))
					# elif tec['encoding'] == 'Windows-1252':
					# 	print ('decoding as Windows-1252: ', last.decode('cp1252'))
					# elif tec['encoding'] == 'Windows-1254':
					# 	print ('decoding as Windows-1254: ', last.decode('cp1254'))
					# else:
					# 	print ('decoding as latin-1: ', last.decode('latin-1'))

					# code
					# # current_file = list(f)
					# for line in current_file:
					# 	try:
					# 	decoded_line = line.decode()
					# 	#check if the file contains this property
					# 	if full_IRI in line or short_URI in line:
					# 		# remove .\n from the end of the line
					# 		line = line.replace(' .\n','')
					# 		# Add to the line: The most frequent site + the dataset's folder's name + end to the line
					# 		line = line + " <" + folder[6]+"/"+folder[7]+"> .\n"
					# 		# raise Exception(line)
					# 		# output the result to the output file
					# 		output.write(str(line))
					#
					# 		# throw an error with mentioning the file's path and the error.
					# 		# then store it to the exception File.
					# 	except Exception as err:
					# 		print ('error found : ', err)
					# 		with open(which+"_exception.txt", "a") as error:
					# 			error.write('\n\nFile path = ' +str(file_path) + '\n')
					# 			error.write(" Line: {}".format(err))
		# 	total_files_processed += 1
		# print ("I processed ", total_files_processed , ' files')



		end = time.time()
		hours, rem = divmod(end-start, 3600)
		minutes, seconds = divmod(rem, 60)
		print("Time taken: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

print ('count_short_URI ', count_short)
