# This is a new sameAs

import gzip
import glob
from urllib.parse import urlparse
import collections
import datetime
import time
import chardet
import validators
import csv

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
my_has_num_occurences_in_files = "https://krr.triply.cc/krr/metalink/def/numOccurences" #
rdf_type = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"


count_short = 0

with open( which + "_laundromat_new.tsv", 'w', newline='') as output:
	writer = csv.writer(output, delimiter='\t')

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
			if total_files_processed % 1000 == 0:
				print ('processing ...', int (total_files_processed/1000), 'k')
				print ('now the path is ', gzfile)
			total_files_processed += 1
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
					predicate = bline_split[1].decode('latin-1') [1:-1]
					subject = bline_split[0].decode('latin-1') [1:-1]
					object = bline_split[2].decode('latin-1') [1:-1]

					# print ('predicate decoded to ', predicate)
					md5 = folder[6]
					if predicate == full_URI:
						writer.writerow([subject, object, my_file_IRI_prefix+md5])

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
		print("Time taken: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
