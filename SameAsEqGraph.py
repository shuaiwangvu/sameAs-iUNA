# this is an abstract class
import networkx as nx
import pandas as pd
import tldextract



def get_simp_IRI(e):
	# simplify this uri by introducing the namespace abbreviation
	ext = tldextract.extract(e)
	# ExtractResult(subdomain='af', domain='dbpedia', suffix='org')

	if 'dbpedia' == ext.domain and ext.subdomain != '' and ext.subdomain != None:
		namespace = ext.subdomain +'.'+ext.domain
	else :
		namespace = ext.domain

	short_IRI = ''

	if e.split('#') == [e] :
		if e.split('/') != [e]:
			name = e.split('/')[-1]
	else:
		name = e.split('/')[-1]

	if len (name) < 10:
		short_IRI  = namespace + ':' + name # this can be shortened !!!
	else:
		short_IRI = namespace + ':' + name[:10] + '...' # this can be shortened !!!

	return short_IRI

def get_namespace (e):
	sign_index  = 0
	if e.rfind('#') == -1: # if not found , return -1
		if e.rfind('/') != -1:
			sign_index = e.rfind('/')
	else:
		sign_index = e.rfind('#')

	# print ('\noriginal = ', e)
	# print ('sharp index = ', sign_index)
	# print ('namespace = ', e[:sign_index+1])
	return e[:sign_index+1]
