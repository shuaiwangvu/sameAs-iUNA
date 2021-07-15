# this is an abstract class
import networkx as nx
import pandas as pd
import tldextract



def simp(e):
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

	# print ('\n\noriginal e = ', e)
	# print ('short_IRI =  ', short_IRI)

	return short_IRI
