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

	if e.split('/') == [e] :
		if e.split('#') != [e]:
			name = e.split('#')[-1]
	else:
		name = e.split('/')[-1]

	if len (name) < 10:
		short_IRI  = namespace + ':' + name
	else:
		short_IRI = namespace + ':' + name[:10] + '...'

	return short_IRI

def get_namespace (e):
	prefix, name, sign = get_name(e)
	return prefix


def get_name (e):
	name = ''
	prefix = ''
	sign = ''
	if e.rfind('/') == -1 : # the char '/' is not in the iri
		if e.split('#') != [e]: # but the char '#' is in the iri
			name = e.split('#')[-1]
			prefix = '#'.join(e.split('#')[:-1]) + '#'
			sign = '#'
		else:
			name = None
			sign = None
			prefix =  None
	else:
		name = e.split('/')[-1]
		prefix = '/'.join(e.split('/')[:-1]) + '/'
		sign = '/'

	return prefix, sign, name
