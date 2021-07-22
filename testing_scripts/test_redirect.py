# this script tests the redirection of URIs
import requests
import networkx as nx
from collections import Counter
# responses = requests.get("http://gooogle.com")
# print (type(responses))
# for response in responses.history:
# 	print(response.url)
#


# from urllib.request import urlopen
# html = urlopen("http://www.google.com/").read()
# print(html)
# import requests
# r = requests.get('http://github.com/', timeout=0.01)
# print ('test 1')
# response = requests.get("http://www.google.com/", timeout=0.1, allow_redirects=True)
# if response.history:
# 	print("Request was redirected")
# 	for resp in response.history:
# 		print(resp.status_code, resp.url)

def find_redirects (iri):
	try:
		print ('test 2')
		collect_urls = []
		response = requests.get(iri, timeout=1.0, allow_redirects=True)
		if response.history:
			print("Request was redirected")
			for resp in response.history:
				print(resp.status_code, resp.url)
				collect_urls.append(resp.url)
			print("Final destination:")
			print(response.status_code, response.url)
			collect_urls.append(response.url)
			return collect_urls
		else:
			print("Request was not redirected")
			return []
	except:
		print ('error')
		return []




def obtain_redirect_graph (graph):
	redi_graph = nx.DiGraph()
	for n in graph.nodes:
		redirected = find_redirects(n)
		for index, iri in enumerate(redirected):
			if index == len (redirected) - 1:
				pass
				# redi_graph.add_edge(iri, redirected[0])
			else:
				redi_graph.add_edge(iri, redirected[index+1])

	return redi_graph


iri = 'https://shorturl.at/HIK58'
collect_urls = find_redirects(iri)
print('my collected urls = ', collect_urls)

g = nx.DiGraph()
g.add_node(iri)

r = obtain_redirect_graph(g)
print (r.nodes())
print (r.edges())

# How many pages do exist?

# url = 'http://github.com/'
# r = requests.head(url, allow_redirects=True)
# print(r.url)

#
# import urllib.request
# res = urllib.request.urlopen(url)
# finalurl = res.geturl()
# print(finalurl)

# first step: r = requests.get('http://github.com/', allow_redirects=False)
# second step: r = requests.get('http://github.com/', allow_redirects=True)
# print (r.url)
# print (r.status_code)
# print (r.history)
#
# if there is a page corresponding to the IRI
# if there is no page corresponding to the IRI or timeout
# if there is a redirection

#
# https://photos.google.com/


# https://yago-knowledge.org/resource/Propositional_function
# http://yago-knowledge.org/resource/Propositional_function
