
from hdt import HDTDocument, IdentifierPosition
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

PATH_LOD = "/scratch/wbeek/data/LOD-a-lot/data.hdt"
hdt = HDTDocument(PATH_LOD)

a = "http://dbpedia.org/resource/Bourgogne_(disambiguation)"
b = "http://ca.dbpedia.org/resource/Funció_(desambiguació)"
c = 'http://dbpedia.org/resource/Pensacola_(disambiguation)'

print ('now testing: ', c)
triples, total_triples = hdt.search_triples(c, "", "")
for (s, p, o) in triples:
	print (p, o)

p = "http://dbpedia.org/property/wikiPageUsesTemplate"
o = "http://dbpedia.org/resource/Template:Disambig"
#
# p = 'http://dbpedia.org/ontology/wikiPageDisambiguates'
# # between disambiguating entities
#
# p = "http://dbpedia.org/ontology/wikiPageInterLanguageLink"
# # between entities of different languages
#
# p = "http://dbpedia.org/property/wikilink"
# o = "http://dbpedia.org/resource/Pensacola_Bay"
#
# p = "http://www.w3.org/ns/prov#wasDerivedFrom"
# o = "http://en.wikipedia.org/wiki/Pensacola_(disambiguation)?oldid=491488608"
