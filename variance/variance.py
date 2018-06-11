import numpy as np
import operator
import nltk
from collections import Counter
from nltk.corpus import wordnet as wn
from collections import Iterable
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import sys

def makecounter(listallhypernym,input):
	for i in input:
		target=i[0]
		try:
			synset=wn.synsets(target)
		except UnicodeDecodeError:
			pass
		if synset==[] :
			pass
		else:	
			try:
				listallhypernym.append(hypernym(synset[0]))
			except IndexError:
				pass		
	count=Counter(flatten(listallhypernym))
	sumcounter=sum(count.values())
	return(count,sumcounter)

def hypernym(word):
	hypernym=word.hypernym_paths()
	return hypernym


def flatten(seq,container=None):
    if container is None:
        container = []
    for s in seq:
        if hasattr(s,'__iter__'):
            flatten(s,container)
        else:
            container.append(s)
    return container

def main():
	if len(sys.argv) == 2:
		space=sys.argv[1]
	if space=='fullcolors':
		f=open('fullcolorsdm.txt','r')
	if space=='reducedcolors':
		f=open('reducedcolorsdm.txt','r')

	variancedict={}
	for line in f: 
		featurelist=[]
		line=line.split('\t')
		target=line[0]
		features=line[1].strip('\n')
		features=features.split(' ')
		for i in features: 
			featurelist.append(float(i))
		
		variance=np.var(featurelist)
		variancedict[target]=variance
	sortedvariance= sorted(variancedict.items(), key=operator.itemgetter(1))
	
	top500=sortedvariance[-500:]
	counttop,sumcountertop=makecounter([],top500)
	last500=sortedvariance[:500]
	countlast,sumcounterlast=makecounter([],last500)


	topvariance=counttop.most_common(50)
	leastvariance=countlast.most_common(50)

	topvariancecategories=[x[0].lemma_names()[0] for x in topvariance]
	leastvariancecategories=[x[0].lemma_names()[0] for x in leastvariance]

	categoriesonlytop= list(set(topvariancecategories).difference(leastvariancecategories))
	categoriesonlyleast=list(set(leastvariancecategories).difference(topvariancecategories))
	print('categories with most variance:',categoriesonlytop)
	print('categories with least variance:',categoriesonlyleast)

	# x_val= [x[0].lemma_names()[0] for x in leastvariance]
	# y_val = [x[1] for x in leastvariance]
	# print(x_val)
	# plt.bar(x_val,y_val,color='g')
	# plt.suptitle('least variance most common hypernyms', fontsize=12)
	# plt.xlabel('color')
	# plt.ylabel('frequency')
	# plt.xticks(rotation=20)
	# plt.rc('xtick', labelsize=5) 
	# plt.grid(True)
	# plt.show()







main()