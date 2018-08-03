import numpy as np
import operator
import nltk
from collections import Counter
from nltk.corpus import wordnet as wn
from collections import Iterable
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

import math
from PIL import Image,ImageDraw

from PIL import Image, ImageDraw
import colorsys

from webcolors import name_to_rgb

def make_rainbow_rgb(colors, width, height):
    """colors is an array of RGB tuples, with values between 0 and 255"""

    img = Image.new("RGBA", (width, height))
    canvas = ImageDraw.Draw(img)

    def hsl(x):
        to_float = lambda x : x / 255.0
        (r, g, b) = map(to_float, x)
        h, s, l = colorsys.rgb_to_hsv(r,g,b)
        h = h if 0 < h else 1 # 0 -> 1
        return h, s, l

    rainbow = sorted(colors, key=hsl)

    dx = width / float(len(colors)) 
    x = 0
    y = height / 2.0
    for rgb in rainbow:
        canvas.line((x, y, x + dx, y), width=height, fill=rgb)
        x += dx
    #img.show() 
    return img

def main():
	f=open('../reducedcolorsdm.txt','r')
	colors=['green', 'blue', 'purple', 'pink', 'brown', 'red', 'yellow', 'grey', 'teal', 'gray', 'violet', 'magenta', 'turquoise', 'cyan', 'white', 'beige', 'lilac', 'mauve', 'indigo', 'black']
	colorsbruni=open('colorsbruni2012.txt','r')
	featuredict={}
	for line in f: 
		featurelist=[]
		line=line.split('\t')
		target=line[0]
		features=line[1].strip('\n')
		features=features.split(' ')
		for i in features: 
			featurelist.append(float(i))
		featuredict[target]=featurelist

	nouns=[]
	for line in colorsbruni:
		line=line.split('\t')
		nouns.append(line[0])

	finaldictcolors={}
	for i in nouns:
		colordict={}
		vectors=[]
		vectors.append(featuredict[i])
		mean=np.mean(np.array(vectors),axis=0)
		summean=sum(mean)
		count=0
		for c in colors: 
			if mean[count]==0:
				colordict[c]=0
				count+=1
			else:
				colordict[c]=int((mean[count]/summean)*100)
				count+=1

		finaldictcolors[i]=colordict
		print(i, sorted(Counter(colordict).most_common(2)))

	for key,value in finaldictcolors.iteritems():
			#print(value)
			pixels=[]
			name=key
			
			for key,value in finaldictcolors[key].iteritems():
				for i in range(value):
					
					if key=='color':
						pass
					elif key=='drab':
						pixels.append((50, 113, 23))
					elif key=='mint':
						pixels.append((152,255,152))
					elif key=='umber':
						pixels.append((	99, 81, 71))
					elif key=='sky':
						pixels.append((135,206,250))
					elif key=='rose':
						pixels.append(((255, 0, 128)))
					elif key=='fuschia':
						pixels.append((202, 44, 146))
					elif key=='peach':
						pixels.append((255,218,185))
					elif key=='mustard':
						pixels.append((255, 219, 88))
					elif key=='lilac':
						pixels.append((200,162,200))
					elif key=='mauve':
						pixels.append((224, 176, 255))
					elif key=='periwinkle':
						pixels.append((195,205,230))
					else:
						pixels.append(name_to_rgb(key))

			size=int(math.sqrt(len(pixels)))+1
			width=size*15
			
			image=make_rainbow_rgb(pixels, width, size)


			try:
				image.save('visualizations/'+name+'.png')
			except UnicodeDecodeError:
				pass


main()