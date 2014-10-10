#import matplotlib.pyplot as plt
import numpy as np

def removeContinuum(spectra):
	d={}
	x1=spectra.shape[0]
	
	# handle zero endpoint
	spectra[1]=spectra[1]+1e-6
	spectra[0]=spectra[1]
	
	# initalize envelope at endpoints
	d[x1-1]=spectra[x1-1]
	d[0]=spectra[1]
	
	# calculates the pivot points need to envelope spectra
	# not handling right hand side of max correctly
	x=[i for i in range(spectra.shape[0])]
	diff=spectra
	npivot=0
	while sum(np.array(diff)>0):
		m=max(diff)
		xm=list(diff).index(m)
		d[xm]=m
		y=d.keys()
		y=list(set(y)) # removes dupes
		y.sort()
		z=[]
		for i in range(len(y)-1):
			z=z+[spectra[y[i]]+j*(spectra[y[i+1]]-spectra[y[i]])/(y[i+1]-y[i]) for j in range(y[i+1]-y[i])]
		
		z= [z[0]] + z # not sure why z ends up 1 element short, but this "fixes" 
		diff=spectra-z
		sum(np.array(diff)>0)
		if len(y) == npivot:
			break
		else:
			npivot=len(y)
	
	# calcuates envelope from pivot points
	y=d.keys()
	#d[0]=spectra[0]
	#d[x1-1]=spectra[x1-1]
	#y=y+[0]
	#y=y+[x1-1]
	y=list(set(y)) # removes dupes
	y.sort()
	z=[]
	for i in range(len(y)-1):
		z=z+[spectra[y[i]]+j*(spectra[y[i+1]]-spectra[y[i]])/(y[i+1]-y[i]) for j in range(y[i+1]-y[i])]
	
	z=[z[0]]+z  # not sure why z ends up 1 element short, but this "fixes" 
	
	#plt.plot(spectra)
	#plt.plot(z)
	#plt.show()
	
	# remaining outliers?
	zz=spectra/z
	zz[zz>1.5]=1.0
	
	#plt.plot(zz)
	#plt.show()
	return(zz)

