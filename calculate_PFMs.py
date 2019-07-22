# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 14:16:00 2016

@author: Alexander
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from pylab import rcParams
from matplotlib.ticker import MaxNLocator
import os
import cPickle



"""

plus it look like they have som sort of boks around them. 
this should be deleted


If you are using this function on sequences - set space = 0


if tou are using the zoom function, you need to let it show on the y-axis

"""


def printing_seq_logo(psfm, score, name, zoom = 0):
    with open(os.path.dirname(os.path.abspath(__file__)) + '/let_a.pkl','rb') as ff:
        A = cPickle.load(ff)
    with open(os.path.dirname(os.path.abspath(__file__)) + '/let_c.pkl','rb') as ff:
        C = cPickle.load(ff)
    with open(os.path.dirname(os.path.abspath(__file__)) + '/let_g.pkl','rb') as ff:
        G = cPickle.load(ff)
    with open(os.path.dirname(os.path.abspath(__file__)) + '/let_u.pkl','rb') as ff:
        U = cPickle.load(ff)

    ox,b = psfm_to_bit_matrix(psfm)

    letter_for_print = []

    #b = b[0]/2. # i divide by to bits to get the bits per base per position as a number between 0 and 1
    ox = ox[0]  

    b = psfm
    o = [1] * len(b[0])

    sq = np.zeros((610,84*len(o),4))
    seq_logo = np.zeros((610,84*len(o),4))

    d = np.hsplit(b,len(b[0])) # det bit-matrix is divided into columns



    for i in range(len(o)):
        column_arg = np.argsort(d[i],axis=0) # this sorts te column so the lowest number is first
        column_val = np.sort(d[i],axis=0) 

        for ii,jj in enumerate(column_arg):
            if jj[0] == 0:
                seq_logo += creating_letter(d[i][jj[0]][0],'a',A,i,o[i]-np.sum(column_val[ii:],axis=0)[0],o,b)[::-1]

            if jj[0] == 1:
                seq_logo += creating_letter(d[i][jj[0]][0],'c',C,i,o[i]-np.sum(column_val[ii:],axis=0)[0],o,b)[::-1]
            if jj[0] == 2: 
                seq_logo += creating_letter(d[i][jj[0]][0],'g',G,i,o[i]-np.sum(column_val[ii:],axis=0)[0],o,b)[::-1]
            if jj[0] == 3: 
                seq_logo += creating_letter(d[i][jj[0]][0],'u',U,i,o[i]-np.sum(column_val[ii:],axis=0)[0],o,b)[::-1]

    seq_logo = seq_logo[zoom:,:]

    #plt.subplots(figsize=(20,10))
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel('Position')
    plt.ylabel('Pseudo-Bits')
    plt.title('Pseudo-Sequence logo')
    plt.imshow(seq_logo,extent=[0,len(o),0,2])
    plt.savefig('{}{}_scorePbase_{}_totInfo_{}_infoPbp_{}_infoXScore_{}_infoPbpXScore_{}_infoPbpXScoreXconvscorePbase_{}_seqlogo.png'.format(name,score,score/len(b[0]),np.sum(np.array(ox)),np.sum(np.array(ox))/len(b[0]),score*np.sum(np.array(ox)), score*(np.sum(np.array(ox))/len(b[0])), (score/len(b[0]))*(np.sum(np.array(ox))/len(b[0])) ))
    plt.close()

    return seq_logo



def creating_letter(importance,let,letter,base_pos,space,o,b):
    
    """
    importance: no between 0 and 1 - the bits pr base at a give position is 
    calculated and is divided with 2
    
    let: name of the letter. Makes it easier to see, and serves minor function
    
    letter: the arrays describing the letters
    
    base_pos: a number that locates the bases at the correct position
    
    space: describes the space below every base. They are stacked in a seg-logo
    with the most important letter at the top.
    
    """    
    
    x_axis = len(letter[0])
    new_height = int(np.round(importance*len(letter))) # calculating heigt based on importance
    space = int(np.round(space*len(letter)))

    seq_logo = np.zeros((len(letter),x_axis*len(o),4))

    new_let = letter # making a letter size changeing

    while new_height < int(np.round(len(new_let)*0.90)):
        new_let = np.delete(new_let,range(0,len(new_let),10),axis=0)
        

    if len(new_let)-new_height != 0:
        if let == 'a':
            new_let = np.delete(new_let,range(len(new_let)-(len(new_let)-new_height),len(new_let)),axis=0)
        else:
            new_let = np.delete(new_let,range(len(new_let)/2,len(new_let)/2+len(new_let)-new_height),axis=0)

    if let == 'g' and 5 < new_height < 50:
        new_let[0:5] = letter[95:100]
    
    if let == 'c' and 5 < new_height < 50:
        new_let[0:5] = letter[70:75]


    if new_height < len(new_let):
        print 'this sequence logo is very incorrect - too large'
    if new_height > len(new_let):
        print 'this sequence logo is very incorrect - too small'


    if space > 0:
        new_let = np.append(new_let,np.zeros((space,x_axis,4)),axis=0)

    if base_pos > 0:
        air = np.zeros((len(new_let),x_axis*base_pos,4))
        new_let = np.append(air,new_let,axis=1)

    new_let = np.append(new_let,np.zeros((len(new_let),x_axis*(len(o)-base_pos-1),4)),axis=1)

    seq_logo[:new_let.shape[0]] += new_let[::-1]


    return seq_logo
    





def psfm_to_bit_matrix(A1):
    """
    
    """
    
    overall_info = []
    x2h = np.hsplit(A1,len(A1[0]))
    temp = []
    for ii in range(len(x2h)):  # this gives the overall_info for one prob column
        x3 = np.log2(4)

	print x2h[ii], np.sum(x2h[ii])
        for iii in range(len(x2h[0])):
            if x2h[ii][iii] == 0:
                x3 += 0
            else:
                x3 += -(-x2h[ii][iii]*np.log2(x2h[ii][iii])) 
	    
	if np.sum(x2h[ii]) <= 4e-09:
	    x3 = 0
	    x2h[ii] += 0.25
        temp.append(x3)
    temp = np.hstack(temp)
    overall_info.append(temp)
    
    bits_per_base = []
    temp2 = []
    for ii in range(len(overall_info[0])):        # this givs the bits per base at a 
        temp2.append(overall_info[0][ii]*x2h[ii]) # given position
    temp2 = np.hstack(temp2)
    bits_per_base.append(temp2)
    
    return overall_info, bits_per_base



