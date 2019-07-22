#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("-seqs",
                        required=True,
                        type=str,
                        default=None,
                        help="File containing sequences")
args = parser.parse_args()


dicts = []
with open(args.seqs,'r') as inf:
    for line in inf:
        dicts.append(eval(line))


try:
    assert len(dicts) == 5
except AssertionError:
    print "AssertionError: assert len(dicts) == 5"
    print "len(dicts) =",str(len(dicts))
    print str(dicts)
    exit(1)

sizes = [4,5,6,7,8]
tot_info = []

for i in range(len(dicts)):
    temp_pfm = np.array(dicts[i]['PFM_'+str(sizes[i])+'_0']).T
    #print temp_pfm.shape

    temp_pfm1 = temp_pfm + 0.000000001
    tot_info.append([np.sum(np.log2(4) + np.sum(temp_pfm * np.log2(temp_pfm1), axis=1, keepdims = True))/len(temp_pfm), sizes[i]])


#print tot_info
tot_info.sort()
tot_info.reverse()
#print tot_info

new_json = {}
for mp in range(len(tot_info)):
    for qq in range(len(dicts)):
	if 'PFM_'+str(tot_info[mp][1])+'_0' in dicts[qq]:
	    print 'PFM_'+str(tot_info[mp][1])+'_0'
	    new_json['PFM_'+str(tot_info[mp][1])+'_0'] = dicts[qq]['PFM_'+str(tot_info[mp][1])+'_0']




with open('PFMs.json', 'w') as outfile:
    json.dump(new_json, outfile)

with open('PFM_order.txt', 'w') as txt:
    print >> txt, "name, info_per_bp"
    for i in tot_info:
    	print >> txt, "PFM_" +str(i[1])+ "_0,", "{0:.3f}".format(i[0])

