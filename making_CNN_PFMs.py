from __future__ import print_function
import numpy as np
import json
from calculate_PFMs import printing_seq_logo

def find_cnpar_and_cnsqlogo(c1, c2, cnsq, sFS, FS, nfs, fsz, VOCAB, mn):
    """
    This functions find sequence logos given the convolutional score of the 
    input sequences to the network.
    Find a sequence logo for every filter used in the run.
    """

    # c1 = sequences containig argmax of convolutions
    # c2 = sequences containing cn_scores
    # sFS = sum of feature maps up til the one given
    # FS = Feature maps from the given layer
    # nfs = number of filters in the layer
    # fsz = the size of the filter in the layer
    # VOCAB = the vocabulary

    count = np.zeros((nfs))
    fsz = int(fsz)
    nfs = int(nfs)

    sqlogos = np.zeros((nfs, (fsz // len(VOCAB)), len(VOCAB)))
    overall_par = []
    for i, j in enumerate(c1):
        for ii in range(len(j[0])):
            if ii < FS + sFS and ii >= sFS:

                if len(cnsq[i][int(len(VOCAB) * (ii - sFS)):int(fsz + len(VOCAB) * (ii - sFS))]) == fsz:
                    sqlogos[int(j[0][ii])] += cnsq[i][int(len(VOCAB) * (ii - sFS)):int(fsz + len(VOCAB) * (ii - sFS))].reshape(
                        (int(fsz) // int(len(VOCAB)), int(len(VOCAB)))) * c2[i][0][ii]**2

                    if j[0][ii] in overall_par:
                        count[int(j[0][ii])] += 1

                    if j[0][ii] not in overall_par:
                        overall_par.append(j[0][ii])
                        count[int(j[0][ii])] += 1
    return overall_par, count, sqlogos


def make_cn_seqlogos(argmax, cnscore, inseq, FS, FILTERS, FILTER_SIZES, VOCAB):
    # t = A list with index numbers of the used filters
    # tt = A list with the times the filters in t where applied in the run.
    # ttt = A list with the seuquene logos of the different applied filters.

    t = []
    tt = []
    ttt = []

    for i in range(len(FS)):
        x, z, q = find_cnpar_and_cnsqlogo(argmax, cnscore, inseq, sum(FS[0:i]), FS[i], FILTERS[i], FILTER_SIZES[i],
                                          VOCAB, i)
        t.append(x)
        tt.append(z)
        ttt.append(q)

    sqlg_score = []
    for i in ttt:
        temp = []
        for ii in i:
            temp.append(np.sum(ii))
        sqlg_score.append(temp)

    return t, tt, ttt, sqlg_score


def convolutional_logos(argmax, cnscore, inseq, FS, FILTERS, FILTER_SIZES, VOCAB, pfm_json_outfile, draw_seq_logos):
    print('\n Making PFMs based on convolutional filters')
    t, tt, ttt, tttt = make_cn_seqlogos(argmax, cnscore, inseq, FS, FILTERS, FILTER_SIZES, VOCAB)
    json_obj = {}
    json_obj["logos"] = []
    pfm_add = 0.000000001
    print(" Filter\tLength\tName\tInformation-score-per-bp")
    filter_num = 0
    for i in range(len(FILTER_SIZES)):
        for ii in range(FILTERS[i]):
            filter_num += 1
            pfm_name = "PFM_" + str(int(FILTER_SIZES[i]/len(VOCAB)))+ "_" + str(ii)
            lg_raw = (ttt[i][ii].T) / np.sum((ttt[i][ii] + pfm_add).T, axis=0)
            #print("lg_raw: {}".format(lg_raw))
            if np.sum(lg_raw) > 0:
                lg_raw[lg_raw < 0] = 0
                lg = lg_raw + pfm_add # offset to prevent log(0) error

                #pfm_R_json_obj[pfm_name] = list([list(k) for k in lg])

                lg_raw = lg_raw.T
                pfm = lg.T
                info_per_bp = np.sum(np.sum(np.log2(((pfm.T)/np.sum(pfm,axis=1)).T)*((pfm.T)/np.sum(pfm,axis=1)).T, axis=1) + np.log2(len(pfm.T)))/len(pfm)

                json_obj["logos"].append({
                    'size': int(FILTER_SIZES[i] / len(VOCAB)),
                    'filter': ii,
                    "pfm": list([list(k) for k in lg]),
                    'raw-scores': list([list(kk) for kk in ttt[i][ii]]),
                    'points': tttt[i][ii],
                    'points_pct' : tttt[i][ii] / sum(tttt[i]),
                    'info_per_bp' : info_per_bp
                })

                if draw_seq_logos:
                    ddd = printing_seq_logo(lg, tttt[i][ii], name= path + '_filter_'+str(int(FILTER_SIZES[i] / len(VOCAB)))+'bp_number'+str(ii+1)+'_score-')

                print(" #{}\t{}\t{}\t{:.6f}".format(filter_num,int(FILTER_SIZES[i]/len(VOCAB)),pfm_name,info_per_bp))
    with open(pfm_json_outfile, 'w') as f:
        f.write(json.dumps(json_obj))

    #print '\n PFMs have been made'
