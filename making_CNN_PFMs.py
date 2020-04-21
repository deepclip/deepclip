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

    sqlogos = np.zeros((nfs, fsz / len(VOCAB), len(VOCAB)))
    overall_par = []
    for i, j in enumerate(c1):
        for ii in range(len(j[0])):
            if ii < FS + sFS and ii >= sFS:

                if len(cnsq[i][len(VOCAB) * (ii - sFS):fsz + len(VOCAB) * (ii - sFS)]) == fsz:
                    sqlogos[int(j[0][ii])] += cnsq[i][len(VOCAB) * (ii - sFS):fsz + len(VOCAB) * (ii - sFS)].reshape(
                        (fsz / len(VOCAB), len(VOCAB))) * c2[i][0][ii]**2

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
    print '\n Making PFMs based on convolutional filters'
    t, tt, ttt, tttt = make_cn_seqlogos(argmax, cnscore, inseq, FS, FILTERS, FILTER_SIZES, VOCAB)
    json_obj = {}
    json_obj["logos"] = []
    pfm_add = 0.000000001
    for i in range(len(FILTER_SIZES)):
        for ii in range(FILTERS[i]):
            pfm_name = "PFM_" + str(FILTER_SIZES[i]/4)+ "_" + str(ii)
            lg_raw = (ttt[i][ii].T) / np.sum((ttt[i][ii] + pfm_add).T, axis=0)
            #print lg_raw.shape
            if np.sum(lg_raw) > 0:
                lg_raw[lg_raw < 0] = 0
                lg = lg_raw + pfm_add # offset to prevent log(0) error

                #pfm_R_json_obj[pfm_name] = list([list(k) for k in lg])

                lg_raw = lg_raw.T
                info_per_bp = np.sum(2 + np.sum(lg_raw/np.sum(lg_raw+pfm_add,axis=0) * np.log2(lg_raw/np.sum(lg_raw+pfm_add,axis=0)), axis=-1, keepdims = True))/len(lg_raw)

                json_obj["logos"].append({
                    'size': FILTER_SIZES[i] / len(VOCAB),
                    'filter': ii,
                    "pfm": list([list(k) for k in lg]),
                    'raw-scores': list([list(kk) for kk in ttt[i][ii]]),
                    'points': tttt[i][ii],
                    'points_pct' : tttt[i][ii] / sum(tttt[i]),
                    'info_per_bp' : info_per_bp
                })

                if draw_seq_logos:
                    ddd = printing_seq_logo(lg, tttt[i][ii], name= path + '_filter_'+str(FILTER_SIZES[i] / len(VOCAB))+'bp_number'+str(ii+1)+'_score-')

                print pfm_name,str(info_per_bp)
    with open(pfm_json_outfile, 'w') as f:
        f.write(json.dumps(json_obj))

    print '\n PFMs have been made'
