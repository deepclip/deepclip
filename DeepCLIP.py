#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
import json
import random
import math
import string
import time
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import constants
import network
from data_gen import onehot_encode
from data_gen import onehot_binary

sys.setrecursionlimit(500000)  # we sometimes run into Python's default limit of 999. Note: this can cause a crash!


def parse_arguments():
    version = "1.0.0"
    description = "Constructs a neural network to identify protein binding motifs."
    epilog = "Protein binding sites should be supplied as FASTA files.\n\nAuthors:\n\tAlexander Gr0nning <agroen@imada.sdu.dk>,\n\tThomas Koed Doktor <thomaskd@bmb.sdu.dk>"

    parser = argparse.ArgumentParser(description=description, version=version, epilog=epilog, formatter_class=argparse.RawDescriptionHelpFormatter)
    required = parser.add_argument_group('required arguments')

    parser.add_argument("--runmode",
                        required=True,
                        choices=["train", "predict", "predict_long", "cv"],
                        type=str,
                        default="train",
                        help='Operation to perform. Use \'train\' to train a neural network. Use \'predict\' to predict binding affinity for a set of sequences.')

    parser.add_argument('-n', "--network_file",
                        required=False,
                        type=str,
                        default=None,
                        help="Path to network parameters file")

    parser.add_argument('-P', "--predict_function_file",
                        required=False,
                        type=str,
                        default=None,
                        help="Path to prediction function file")

    parser.add_argument("--sequences",
                        required=False,
                        type=str,
                        default=None,
                        help="FASTA file containing binding sequences.")

    parser.add_argument("--background_sequences",
                        required=False,
                        type=str,
                        default=None,
                        help="FASTA file containing background (non-binding) sequences.")

    parser.add_argument("--write_sequences",
                        required=False,
                        type=str,
                        default=None,
                        help="Write sequences used for training to FASTA file.")

    parser.add_argument("--write_background_sequences",
                        required=False,
                        type=str,
                        default=None,
                        help="Write background sequences used for training to FASTA file.")

    parser.add_argument("--variant_sequences",
                        required=False,
                        type=str,
                        default=None,
                        help="FASTA file containing variant sequences to be predicted and compared to the reference sequence(s).")

    parser.add_argument("--force_bed",
                        action='store_true',
                        help="Force sequence file to considered a BED file regardless of file ending.")

    parser.add_argument("--background_shuffle",
                        action="store_true",
                        help="Produce background sequences by shuffling input sequences"
                        )

    parser.add_argument("--genome_file",
                        required=False,
                        type=str,
                        default=None,
                        help="FASTA file for extracting sequences from BED file.")

    parser.add_argument("--gtf_file",
                        required=False,
                        type=str,
                        default=None,
                        help="GTF file for extrating sequences from BED file.")

    parser.add_argument("--min_length",
                        required=False,
                        type=int,
                        default=1,
                        help="Minimum sequence length.")

    parser.add_argument("--max_length",
                        required=False,
                        type=int,
                        default=400,
                        help="Maximum sequence length.")

    parser.add_argument("--bed_width",
                        required=False,
                        type=int,
                        default=0,
                        help="Fixed sequences from BED file to this length."
                        )

    parser.add_argument("--bed_padding",
                        required=False,
                        type=int,
                        default=0,
                        help="Pad sequences from BED file with this number of bases.")

    parser.add_argument('-p', "--padding",
                        required=False,
                        type=int,
                        default=None,
                        help="If provided, the padding of intervals around the center")

    parser.add_argument("--data_split",
                        required=False,
                        nargs='+',
                        type=float,
                        default=[0.8,0.1,0.1],
                        help="Ratios to split sequence data into [train, validation, test]")

    parser.add_argument("--batch_size",
                        required=False,
                        type=int,
                        default=constants.MINI_BATCH_SIZE,
                        help="The batch size used in training")

    parser.add_argument("--lstm_layers",
                        required=False,
                        type=int,
                        default=constants.LSTM_LAYERS,
                        help="The number of bidirectional LSTM layers. The final number of LSTM layers will be twice this number.")

    parser.add_argument("--lstm_nodes",
                        required=False,
                        type=int,
                        default=constants.LSTM_NODES,
                        help="The number of nodes in each LSTM layer")

    parser.add_argument("--lstm_dropout",
                        required=False,
                        type=float,
                        default=constants.LSTM_DROPOUT,
                        help="LSTM dropout ratio")

    parser.add_argument("--dropout_in",
                        required=False,
                        type=float,
                        default=constants.DROPOUT_IN,
                        help="Input layer dropout ratio")

    parser.add_argument("--dropout_out",
                        required=False,
                        type=float,
                        default=constants.DROPOUT_OUT,
                        help="Output layer dropout ratio")

    parser.add_argument('-e', "--num_epochs",
                        required=False,
                        type=int,
                        default=10,
                        help="Number of training epochs")

    parser.add_argument("--early_stopping",
                        required=False,
                        type=int,
                        default=10,
                        help="If provided, if this many training epochs in a row fail to improve the best AUROC, the training is stopped.")

    parser.add_argument("--num_filters",
                        required=False,
                        type=int,
                        default=constants.NUM_FILTERS,
                        help="Number of filters per convolution")

    parser.add_argument("--filter_sizes",
                        required=False,
                        nargs='+',
                        type=int,
                        default=constants.FILTER_SIZES,
                        help="Convolutional filter sizes in bp.")

    parser.add_argument("--learning_rate",
                        required=False,
                        type=float,
                        default=constants.LEARNING_RATE,
                        help="Learning rate.")

    parser.add_argument("--l2",
                        required=False,
                        type=float,
                        default=constants.L2,
                        help="L2.")

    parser.add_argument("--random_seed",
                        required=False,
                        type=int,
                        default=1234,
                        help="Set random seed")

    parser.add_argument("--test_output_file",
                        required=False,
                        type=str,
                        default="",
                        help="Write results of test to JSON file.")

    parser.add_argument("--test_predictions_file",
                        required=False,
                        type=str,
                        default="",
                        help="Write prediction results of test sequences to file.")

    parser.add_argument("--predict_output_file",
                        required=False,
                        type=str,
                        default="",
                        help="Write results of prediction to JSON file.")

    parser.add_argument("--predict_PFM_file",
                        required=False,
                        type=str,
                        default="",
                        help="Write sequence logo data to JSON file.")

    parser.add_argument("--PFM_on_half",
                        required=False,
                        type=bool,
                        default=False,
                        help="Write sequence logo data to JSON file based top 50 % sequences with highest prediction.")

    parser.add_argument("--balanced_input",
                        action='store_true',
                        help="Force the number of negative and positive sequences to be equal.")

    parser.add_argument("--auc_thr",
                        required=False,
                        type=float,
                        default=0.999,
                        help="Will stop a run if 2 epochs produced either training or validation AUROCs above the float value")

    parser.add_argument("--make_diff",
                        action='store_true',
                        help="Prints the delta binding profiles.")

    parser.add_argument("--sigmoid_profile",
                        action='store_true',
                        help="Plots the sigmoidal profile values instead of 'raw' values.")

    parser.add_argument("--draw_seq_logos",
                        action="store_true",
                        help="Draw convolutional sequence logos.")

    parser.add_argument("--draw_profiles",
                        action="store_true",
                        help="Draw binding profiles.")

    parser.add_argument("--performance_selection",
                        required=False,
                        choices=["auroc", "loss"],
                        type=str,
                        default="auroc",
                        help='Use this measure to select the best performing model [auroc, loss]. Default: auc')
    parser.add_argument("--export_test_sets",
                        action='store_true',
                        help="Enables exporting of test datasets when running in CV runmode. Any 'n' bases in the input are removed.")

    parser.add_argument("--no_extra_padding",
                        required=False,
                        default=False,
                        action="store_true",
                        help="No sequence padding will occur.")

    return parser.parse_args()

def make_filename_safe(filename):
    """" Make sure the filename can be copied between POSIX systems and Windows, and it's url-safe"""
    allowed_length = 255 # windows doesn't support more than 255 character filenames
    allowed_chars = string.ascii_letters + string.digits + "~ -_.()"
    safe_filename = ''.join(c for c in filename if c in allowed_chars)
    return safe_filename[:allowed_length]

def make_profiles(srrap, rapsq, seq_ids_list, rang, name):

    color = ['black','red','blue','green']
    name1 = ['WT','MUT', 'WT-C', 'MUT-C']

    for i in range(0,len(rapsq),2):
        ys = []
        if rang[i/2] == rang[-1]:
            tmp = len(srrap)-rang[-1]
        else:
            tmp = rang[i/2+1]-rang[i/2]

        for ii in range(2):
            tmp = []
            y = list(srrap[i+ii])
            for kk in range(len(rapsq[i].lower())):
                if rapsq[i].lower()[kk] != 'n':
                    tmp.append(y[kk])
                    print rapsq[i].lower()[kk],
            print ''
            ys.append(tmp)


        print len(ys[0])
        x = np.array(range(len(ys[0])))

        id = seq_ids_list[i]
        mt = rapsq[i+1].upper().replace('N','')
        mt = list(mt)
        #plt.ylim(0,1)
        plt.xlim(0,len(x)-1)
        plt.yticks(size=18)
        plt.xticks(x,mt,size=18)
        for ii in range(len(ys)):
            plt.plot(x,ys[ii], color[ii], label=name1[ii],linewidth=6)

        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        ax.grid('on')
        ax.set_ylabel('DeepCLIP score', size=18)

        plt.title(id, size=30)
        filename = make_filename_safe(name + '-' + str(id) + '_' + str(id2)+'_profile' + '.png')
        plt.savefig(filename, bbox_inches='tight')
        #plt.savefig(name+str(i)+'_profile' + '.png')

        plt.close()

def sigmoid(x, derivative=False):
    return x*(1-x) if derivative else 1/(1+np.exp(-x))

def make_profiles2(srrap, rapsq, seq_ids_list, rang, name, make_diff, sigmoid_profile):

    if make_diff == True:
        wt = 0
        wt += srrap[0]

        print len(srrap), 'GGGGGGGGGGGGGGGGGGGG'
        for qq in range(len(srrap)):
            srrap[qq] = srrap[qq] - wt


    color = ['black','red','blue','green']
    name1 = ['WT','MUT', 'MUT2', 'MUT3']

    print " Making weight profiles . . .",
    for i in range(0,len(rapsq),2):
        print '.',
        sys.stdout.flush()

        # calculated total profile scores
        score1 = 0
        score2 = 0
        difference = 0
        for j in range(len(srrap[i])):
            score1 += srrap[i][j]
            score2 += srrap[i+1][j]
            difference += (srrap[i+1][j] - srrap[i][j])
            if sigmoid_profile:
                srrap[i][j] = sigmoid(srrap[i][j])
                srrap[i+1][j] = sigmoid(srrap[i+1][j])

        ys = []
        if rang[i/2] == rang[-1]:
            tmp = len(srrap)-rang[-1]
        else:
            tmp = rang[i/2+1]-rang[i/2]

        for ii in range(2):
            tmp = []
            y = list(srrap[i+ii])
            for kk in range(len(rapsq[i].lower())):
                if rapsq[i].lower()[kk] != 'n':
                    tmp.append(y[kk])
                    #print rapsq[i].lower()[kk],
            #print ''
            ys.append(tmp)

        #print len(ys[0]), 'FFFFFFFFFFFF'
        x = np.array(range(len(ys[0])))

        # IF MCAD
        id = seq_ids_list[i]
        id2 = seq_ids_list[i+1]
        mt1 = rapsq[i+1].upper().strip('N')
        mt1 = list(mt1)
        mt2 = rapsq[i].upper().strip('N')
        mt2 = list(mt2)

        if len(mt1) != len(mt2):
            continue

        for qq in range(len(mt1)):
            if mt1[qq] == mt2[qq]:
                mt1[qq] = ''

        # Try to scale size of the plot according to length of the sequences
        scaling_factor = int((len(mt1)-1)/7)
        fig = plt.figure(figsize=(2*scaling_factor+9, scaling_factor+5))
        plt.style.use('seaborn-white')
        ax1 = plt.axes(frame_on=False)  # standard axes
        ## IF MCAD:
        ##ax1 = plt.axes([0.127, 0.13, 3, 2.9], frame_on=False)

        #plt.plot(x, color='w')
        for ii in range(len(ys)):
            plt.plot(x,ys[ii], color='w')
        ax1.axes.get_yaxis().set_visible(False)
        ax1.tick_params(axis='x', which='major', pad=scaling_factor+5)
        plt.xticks(x, mt1)
        plt.xticks(color='red')
        plt.xlim(0, len(x)-1)
        #plt.ylim(0, 1)
        plt.yticks(size=20)
        plt.xticks(x,mt1,size=25)
        #plt.style.available: ['seaborn-darkgrid', 'Solarize_Light2', 'seaborn-notebook', 'classic', 'seaborn-ticks', 'grayscale', 'bmh', 'seaborn-talk', 'dark_background', 'ggplot', 'fivethirtyeight', '_classic_test', 'seaborn-colorblind', 'seaborn-deep', 'seaborn-whitegrid', 'seaborn-bright', 'seaborn-poster', 'seaborn-muted', 'seaborn-paper', 'seaborn-white', 'fast', 'seaborn-pastel', 'seaborn-dark', 'tableau-colorblind10', 'seaborn', 'seaborn-dark-palette']
        #plt.style.use('seaborn-whitegrid')
        #ax2 = plt.axes([0.125, 0.17, 0.7715, 0.8])
        ## IF MCAD 75:
        ax2 = plt.axes([0.126, 0.1+0.025/(math.log(scaling_factor+1,10)*0.9), 0.7715, 0.8]) #plt.axes((left, bottom, width, height))
        ax2.grid(True)
        ax2.set_ylabel('DeepCLIP score', size=min(35,30+scaling_factor))
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.axhline(linewidth=4)


        for ii in range(len(ys)):
            plt.plot(x,ys[ii], color[ii], label=name1[ii],linewidth=10)
        plt.xticks(x, mt2)
        plt.xlim(0, len(x)-1)
        #plt.ylim(0, 1)
        plt.yticks(size=30)
        plt.xticks(x,mt2,size=25)

        #print name+str(i)+'TESTT'
        title = "{}>{} Score: {:.3f} -> {:.3f} (Difference: {:.3f}).".format(id, id2, score1, score2, difference)
        #plt.title(title, size=2*scaling_factor)
        filename = make_filename_safe(name + '-' + str(id) + '_' + str(id2)+'_profile' + '.png')
        print "\n Saving", filename, "with scaling_factor =", str(scaling_factor)
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

def make_paired_profile(wt_scores, var_scores, wt_seq, var_seq, wt_name, var_name, make_diff, sigmoid_profile, output_name, color = ['black','red'], legend_names = ['WT','MUT']):

    if make_diff == True:
        for base_index in range(len(wt_scores)):
            var_scores[base_index] = var_scores[base_index] - wt_scores[base_index]

    # calculated total profile scores
    score1 = 0
    score2 = 0
    difference = 0
    for j in range(len(wt_scores)):
        score1 += wt_scores[j]
        score2 += var_scores[j]
        difference += score2 - score1
        if sigmoid_profile:
            wt_scores[j] = sigmoid(wt_scores[j])
            var_scores[j] = sigmoid(var_scores[j])


    # remove any leading and trailing 'n' in the sequences, and their associated profile scores
    y_scores = []
    start_index = 0
    end_index = len(wt_seq)-1
    for i in range(start_index, end_index):
        if wt_seq[i].lower() == 'n':
            start_index = i+1
        elif wt_seq[i].lower() != 'n':
            break
    for i in range(end_index, start_index, -1):
        if wt_seq[i].lower() == 'n':
            end_index = i
        else:
            break
    seq1 = list(wt_seq[start_index:end_index].upper().strip('N'))
    y_scores.append(list(wt_scores[start_index:end_index]))

    start_index = 0
    end_index = len(var_seq)-1
    for i in range(start_index, end_index):
        if var_seq[i].lower() == 'n':
            start_index = i+1
        else:
            break
    for i in range(end_index, start_index, -1):
        if var_seq[i].lower() == 'n':
            end_index = i
        else:
            break
    seq2 = list(var_seq[start_index:end_index].upper().strip('N'))
    y_scores.append(list(var_scores[start_index:end_index]))

    if len(seq1) != len(seq2):
        return 0
    for base_index in range(len(seq2)):
        if seq2[base_index] == seq1[base_index]:
            seq2[base_index] = ''

    x = np.array(range(len(seq1)))

    id = wt_name
    id2 = var_name

    # Try to scale size of the plot according to length of the sequences
    scaling_factor = int((len(seq2)-1)/7)
    fig = plt.figure(figsize=(2*scaling_factor+9, scaling_factor+5))
    plt.style.use('seaborn-white')
    ax1 = plt.axes(frame_on=False)  # standard axes
    ## IF MCAD:
    ##ax1 = plt.axes([0.127, 0.13, 3, 2.9], frame_on=False)

    #plt.plot(x, color='w')
    for ii in range(len(y_scores)):
        plt.plot(x,y_scores[ii], color='w')
    ax1.axes.get_yaxis().set_visible(False)
    ax1.tick_params(axis='x', which='major', pad=scaling_factor+5)
    plt.xticks(x, seq2)
    plt.xticks(color='red')
    plt.xlim(0, len(x)-1)
    #plt.ylim(0, 1)
    plt.yticks(size=20)
    plt.xticks(x,seq2,size=25)
    #plt.style.available: ['seaborn-darkgrid', 'Solarize_Light2', 'seaborn-notebook', 'classic', 'seaborn-ticks', 'grayscale', 'bmh', 'seaborn-talk', 'dark_background', 'ggplot', 'fivethirtyeight', '_classic_test', 'seaborn-colorblind', 'seaborn-deep', 'seaborn-whitegrid', 'seaborn-bright', 'seaborn-poster', 'seaborn-muted', 'seaborn-paper', 'seaborn-white', 'fast', 'seaborn-pastel', 'seaborn-dark', 'tableau-colorblind10', 'seaborn', 'seaborn-dark-palette']
    #plt.style.use('seaborn-whitegrid')
    #ax2 = plt.axes([0.125, 0.17, 0.7715, 0.8])
    ## IF MCAD 75:
    ax2 = plt.axes([0.126, 0.1+0.025/(math.log(scaling_factor+1,10)*0.9), 0.7715, 0.8]) #plt.axes((left, bottom, width, height))
    ax2.grid(True)
    ax2.set_ylabel('DeepCLIP score', size=min(35,30+scaling_factor))
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.axhline(linewidth=4)

    for i in range(len(y_scores)):
        plt.plot(x,y_scores[i], color=color[i], label=legend_names[i], linewidth=10)
    plt.xticks(x, seq1)
    plt.xlim(0, len(x)-1)
    #plt.ylim(0, 1)
    plt.yticks(size=30)
    plt.xticks(x,seq1,size=25)

    #print name+str(i)+'TESTT'
    title = "{}>{} Score: {:.3f} -> {:.3f} (Difference: {:.3f}).".format(id, id2, score1, score2, difference)
    #plt.title(title, size=2*scaling_factor)
    filename = make_filename_safe(output_name + '-' + str(id) + '_' + str(id2)+'_profile' + '.png')
    print "\n Saving", filename, "with scaling_factor =", str(scaling_factor)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def make_single_profile(scores, seq, name, sigmoid_profile, output_name, color = 'black', legend_name = 'WT'):

    # calculated total profile score
    score1 = 0
    for j in range(len(scores)):
        score1 += scores[j]
        if sigmoid_profile:
            scores[j] = sigmoid(scores[j])


    # remove any leading and trailing 'n' in the sequences, and their associated profile scores
    start_index = 0
    end_index = len(seq)-1
    for i in range(start_index, end_index):
        if seq[i].lower() == 'n':
            start_index = i+1
        else:
            break
    for i in range(end_index, start_index, -1):
        if seq[i].lower() == 'n':
            end_index = i
        else:
            break
    seq1 = list(seq[start_index:end_index].upper().strip('N'))
    y_scores = list(scores[start_index:end_index])

    x = np.array(range(len(seq1)))

    id = name

    # Try to scale size of the plot according to length of the sequences
    scaling_factor = int((len(seq1)-1)/7)
    fig = plt.figure(figsize=(2*scaling_factor+9, scaling_factor+5))
    plt.style.use('seaborn-white')
    ax1 = plt.axes(frame_on=False)  # standard axes
    ## IF MCAD:
    ##ax1 = plt.axes([0.127, 0.13, 3, 2.9], frame_on=False)

    #plt.plot(x, color='w')
    plt.plot(x,y_scores, color='w')
    ax1.axes.get_yaxis().set_visible(False)
    ax1.tick_params(axis='x', which='major', pad=scaling_factor+5)
    plt.xlim(0, len(x)-1)
    #plt.ylim(0, 1)
    plt.yticks(size=20)
    #plt.style.available: ['seaborn-darkgrid', 'Solarize_Light2', 'seaborn-notebook', 'classic', 'seaborn-ticks', 'grayscale', 'bmh', 'seaborn-talk', 'dark_background', 'ggplot', 'fivethirtyeight', '_classic_test', 'seaborn-colorblind', 'seaborn-deep', 'seaborn-whitegrid', 'seaborn-bright', 'seaborn-poster', 'seaborn-muted', 'seaborn-paper', 'seaborn-white', 'fast', 'seaborn-pastel', 'seaborn-dark', 'tableau-colorblind10', 'seaborn', 'seaborn-dark-palette']
    #plt.style.use('seaborn-whitegrid')
    #ax2 = plt.axes([0.125, 0.17, 0.7715, 0.8])
    ## IF MCAD 75:
    ax2 = plt.axes([0.126, 0.1+0.025/(math.log(scaling_factor+1,10)*0.9), 0.7715, 0.8]) #plt.axes((left, bottom, width, height))
    ax2.grid(True)
    ax2.set_ylabel('DeepCLIP score', size=min(35,30+scaling_factor))
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.axhline(linewidth=4)

    plt.plot(x,y_scores, color=color, label=legend_name, linewidth=10)
    plt.xticks(x, seq1)
    plt.xlim(0, len(x)-1)
    #plt.ylim(0, 1)
    plt.yticks(size=30)
    plt.xticks(x,seq1,size=25)

    #print name+str(i)+'TESTT'
    title = "{} Score: {:.3f}".format(id, score1)
    #plt.title(title, size=2*scaling_factor)
    filename = make_filename_safe(output_name + '-' + str(id) + '_profile' + '.png')
    print "\n Saving", filename, "with scaling_factor =", str(scaling_factor)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def shape_to_batchsize(input_seqs, batch_size):
    output_seq = input_seqs[:(int(round((len(input_seqs))))/batch_size)*batch_size]
    return output_seq


def check_same_length(input_seqs, length):
    output_seqs = []
    for i in range(len(input_seqs)):
        if len(input_seqs[i]) != length:
            print "\tSequence {} is not {} bp.".format(str(i), str(len(input_seqs[i])))
        else:
            output_seqs.append(input_seqs[i])
    return output_seqs


def encode_input_data(seqs, max_length):
    pad_sequences_with_N(seqs, max_length)
    return seqs


def shuffle_all_seqs(seqs):
    all_seqs = list(''.join(seqs))
    random.shuffle(all_seqs)
    all_seqs = ''.join(all_seqs)

    k = 0
    out = []
    for i in range(len(seqs)):
        out.append(all_seqs[k:k+len(seqs[i])])
        k += len(seqs[i])
    return out


def pad_sequences_with_N(added_seqs, length):
    for i in range(len(added_seqs)):
        begin = end = 0  # make sure these are zero
        if len(added_seqs[i]) < length:
            missing = length - len(added_seqs[i])
            begin = int(missing/2)
            end = missing - begin
        added_seqs[i] = begin*'n' + added_seqs[i] + end*'n'
        if len(added_seqs[i]) != length:
            print len(added_seqs[i])
            print i
            break
    return added_seqs


def write_test_output(auroc, roc, output_file):
    data = [[a, b] for a, b in zip(roc[0], roc[1])]
    json_obj = {"auroc" : auroc, "data" : data}
    with open(output_file, "w") as f:
        f.write(json.dumps(json_obj))
        f.write("\n")


def write_test_predictions(seq, ids, y, predictions, output_file):
    y = np.array(y)
    with open(output_file, "w") as f:
        f.write("id\tseq\tclass\tscore\n")
        for i in range(len(seq)):
            start = next(j for j in range(len(seq[i])) if seq[i][j] != "n")
            end = next(j for j in range(len(seq[i]),0,-1) if seq[i][j-1] != "n")
            f.write('"{}"\t{}\t{}\t{}\n'.format(ids[i], seq[i][start:end], y[i][0], predictions[i][0]))


def write_predict_output(seq, ids, predictions, weights, output_file):
    pred_results = []
    for i in range(len(seq)):
        try:
            if seq[i][0]!="n":
                start = 0
            else:
                start = next(j for j in range(len(seq[i])) if seq[i][j] != "n")
            if seq[i][-1]!="n":
                end = len(seq[i])
            else:
                end = next(j for j in range(len(seq[i]),0,-1) if seq[i][j-1] != "n")
            pred_results.append({
                'sequence': str(seq[i][start:end]),
                'id': ids[i],
                'score': float(predictions[i]),
                'weights': weights[i][start:end]
                })
        except StopIteration: # happens if there's a sequence with all 'n' characters
                    pass
    json_obj = {"predictions" : pred_results}
    with open(output_file, "w") as f:
        f.write(json.dumps(json_obj))
        f.write("\n")

def write_long_predict_output(seq, ids, weights, output_file):
    pred_results = []
    for i in range(len(seq)):
        start = next(j for j in range(len(seq[i])) if seq[i][j].lower() != 'n')
        end = next(j for j in range(len(seq[i]),0,-1) if seq[i][j-1].lower() != 'n')
        pred_results.append({
            'sequence': str(seq[i][start:end]),
            'id': ids[i],
            'weights': weights[i][start:end]
            })
    json_obj = {"predictions" : pred_results}
    with open(output_file, "w") as f:
        f.write(json.dumps(json_obj))
        f.write("\n")


def write_predict_with_variant_output(seq, ids, predictions, weights, var_seq, var_ids, var_predictions, var_weights, output_file):
    pred_results = []
    if len(seq) == len(var_seq):
        for i in range(len(seq)):
            start = next(j for j in range(len(seq[i])) if seq[i][j] != "n")
            end = next(j for j in range(len(seq[i]),0,-1) if seq[i][j-1] != "n")
            pred_results.append({
                'sequence': str(seq[i][start:end]),
                'variant_sequence': str(var_seq[i][start:end]),
                'id': ids[i],
                'variant_id': var_ids[i],
                'score': float(predictions[i]),
                'variant_score': float(var_predictions[i]),
                'weights': weights[i][start:end],
                'variant_weights': var_weights[i][start:end]
                })
    if len(seq) == 1: # just one reference sequence, variants are same-length variations of that
        if len(var_seq) == 1: # just 1 variant sequence, don't loop as it seems to result in 2 identical results in the json output
            start = next(j for j in range(len(seq[0])) if seq[0][j] != "n")
            end = next(j for j in range(len(seq[0]),0,-1) if seq[0][j-1] != "n")
            pred_results = [{
                'sequence': str(seq[0][start:end]),
                'variant_sequence': str(var_seq[0][start:end]),
                'id': ids[0],
                'variant_id': var_ids[0],
                'score': float(predictions[0]),
                'variant_score': float(var_predictions[0]),
                'weights': weights[0][start:end],
                'variant_weights': var_weights[0][start:end]
                }]
        else:
            for i in range(len(var_seq)):
                start = next(j for j in range(len(seq[0])) if seq[0][j] != "n")
                end = next(j for j in range(len(seq[0]),0,-1) if seq[0][j-1] != "n")
                pred_results.append({
                    'sequence': str(seq[0][start:end]),
                    'variant_sequence': str(var_seq[i][start:end]),
                    'id': ids[0],
                    'variant_id': var_ids[i],
                    'score': float(predictions[0]),
                    'variant_score': float(var_predictions[i]),
                    'weights': weights[0][start:end],
                    'variant_weights': var_weights[i][start:end]
                    })

    json_obj = {"predictions" : pred_results}
    with open(output_file, "w") as f:
        f.write(json.dumps(json_obj))
        f.write("\n")

def get_sequence_logo_data(seqs, lstmw):
    weights = []
    for i in range(len(seqs)):
        w = [lstmw[i][j] for j in range(len(seqs[i]))]
        for j in range(len(seqs[i])):
            if seqs[i][j] == 'n':
                w[j] = 0
        weights.append(w)
    return weights


def write_sequence_logo_output(seqs, weights, path):
    data = [[str(a), b] for a, b in zip(seqs, weights)]
    json_obj = {"predictions" : data}
    f = open(path, "w")
    f.write(json.dumps(json_obj))
    f.write("\n")
    f.close()


def split_data(seqs, ids, ratios):
    assert len(seqs) == len(ids)
    assert ratios[0] + ratios[1] < 1.0

    zz = list(zip(seqs, ids))
    random.shuffle(zz)
    seqs, ids = zip(*zz)

    ntrain = int(round(ratios[0] * len(seqs)))
    nval = int(round(ratios[1] * len(seqs)))

    train = (seqs[:ntrain], ids[:ntrain])
    val = (seqs[ntrain:ntrain+nval], ids[ntrain:ntrain+nval])
    test = (seqs[ntrain+nval:], ids[ntrain+nval:])

    return train, val, test


def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]


def k_fold_generator(X, y, k_fold=10):
    freq = np.array([1.0, 1.0, 1.0, 1.0])
    out = []
    subset_sizeX = int(len(X) / k_fold)
    subset_sizeY = int(len(y) / k_fold)
    for k in range(k_fold):
        X_train = X[:k * subset_sizeX] + X[(k + 1) * subset_sizeX:]
        X_valid = X[k * subset_sizeX:][:subset_sizeX]
        y_train = y[:k * subset_sizeY] + y[(k + 1) * subset_sizeY:]
        y_valid = y[k * subset_sizeY:][:subset_sizeY]
        subset_size2X = int(len(X_train) / (k_fold-1))
        subset_size2Y = int(len(y_train) / (k_fold-1))
        if k==(k_fold-1):
            k=0
        X_test = X_train[k * subset_size2X:][:subset_size2X]
        y_test = y_train[k * subset_size2Y:][:subset_size2Y]
        X_train = X_train[:k * subset_size2X] + X_train[(k + 1) * subset_size2X:]
        y_train = y_train[:k * subset_size2Y] + y_train[(k + 1) * subset_size2Y:]

        out.append([onehot_binary(X_train, y_train, freq, vocab=constants.VOCAB),
            onehot_binary(X_valid, y_valid, freq, vocab=constants.VOCAB),
            onehot_binary(X_test, y_test, freq, vocab=constants.VOCAB)
            ])
    return out


def k_fold_generator_strings(X, y, k_fold=10):
    freq = np.array([1.0, 1.0, 1.0, 1.0])
    out = []
    subset_sizeX = int(len(X) / k_fold)
    subset_sizeY = int(len(y) / k_fold)
    for k in range(k_fold):
        X_train = X[:k * subset_sizeX] + X[(k + 1) * subset_sizeX:]
        X_valid = X[k * subset_sizeX:][:subset_sizeX]
        y_train = y[:k * subset_sizeY] + y[(k + 1) * subset_sizeY:]
        y_valid = y[k * subset_sizeY:][:subset_sizeY]
        subset_size2X = int(len(X_train) / (k_fold-1))
        subset_size2Y = int(len(y_train) / (k_fold-1))
        if k==(k_fold-1):
            k=0
        X_test = X_train[k * subset_size2X:][:subset_size2X]
        y_test = y_train[k * subset_size2Y:][:subset_size2Y]
        X_train = X_train[:k * subset_size2X] + X_train[(k + 1) * subset_size2X:]
        y_train = y_train[:k * subset_size2Y] + y_train[(k + 1) * subset_size2Y:]

        out.append([X_train, y_train,
          X_valid, y_valid,
          X_test, y_test])
    return out

def k_fold_generator_strings2(train_seqs, train_ids, train_bkgs, train_bkg_ids, k_fold=10):
    freq = np.array([1.0, 1.0, 1.0, 1.0])
    out = []
    subset_sizeX = int(len(train_bkgs) / k_fold)
    subset_sizeY = int(len(train_seqs) / k_fold)
    for k in range(k_fold):
        X_train_sqs = train_bkgs[:k * subset_sizeX] + train_bkgs[(k + 1) * subset_sizeX:]
        X_train_ids = train_bkg_ids[:k * subset_sizeX] + train_bkg_ids[(k + 1) * subset_sizeX:]
        X_valid_sqs = train_bkgs[k * subset_sizeX:][:subset_sizeX]
        X_valid_ids = train_bkg_ids[k * subset_sizeX:][:subset_sizeX]
        y_train_sqs = train_seqs[:k * subset_sizeY] + train_seqs[(k + 1) * subset_sizeY:]
        y_train_ids = train_ids[:k * subset_sizeY] + train_ids[(k + 1) * subset_sizeY:]
        y_valid_sqs = train_seqs[k * subset_sizeY:][:subset_sizeY]
        y_valid_ids = train_ids[k * subset_sizeY:][:subset_sizeY]
        subset_size2X = int(len(X_train_sqs) / (k_fold-1))
        subset_size2Y = int(len(y_train_sqs) / (k_fold-1))
        if k==(k_fold-1):
            k=0
        X_test_sqs = X_train_sqs[k * subset_size2X:][:subset_size2X]
        X_test_ids = X_train_ids[k * subset_size2X:][:subset_size2X]
        y_test_sqs = y_train_sqs[k * subset_size2Y:][:subset_size2Y]
        y_test_ids = y_train_ids[k * subset_size2Y:][:subset_size2Y]
        X_train_sqs = X_train_sqs[:k * subset_size2X] + X_train_sqs[(k + 1) * subset_size2X:]
        X_train_ids = X_train_ids[:k * subset_size2X] + X_train_ids[(k + 1) * subset_size2X:]
        y_train_sqs = y_train_sqs[:k * subset_size2Y] + y_train_sqs[(k + 1) * subset_size2Y:]
        y_train_ids = y_train_sqs[:k * subset_size2Y] + y_train_ids[(k + 1) * subset_size2Y:]

        out.append([X_train_ids, X_train_sqs, y_train_ids, y_train_sqs,
          X_valid_ids, X_valid_sqs, y_valid_ids, y_valid_sqs,
          X_test_ids, X_test_sqs, y_test_ids, y_test_sqs])
    return out


def build_network(args, max_length, filter_sizes):
    return network.Network(
        par_selection=args.performance_selection,
        auc_thr=args.auc_thr,
        cvfile=args.network_file,
        runmode=args.runmode,
        VOCAB=constants.VOCAB,
        MINI_BATCH_SIZE=args.batch_size,
        SEQ_SIZE=max_length,
        early_stopping=args.early_stopping,
        ETA=args.learning_rate,
        L2=args.l2,
        GRAD_CLIP=constants.GRAD_CLIP,
        NUMBER_OF_LSTM_LAYERS=args.lstm_layers,
        NUMBER_OF_CONV_LAYERS=len(filter_sizes),
        N_LSTM=args.lstm_nodes,
        DROPOUT_IN=args.dropout_in,
        DROPOUT_OUT=args.dropout_out,
        DROPOUT_LSTM=args.lstm_dropout,
        DROPOUT_CONV=constants.DROPOUT_CONV,
        FILTERS=args.num_filters,
        FILTER_SIZES=filter_sizes,
        PADDING=constants.PADDING)


def read_fasta_file_and_prepare_data(fasta_file):

    """
    This function extracts sequences from fasta files
    """

    seq_data = []
    ids = []
    with open(fasta_file,"r") as f:
        id = False
        for line in f:
            if len(line) == 0:
                continue
            if line[0] == ">":
                if not id: # first line
                    id = line.rstrip()[1:]
                    seq = ""
                else:
                    ids.append(id)
                    seq_data.append(seq)
                    id = line.rstrip()[1:]
                    seq = ""
            else:
                seq += line.rstrip()
        # end of file, append the last fasta line
        ids.append(id)
        seq_data.append(seq)
    return seq_data, ids


def regulate_length(inp, ids, mn, mx):
    n_inp = []
    n_ids = []
    for i in range(len(inp)):
        if len(inp[i]) >= mn and len(inp[i])<= mx:
            n_inp.append(inp[i])
            n_ids.append(ids[i])
    return n_inp, n_ids


def main():
    args = parse_arguments()
    import bed
    import fasta
    import cPickle as pickle
    from roc import get_auroc_data
    from making_CNN_PFMs import convolutional_logos

    if args.no_extra_padding:
        nep = 0
    if not args.no_extra_padding:
        nep = 1

    if args.random_seed:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)

    bkg_list, bkg_ids_list = [], []
    if args.force_bed or args.sequences.lower().endswith(".bed"):
        print " Reading sequences from BED file"
        if args.genome_file is None:
            raise Exception(" Genome FASTA file needed for reading BED file.")
        if args.gtf_file is None:
            raise Exception(" GTF file needed for reading BED file.")

        seq_list, bkg_list, seq_ids_list = bed.produce_sequences(args.sequences,
                                                                 args.genome_file,
                                                                 args.gtf_file,
                                                                 args.min_length,
                                                                 args.max_length,
                                                                 args.bed_width,
                                                                 args.bed_padding,
                                                                 False)

        bkg_ids_list = ["bg_"+s for s in seq_ids_list]
    else:
        print " Reading sequences from FASTA file:",str(args.sequences)
        if args.runmode == "predict":
            seq_list, seq_ids_list = read_fasta_file_and_prepare_data(args.sequences)
        elif args.runmode == "predict_long":
            seq_list, seq_ids_list = fasta.read_long_fasta_file(args.sequences)
        else:
            seq_list, seq_ids_list = fasta.read_fasta_file(args.sequences, args.min_length, args.max_length)
            print len(seq_list), seq_list[:10]

    if args.background_sequences:
        print " Reading background sequences"
        bkg_list, bkg_ids_list = fasta.read_fasta_file(args.background_sequences, args.min_length, args.max_length)
    elif args.background_shuffle:
        print " Making background by shuffling input bases"
        bkg_list = shuffle_all_seqs(seq_list)
        bkg_ids_list = ["bg"+str(i) for i in range(len(bkg_list))]

    if len(seq_list) == 0:
        if args.runmode == "predict":
            raise Exception(" Set of input sequences is empty.")
        else:
            raise Exception(" Set of binding sequences is empty.")

    if args.runmode == "train" or args.runmode == "cv":
        print args.network_file
        print args.predict_function_file

        freq = np.array([1.0, 1.0, 1.0, 1.0])


        max_input_length = max(max(map(len, seq_list)), (max(map(len, bkg_list)) if len(bkg_list) > 0 else 0))
        print " Max input length:", str(max_input_length)

        filter_sizes = [len(constants.VOCAB)*int(x) for x in args.filter_sizes]
        max_length = max_input_length + (max(filter_sizes)/len(constants.VOCAB)-1)*2*nep
        #print " Max used length:", str(max_length)

        print("\n Setting up CNN_BLSTM model")
        print('\n General network info:')
        print ' Number of epochs: {}'.format(args.num_epochs)
        print ' Stopping early after {} epochs without model improvement.'.format(args.early_stopping)
        print(' Class 0: background')
        print(' Class 1: positive binding sites')
        print " Filter sizes: " + ", ".join(str(i) for i in filter_sizes)
        print " Number of BLSTM nodes: " + str(args.lstm_nodes)
        sys.stdout.flush()

    if args.runmode == "train":
        if len(bkg_list) == 0:
            raise Exception("No background sequences provided for training.")

        print " Building network"
        net = build_network(args, max_length, filter_sizes)

        print " Equalizing sequences"
        encode_input_data(seq_list, max_length)
        encode_input_data(bkg_list, max_length)

        if args.balanced_input:
            set_size = min([len(seq_list), len(bkg_list)])
            print " Setting set size to " + str(set_size)
            seq_list = seq_list[:set_size]
            seq_ids_list = seq_ids_list[:set_size]
            bkg_list = bkg_list[:set_size]
            bkg_ids_list = bkg_ids_list[:set_size]

        if args.write_sequences:
            fasta.write_fasta_file(args.write_sequences, seq_list, seq_ids_list)
        if args.write_background_sequences:
            fasta.write_fasta_file(args.write_background_sequences, bkg_list, bkg_ids_list)

        (train_seqs, train_ids), (val_seqs, val_ids), (test_seqs, test_ids) = split_data(seq_list, seq_ids_list, args.data_split)
        (train_bkgs, train_bkg_ids), (val_bkgs, val_bkg_ids), (test_bkgs, test_bkg_ids) = split_data(bkg_list, bkg_ids_list, args.data_split)

        print " One-hot encoding sequences"
        all_inputs = [[onehot_binary(train_bkgs, train_seqs, freq, vocab=constants.VOCAB),
            onehot_binary(val_bkgs, val_seqs, freq, vocab=constants.VOCAB),
            onehot_binary(test_bkgs, test_seqs, freq, vocab=constants.VOCAB)]]

        X_test, y_test = onehot_binary(test_bkgs, test_seqs, freq, vocab=constants.VOCAB)

        print X_test.shape, X_test[0], X_test[1]

        net.fit(all_inputs, num_epochs=args.num_epochs)

        if args.network_file:
            print " Saving network to file"
            network.save_network(net.network, net.options, args.network_file, freq)
        if args.predict_function_file:
            print " Saving prediction function to file"
            network.save_prediction_function(net, args.predict_function_file, freq)

        print "\n Testing network"
        predict_fn, outpar = net.compile_prediction_function()
        results = network.predict(net, net.options, predict_fn, X_test, outpar)

        print 'overall max score of test set: ', np.max(abs(np.array(results["weights"])))

        # returns a dict with different results, but always
        # at least "predictions" (classifier) and "profiles" (binding profiles)
        predictions = results["predictions"]
        auroc, roc = get_auroc_data(y_test, predictions, segments=100)
        print "\n Test AUROC: " + str(auroc)
        if args.predict_PFM_file:
            temp = np.array(X_test).reshape((-1, max_length, 4)) * results["weights"].reshape(-1, max_length, 1)
            pfmin = np.array(X_test) * temp.reshape(X_test.shape)
            pfmin[pfmin < 0] = 0
            pfmin = pfmin / (pfmin + 0.000000001)
            FS = [len(seq_list[0]) - filter_sizes[i] / len(constants.VOCAB) + 1 for i in range(len(filter_sizes))]
            convolutional_logos(results["argmax"], results["cnscore"], pfmin, FS,
                                    [args.num_filters] * len(filter_sizes), filter_sizes, constants.VOCAB,
                                    args.predict_PFM_file, args.draw_seq_logos)
        if args.test_output_file:
            write_test_output(auroc, roc, args.test_output_file)
            print len(seq_list), len(seq_ids_list), len(predictions), predictions[0]
        if args.test_predictions_file:
            write_test_predictions(test_bkgs+test_seqs, test_bkg_ids+test_ids, y_test, predictions, args.test_predictions_file)

    elif args.runmode == "predict" or args.runmode == "predict_long":
        # try loading the network, either from prediction function, or from the network itself and then compile a prediction function
        if args.predict_function_file:
            print " Loading prediction function from: " + str(args.predict_function_file)
            try:
                predict_fn, options, output_shape, outpar, freq = network.load_prediction_function(args.predict_function_file)
            except ValueError:
                print " Error loading prediction function, trying loading as network instead"
                try:
                    net,freq = network.load_network(args.predict_function_file)
                    options = net.options
                    predict_fn, outpar = net.compile_prediction_function()
                    output_shape = net.network['l_in'].output_shape
                except TypeError:
                    print " Error loading network"
        elif args.network_file:
            try:
                net,freq = network.load_network(args.network_file)
                options = net.options
                predict_fn, outpar = net.compile_prediction_function()
                output_shape = net.network['l_in'].output_shape
            except ValueError:
                print " Error loading network, trying loading as prediction function instead."
                try:
                    predict_fn, options, output_shape, outpar, freq = network.load_prediction_function(args.network_file)
                except TypeError:
                    print " Error loading network"

        if not predict_fn:
            print "Couldn't load model!"
            sys.exit()

        max_filter_size = max(options["FILTER_SIZES"])/4
        max_network_length = int(options["SEQ_SIZE"] - 2*(max_filter_size - 1))
        if args.no_extra_padding:
            max_network_length = int(options["SEQ_SIZE"])
        max_length = int(options["SEQ_SIZE"])
    if args.runmode == "predict":
        start_time = time.time()
        var_list = []
        if args.variant_sequences:
            print " Reading variant sequences from FASTA file:", str(args.variant_sequences)
            var_list, var_ids_list = fasta.read_fasta_file(args.variant_sequences, 0, max_network_length)
        max_input_length = max(max(map(len, seq_list)), (max(map(len, var_list)) if len(var_list) > 0 else 0))
        if max_input_length > max_network_length:
            raise Exception("Cannot predict on sequences longer than the network was trained on.\n Maximum input length was {} bp, but the network cannot handle sequences longer than {} bp.\n Try using --runmode predict_long instead.".format(str(max_input_length),str(max_network_length)))

        print seq_list[0]
        print len(seq_list[0])
        seq_list = encode_input_data(seq_list, max_network_length + 2*(max_filter_size - 1)*nep)
        print seq_list[0]
        print len(seq_list[0])

        print " One-hot encoding sequences"
        X_test = onehot_encode(seq_list, freq,  vocab=constants.VOCAB)


#        print " Predicting binding"
        results = network.predict_without_network(predict_fn, options, output_shape, X_test, outpar)
        predictions = results["predictions"]
        inp = np.sum(np.reshape(X_test, (-1, options["SEQ_SIZE"], len(options["VOCAB"]))), axis =-1)
        weight = results["weights_par"]*inp
#        print "predicted " + str(len(predictions)) + " sequences."
        if args.variant_sequences:
            var_list = encode_input_data(var_list, max_network_length + 2*(max_filter_size - 1)*nep)
            X_var = onehot_encode(var_list, freq,  vocab=constants.VOCAB)
            var_results = network.predict_without_network(predict_fn, options, output_shape, X_var, outpar)
            var_predictions = var_results["predictions"]
            var_weight = var_results["weights_par"]*inp

        if args.predict_output_file:
            if args.predict_output_file.lower().endswith(".json"):
                predict_weights = get_sequence_logo_data(seq_list, results["weights_par"])
                if args.variant_sequences:
                    var_predict_weights = get_sequence_logo_data(var_list, var_results["weights_par"])
                    write_predict_with_variant_output(seq_list, seq_ids_list, predictions, predict_weights, var_list, var_ids_list, var_predictions, var_predict_weights, args.predict_output_file)
                else:
                    write_predict_output(seq_list, seq_ids_list, predictions, predict_weights, args.predict_output_file)

            else:
                if args.variant_sequences:
                    with open(args.predict_output_file, "w") as f:
                        for i in range(len(var_predictions)):
                            if len(predictions) == len(var_predictions):
                                f.write(str(seq_ids_list[i]) + "\t" + str(var_ids_list[i]) + "\t" + str(seq_list[i].strip('n')) + "\t" + str(var_list[i].strip('n')) + "\t" + str(predictions[i][0]) + "\t" + str(var_predictions[i][0]) + "\n")
                                if args.draw_profiles:
                                    make_paired_profile(wt_scores=weight[i], var_scores=var_weight[i], wt_seq=seq_list[i], var_seq=var_list[i], wt_name=seq_ids_list[i], var_name=var_ids_list[i], make_diff = args.make_diff, sigmoid_profile = args.sigmoid_profile, output_name=args.predict_output_file)
                            elif len(predictions) == 1: # just one reference sequence, variants are same-length variations of that
                                f.write(str(seq_ids_list[0]) + "\t" + str(var_ids_list[i]) + "\t" + str(seq_list[0].strip('n')) + "\t" + str(var_list[i].strip('n')) + "\t" + str(predictions[0][0]) + "\t" + str(var_predictions[i][0]) + "\n")
                                if args.draw_profiles:
                                    make_paired_profile(wt_scores=weight[0], var_scores=var_weight[i], wt_seq=seq_list[0], var_seq=var_list[i], wt_name=seq_ids_list[0], var_name=var_ids_list[i], make_diff = args.make_diff, sigmoid_profile = args.sigmoid_profile, output_name=args.predict_output_file)
                else:
                    with open(args.predict_output_file, "w") as f:
                        for i in range(len(predictions)):
                            f.write(str(seq_ids_list[i]) + "\t" + str(seq_list[i].strip('n')) + "\t" + str(predictions[i][0]) + "\n")
                            if args.draw_profiles:
                                make_single_profile(scores=weight[i], seq=seq_list[i], name=seq_ids_list[i], sigmoid_profile=args.sigmoid_profile, output_name=args.predict_output_file)

        if args.predict_PFM_file:
            temp = np.array(X_test).reshape((-1, max_length, 4)) * results["weights_par"].reshape(-1, max_length, 1)
            pfmin = np.array(X_test) * temp.reshape(X_test.shape)
            pfmin[pfmin < 0] = 0
            pfmin = pfmin / (pfmin + 0.000000001)
            filter_sizes = options["FILTER_SIZES"]

            cn = results["cnscore"]
            arg = results["argmax"]
            ins = pfmin

            pred = []
            #for pr in range(len(predictions)):
            #   pred.append( [predictions[pr][0], arg[pr], cn[pr], ins[pr]] )

            for pr in range(len(predictions)):
                pred.append( [predictions[pr][0], pr] )

            pred.sort()
            pred.reverse()

            arg1 = []
            cn1 = []
            ins1 = []
            if args.PFM_on_half == True:
                number_of_seqs = int(len(pred)/2)
                for i in range(number_of_seqs):
                    arg1.append(arg[pred[i][1]])
                    cn1.append(cn[pred[i][1]])
                    ins1.append(ins[pred[i][1]])

            if args.PFM_on_half != True:
                number_of_seqs = int(min([1000, len(pred)]))
                for i in range(number_of_seqs):
                    arg1.append(arg[pred[i][1]])
                    cn1.append(cn[pred[i][1]])
                    ins1.append(ins[pred[i][1]])


            FS = [len(seq_list[0]) - filter_sizes[i] / len(options["VOCAB"]) + 1 for i in range(len(filter_sizes))]

            convolutional_logos(arg1, cn1, ins1, FS,
                                [options["FILTERS"]] * len(filter_sizes), filter_sizes, options["VOCAB"],
                                args.predict_PFM_file, args.draw_seq_logos)
            print "PFMs based on the " + str(number_of_seqs) + " best scoring sequences have been made"
        ctime = time.time() - start_time
        print("Entire prediction took {:.3f}s".format(ctime))


    elif args.runmode == "predict_long":
        prediction_weights = []
        short_seq_list = []
        for seq_i in range(len(seq_list)):
            seq = seq_list[seq_i]
            temp_seq_list = []
            if len(seq) > max_network_length:
                seq = (max_filter_size - 1)*'n' + seq + (max_filter_size - 1)*'n' # pad the seq at the boundaries
                seq_list[seq_i] = seq
                for s in range(0, len(seq) - (max_network_length+2*(max_filter_size - 1)-1)):
                    segment = seq[0 + s:max_network_length+2*(max_filter_size - 1) + s].lower()
                    temp_seq_list.append(segment)
                temp_seq_list = encode_input_data(temp_seq_list, max_network_length+2*(max_filter_size - 1)*nep)
                #print " One-hot encoding sequences"
                X_test = onehot_encode(temp_seq_list, freq,  vocab=constants.VOCAB)
                results = network.predict_without_network(predict_fn, options, output_shape, X_test, outpar)
                predictions = results["predictions"]
                inp = np.sum(np.reshape(X_test, (-1, options["SEQ_SIZE"], len(options["VOCAB"]))), axis =-1)
                weights = results["weights_par"]*inp
                average_weights = np.zeros(len(seq))
                b = 0
                for weight in weights:
                    if b == 0:
                        for w in range(max_filter_size - 1,int(len(weight)/2)):
                            average_weights[0+b+w] = weight[w]
                    elif b == len(seq) - (max_network_length+2*(max_filter_size - 1)):
                        for w in range(int(len(weight)/2),len(weight)-(max_filter_size - 1)):
                            average_weights[0+b+w] = weight[w]
                    else:
                        w = int(len(weight)/2) + 1
                        average_weights[0+b+w] = weight[w]
                    b += 1
                prediction_weights.append(average_weights)

            else:
                temp_seq_list = encode_input_data([seq], max_network_length+2*(max_filter_size - 1)*nep)
                seq_list[seq_i] = temp_seq_list[0]
                #print " One-hot encoding sequences"
                X_test = onehot_encode(temp_seq_list, freq,  vocab=constants.VOCAB)
                results = network.predict_without_network(predict_fn, options, output_shape, X_test, outpar)
                predictions = results["predictions"]
                inp = np.sum(np.reshape(X_test, (-1, options["SEQ_SIZE"], len(options["VOCAB"]))), axis =-1)
                weights = results["weights_par"]*inp
                temp_weights = []
                for w in weights[0]:
                    temp_weights.append(w)
                prediction_weights.append(temp_weights)

        print " Completed long predictions on",str(len(seq_list)),"input sequences."
        if args.predict_output_file:
            full_weights = get_sequence_logo_data(seq_list, prediction_weights)
            write_long_predict_output(seq_list, seq_ids_list, full_weights, args.predict_output_file)


    elif args.runmode == "cv":
        if len(bkg_list) == 0:
            raise Exception("No background sequences provided for cross-validation.")
        print " Equalizing sequences"
        encode_input_data(seq_list, max_length)
        encode_input_data(bkg_list, max_length)
        if args.balanced_input:
            set_size = min([len(seq_list), len(bkg_list)])
            print " Setting set size to " + str(set_size)
            seq_list = seq_list[:set_size]
            seq_ids_list = seq_ids_list[:set_size]
            bkg_list = bkg_list[:set_size]
            bkg_ids_list = bkg_ids_list[:set_size]

        print " Building network"
        net = build_network(args, max_length, filter_sizes)
        cross_fold = 10 # set to 10-fold cross-validation
        train_seqs, train_ids = seq_list, seq_ids_list
        train_bkgs, train_bkg_ids = bkg_list, bkg_ids_list

        print " One-hot encoding sequences"
        all_inputs = k_fold_generator(train_bkgs, train_seqs, k_fold=cross_fold)
        #all_strings = k_fold_generator_strings(train_bkgs, train_bkg_ids, k_fold=cross_fold)
        all_strings2 = k_fold_generator_strings2(train_seqs, train_ids, train_bkgs, train_bkg_ids, k_fold=cross_fold)
        #for qqb in range(len(all_inputs)):
        #    all_inputs[qqb] += all_strings[qqb]
        if args.export_test_sets:
            for i in range(len(all_strings2)):
                bkg_tr_ids, bkg_tr_sqs, tr_ids, tr_sqs, bkg_va_ids, bkg_va_sqs, va_ids, va_sqs, bkg_te_ids, bkg_te_sqs, te_ids, te_sqs = all_strings2[i]
                with open("cv" + str(i+1) + "-test.pos.fa", "w") as fa_out:
                    for j in range(len(te_sqs)):
                        fa_out.write(">" + str(te_ids[j]) + "\n" + str(te_sqs[j]).replace('n', '').replace('N', '') + "\n")
                with open("cv" + str(i+1) + "-test.neg.fa", "w") as fa_out:
                    for j in range(len(bkg_te_sqs)):
                        fa_out.write(">" + str(bkg_te_ids[j]) + "\n" + str(bkg_te_sqs[j]).replace('n', '').replace('N', '') + "\n")

        net.build_model()
        n, cv_results, auc_values, roc_sets = net.fit(all_inputs, num_epochs=args.num_epochs)

        if args.performance_selection == "loss":
            print " Lowest loss:", str(auc_values[np.argmin(auc_values)]), "from CV set", str(np.argmin(auc_values)+1)
            print " All loss scores:", str(auc_values)
            print " Saving overall best network."
            best_cv = np.argmin(auc_values)+1

        if args.performance_selection == "auroc":
            print " Best AUROC:", str(auc_values[argmax(auc_values)]), "from CV set", str(argmax(auc_values)+1)
            print " All AUROC scores:", str(auc_values)
            print " Saving overall best network."
            best_cv = argmax(auc_values)+1

        net,freq = network.load_network(args.network_file.replace('_cv_cycle_data.pkl','')+"_cv"+str(best_cv))
        network.save_network(net.network, net.options, args.network_file.replace('_cv_cycle_data.pkl','')+"_best_cv_model", freq)
        network.save_prediction_function(net, args.network_file.replace('_cv_cycle_data.pkl','')+"_best_cv_predict_fn", freq)
        print " CV runmode completed."


if __name__ == "__main__":
    main()
