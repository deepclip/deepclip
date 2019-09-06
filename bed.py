#!/usr/bin/env python
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.Alphabet import generic_dna
import HTSeq
import argparse
import random
import sys

def parse_arguments():
    VERSION = "0.7.0"
    description = "Program to construct suitable backgrounds for DeepCLIP input. Input should be a BED file with positive binding sites."

    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # optional arguments - order kept
    parser.add_argument("--min_length",
                        type=int,
                        default=1,
                        help="Minimum region length.")
    parser.add_argument("--max_length",
                        type=int,
                        default=400,
                        help="Maximum region length.")
    parser.add_argument('-p', "--padding",
                        type=int,
                        default=0,
                        help="Padding added to each region.")
    parser.add_argument('-w', "--width",
                        type=int,
                        default=0,
                        help="Make the regions a uniform width. A value of 0 retains the lengths as given. This option only applies to BED input format")

    parser.add_argument('-r', "--random_seed",
                        type=int,
                        default=1234,
                        help="Random seed number.")

    parser.add_argument('-s', "--session",
                        type=str,
                        default="dummy",
                        help="Session id, used to handle multiple users at the same time.")

    parser.add_argument('-v', '--version',
                        action='version',
                        version="%(prog)s (version " + VERSION + ")")

    parser.add_argument('-l', "--log",
                        action='store_true',
                        help="Enable logging")

    parser.add_argument('-c', "--compatible",
                        action='store_true',
                        help="Enable GraphProt compatible output.")

    # positional arguments - order kept
    parser.add_argument("input_file",
                        type=str,
                        help='Input file with positive binding sites.')

    parser.add_argument("fasta_file",
                        type=str,
                        default="/data/Genomes/human/hg19/hg19.fa",
                        help='Genome fasta reference')

    parser.add_argument("gtf_file",
                        type=str,
                        default=None,
                        help='GTF reference')

    parser.add_argument("out_file",
                        type=str,
                        default=None,
                        help="Positive sequence output file")

    parser.add_argument("out_bkg_file",
                        type=str,
                        default=None,
                        help="Negative sequence output file")

    # Create a list object to store all the processed arguments in
    return parser.parse_args()


def read_genome(infile):
    """ Function to read the genome fasta reference.
        Returns a dictionary of the sequences. """
    #print " Reading genome...",
    with open(infile, "rU") as fasta_file:
        genome = SeqIO.to_dict(SeqIO.parse(fasta_file, "fasta"))
    #print "done"
    return genome


def featuretype_filter(feature, featuretype):
    """Function to filter GTF file by feature"""
    if feature[2] == featuretype:
        return True
    return False


def produce_sequences(bed_file, fasta_file, gtf_file, min_length, max_length, width, padding, graphprot_compatible=False):
    print " Reading primary peaks from BED file"
    bed_file = HTSeq.BED_Reader(bed_file)
    input_peaks = HTSeq.GenomicArrayOfSets("auto", stranded=True)
    total_peaks = 0
    for peak in bed_file:
        total_peaks += 1
        if width > 0:
            mid = int((peak.iv.start+peak.iv.end)/2)
            peak.iv.start = mid - int(width/2)
            peak.iv.end = peak.iv.start + width
        if padding > 0:
            peak.iv.start -= padding
            peak.iv.end += padding
        input_peaks[peak.iv] += peak.name

    print " Reading GTF file from " + str(gtf_file)
    genes = HTSeq.GenomicArrayOfSets("auto", stranded=True)
    gene_dict = dict()
    gtf_file = HTSeq.GFF_Reader(gtf_file)
    total_genes = 0
    for feature in gtf_file:
        if feature.type == "gene":
            total_genes += 1
            genes[feature.iv] += feature.name
            gene_dict[feature.name] = feature
    if total_genes == 0: # this GTF file doesn't have 'gene' features, we need to build the gene intervals from the exon intervals instead
        print " No 'gene' features in GTF, building gene intervals from exons instead."
        for feature in gtf_file:
            if feature.type == "exon":
                gene = gene_dict.get(feature.attr["gene_id"], False)
                if not gene:
                    feature.type = 'gene'
                    gene_dict[feature.attr["gene_id"]] = feature
                    total_genes += 1
                else:
                    if gene.iv.start > feature.iv.start:
                        gene.iv.start = feature.iv.start
                    if gene.iv.end < feature.iv.end:
                        gene.iv.end = feature.iv.end
                    gene_dict[feature.attr["gene_id"]] = gene
        for gene in gene_dict.values():
            genes[gene.iv] += gene.attr["gene_id"]
    print " Loaded {} total genes.".format(total_genes)

    print " Reading genome from file " + str(fasta_file) + " ...",
    sys.stdout.flush()
    genome = read_genome(fasta_file)
    print "done"

    print " Filtering and constructing background..."
    pos_peaks = HTSeq.GenomicArrayOfSets("auto", stranded=True)
    neg_peaks = HTSeq.GenomicArrayOfSets("auto", stranded=True)

    pos_seqs = []
    neg_seqs = []
    seq_ids = []

    not_in_gene = 0
    multiple_genes = 0
    redundant = 0
    invalid = 0
    for peak in bed_file:
        valid = True
        iset = None
        if peak.iv.length < min_length or peak.iv.length > max_length:
            valid = False
            invalid += 1
        if valid:
            if width > 0:
                mid = int((peak.iv.start+peak.iv.end)/2)
                peak.iv.start = mid - int(width/2)
                peak.iv.end = peak.iv.start + width
            if padding > 0:
                peak.iv.start -= padding
                peak.iv.end += padding
            for iv2, step_set in input_peaks[peak.iv].steps():
                if iset is None:
                    iset = step_set.copy()
                else:
                    iset.intersection_update(step_set)
        try:
            overlaps = len(iset)
        except TypeError:
            overlaps = 0
        if overlaps == 1 and valid:
            # this peak does not overlap other peaks after padding, so we can assume it's reasonably unique
            pos_peaks[peak.iv] += peak.name

            # now find the gene that it overlaps
            gset = None
            #print " Looking for overlapping gene in list of {} total genes on chromosome {}.".format(len(genes[peak.iv]), peak.iv)
            for iv2, step_set in genes[peak.iv].steps():
                if gset is None:
                    gset = step_set.copy()
                else:
                    gset.intersection_update(step_set)

            if len(gset) == 1:
                # this peak overlaps exactly one gene so we know where to randomly choose a background sequence
                gene = gene_dict[list(gset)[0]]
                overlap = True
                overlap_counter = 0
                while overlap:
                    overlap_counter += 1
                    start = random.randint(gene.iv.start, gene.iv.end - peak.iv.length)
                    end = start + peak.iv.length
                    neg_peak = HTSeq.GenomicInterval(gene.iv.chrom, start, end, gene.iv.strand)
                    overlap_peak = None
                    overlap_neg_peak = None
                    for iv2, step_set in pos_peaks[neg_peak].steps():
                        if overlap_peak is None:
                            overlap_peak = step_set.copy()
                        else:
                            overlap_peak.intersection_update(step_set)
                    for iv2, step_set in neg_peaks[neg_peak].steps():
                        if overlap_neg_peak is None:
                            overlap_neg_peak = step_set.copy()
                        else:
                            overlap_neg_peak.intersection_update(step_set)
                    if not overlap_peak and not overlap_neg_peak: # yes! found a non-overlapping region suitable as background sequence
                        overlap = False
                    if overlap_counter > 1000:  # accept that a non-overlap can't be found but don't use this peak
                        print "Warning: failed to find non-overlapping background for " + str(peak.name)
                        valid = False
                        overlap = False
                        invalid += 1
                if 'n' in str(genome[neg_peak.chrom][neg_peak.start:neg_peak.end].seq).lower():
                    print "Warning: 'n' in background sequence for " + str(peak.name)
                    valid = False
                    invalid += 1
                if valid:
                    neg_peaks[neg_peak] += 1
                    pos_seq = Seq(str(genome[peak.iv.chrom][peak.iv.start:peak.iv.end].seq), generic_dna)
                    if peak.iv.strand == "-":
                        pos_seq = pos_seq.reverse_complement()
                    neg_seq = Seq(str(genome[neg_peak.chrom][neg_peak.start:neg_peak.end].seq), generic_dna)
                    if neg_peak.strand == "-":
                        neg_seq = neg_seq.reverse_complement()
                    pos_seq = str(pos_seq)
                    neg_seq = str(neg_seq)

                    if graphprot_compatible:
                        pos_seq = pos_seq[:padding].lower() + pos_seq[padding:-padding].upper() + pos_seq[-padding:].lower()
                        neg_seq = neg_seq[:padding].lower() + neg_seq[padding:-padding].upper() + neg_seq[-padding:].lower()
                    pos_seqs.append(pos_seq)
                    neg_seqs.append(neg_seq)
                    seq_ids.append(peak.name)
            elif len(gset) == 0:
                not_in_gene += 1
            elif len(gset) > 1:
               multiple_genes += 1
        elif overlaps > 1 and valid:
            redundant += 1

    print " Found {} invalid peaks (too short or too long).".format(invalid)
    print " Found {} valid but redundant peaks.".format(redundant)
    print " Found {} non-redundant peaks that did not overlap any genes, and {} that overlapped multiple genes.".format(not_in_gene, multiple_genes)
    print " Found {} valid non-redundant peaks overlapping genes.".format(len(pos_seqs))

    return pos_seqs, neg_seqs, seq_ids


def main():
    args = parse_arguments()
    random.seed(args.random_seed)  # so this always returns the same results

    session_id = args.session  # can change this later

    pos_seqs, neg_seqs, seq_ids = produce_sequences(args.input_file, args.fasta_file, args.gtf_file, args.min_length, args.max_length, args.width, args.padding, args.compatible)

    outfile = args.out_file
    b_outfile = args.out_bkg_file

    with open(b_outfile, 'w') as b_out, open(outfile, 'w') as out:
        for peak_id, pos_seq, neg_seq in zip(seq_ids, pos_seqs, neg_seqs):
            b_out.write(">bg_" + str(peak_id) + "\n" + neg_seq + "\n")
            out.write(">" + str(peak_id) + "\n" + pos_seq + "\n")

    if args.compatible:
        train_b_graphprot_outfile = str(session_id) + ".train.graphprot.negatives.fa"
        train_graphprot_outfile = str(session_id) + ".train.graphprot.positives.fa"
        ls_b_graphprot_outfile = str(session_id) + ".ls.graphprot.negatives.fa"
        ls_graphprot_outfile = str(session_id) + ".ls.graphprot.positives.fa"

        with open(train_b_graphprot_outfile, 'w') as graph_train_b, open(train_graphprot_outfile, 'w') as graph_train, open(ls_b_graphprot_outfile, 'w') as graph_ls_b, open(ls_graphprot_outfile, 'w') as graph_ls:
            ls_peaks = int(len(pos_seqs)* 0.98)  # we optimize parameters for GraphProt on 2% of the data
            counter = 0
            for peak_id, pos_seq, neg_seq in zip(seq_ids, pos_seqs, neg_seqs):
                counter += 1
                if args.compatible:
                    if counter > ls_peaks:
                        graph_ls_b.write(">bg_" + str(peak_id) + "\n" + neg_seq + "\n")
                        graph_ls.write(">" + str(peak_id) + "\n" + pos_seq + "\n")
                    else:
                        graph_train_b.write(">bg_" + str(peak_id) + "\n" + neg_seq + "\n")
                        graph_train.write(">" + str(peak_id) + "\n" + pos_seq + "\n")

if __name__ == '__main__':
    main()
