# -*- coding: utf-8 -*-

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

def read_fasta_file(path, min_length=1, max_length=100):
    seqgen = SeqIO.parse(open(path), 'fasta')
    seqs = []
    names = []
    for s in seqgen:
        seq = str(s.seq)
        if 'n' not in seq.lower() and len(seq) >= min_length and len(seq) <= max_length:
            seqs.append(seq)
            names.append(s.id)
    return seqs, names

def read_long_fasta_file(path):
    seqgen = SeqIO.parse(open(path), 'fasta')
    seqs = []
    names = []
    for s in seqgen:
        if 'n' not in str(s.seq).lower():
            seqs.append(str(s.seq))
            names.append(s.id)
    return seqs, names


def read_fasta_files(paths, min_length, max_length):
    all_seqs = []
    all_names = []

    for path in paths:
        seqs, names = read_fasta_file(path, min_length, max_length)
        all_seqs.extend(seqs)
        all_names.extend(names)

    return all_seqs, all_names

def write_fasta_file(path, seqs, ids):
    records = [SeqRecord(Seq(s), id=i, description="") for i,s in zip(ids, seqs)]
    SeqIO.write(records, path, "fasta")
