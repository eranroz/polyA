"""
This script is illustrates the regression fit error.
This script calculates the regression fit error for each transcript, for log(data)-log(norm_data) or for log(data)
if no norm data is provided.

data = mean of all input bigwig files which doesn't contain norm in their names
norm_data = mean of all input bigwig files which contain norm in their names
"""

import numpy as np
import pandas as pd
from bx.bbi.bigwig_file import BigWigFile
from utils import read_known_genes

WINDOW_SIZE = 50

LOG_SCALE = True


def generate_bigwig_all(all_bw, exp, chrom_sizes):
    from sklearn.linear_model import LinearRegression

    def win_arr(arr):
        try:
            arr = np.nan_to_num(arr)
        except:
            print(arr)
            raise
        return np.mean(
            arr[:(arr.shape[0] // WINDOW_SIZE) * WINDOW_SIZE].reshape(arr.shape[0] // WINDOW_SIZE, WINDOW_SIZE), -1)

    transcripts = read_known_genes('hg19')
    # skip transcript in special chromosomes
    transcripts = transcripts.loc[~transcripts.chrom.isin([x for x in transcripts.chrom.unique() if '_' in x])]
    transcripts = transcripts.loc[transcripts.cdsStart < transcripts.cdsEnd]

    chrom_sizes = chrom_sizes.loc[chrom_sizes.chrom.isin(transcripts.chrom.unique())]

    # create sum bigwig
    exp_dict = dict()
    norm_dict = dict()
    for chrom, chrom_size in chrom_sizes.itertuples(False):
        print(chrom)
        # to avoid memory overflow
        chrom_exp_data = np.concatenate(
            [win_arr(np.nansum([v.get_as_array(chrom, i - 100000, i) for v in all_bw[exp]], 0)) for i in
             range(100000, chrom_size, 100000)] + [
                win_arr(np.nansum(
                    [v.get_as_array(chrom, chrom_size - chrom_size % 100000, chrom_size) for v in all_bw[exp]], 0))])
        chrom_exp_data = np.log2(chrom_exp_data + 1)

        chrom_norm_data = np.concatenate(
            [win_arr(np.nansum([v.get_as_array(chrom, i - 100000, i) for v in all_bw['norm']], 0)) for i in
             range(100000, chrom_size, 100000)] + [
                win_arr(np.nansum(
                    [v.get_as_array(chrom, chrom_size - chrom_size % 100000, chrom_size) for v in all_bw['norm']], 0))])
        chrom_norm_data = np.log2(chrom_norm_data + 1)

        norm_dict[chrom] = chrom_norm_data
        exp_dict[chrom] = chrom_exp_data

    # compute z-score
    # skip transcripts with no cds
    all_transcripts_padded = transcripts[
        ['name', 'chrom', 'strand', 'txStart', 'txEnd', 'cdsStart', 'cdsEnd', 'exonStarts', 'exonEnds']]
    all_transcripts_padded.cdsStart += all_transcripts_padded.txStart
    all_transcripts_padded.cdsEnd += all_transcripts_padded.txStart
    all_transcripts_padded.exonStarts += all_transcripts_padded.txStart
    all_transcripts_padded.exonEnds += all_transcripts_padded.txStart
    all_transcripts_padded[['exonStarts', 'exonEnds', 'txStart', 'txEnd', 'cdsStart', 'cdsEnd']] /= WINDOW_SIZE
    all_transcripts_padded.txStart = all_transcripts_padded.txStart.astype(int)
    all_transcripts_padded.txEnd = all_transcripts_padded.txEnd.astype(int)

    print('%s zscore' % exp)
    fit_err_score_dict = dict()
    count_dict = dict()
    for chrom in exp_dict.keys():
        fit_err_score_dict[chrom] = np.zeros_like(exp_dict[chrom])
        count_dict[chrom] = np.zeros_like(exp_dict[chrom])

    all_transcripts_padded = all_transcripts_padded.loc[all_transcripts_padded.chrom.isin(exp_dict.keys())]
    print('going over transcripts')
    for chrom, cdsStart, cdsEnd, txStart, txEnd, strand, exonSt, exonEn in all_transcripts_padded[
        ['chrom', 'cdsStart', 'cdsEnd', 'txStart', 'txEnd', 'strand',
         'exonStarts', 'exonEnds']].itertuples(False):
        # ex_windows_pos = np.arange(txStart, txEnd)

        exon_wins = np.concatenate([np.arange(st, en, dtype=int) for st, en in zip(exonSt, exonEn)])

        cds_pos = [np.arange(st, en, dtype=int) for st, en in
                   zip(exonSt, exonEn) if en < cdsEnd and st > cdsStart]
        if len(cds_pos) == 0:
            continue

        lin_model = LinearRegression().fit(np.array([norm_dict[chrom][exon_wins]]).T, exp_dict[chrom][exon_wins])
        # hinge diff
        tx_score = exp_dict[chrom][exon_wins] - np.maximum(lin_model.predict(np.array([norm_dict[chrom][exon_wins]]).T),
                                                           0)

        fit_err_score_dict[chrom][exon_wins] += tx_score
        count_dict[chrom][exon_wins] += 1
    print('end transcripts')
    # write to bedgraph
    all_z_score = []
    for chrom, chrom_vals in fit_err_score_dict.items():
        chrom_vals /= count_dict[chrom]
        chrom_vals = np.nan_to_num(chrom_vals)

        chrom_vals = pd.DataFrame(np.array([chrom_vals[chrom_vals != 0], np.where(chrom_vals != 0)[0]]).T)
        chrom_vals.columns = ['score', 'chromStart']
        chrom_vals['chromStart'] *= WINDOW_SIZE
        chrom_vals['chromEnd'] = chrom_vals.chromStart + WINDOW_SIZE
        chrom_vals['chrom'] = chrom
        all_z_score.append(chrom_vals)
    all_z_score = pd.concat(all_z_score)
    all_z_score.chromStart = all_z_score.chromStart.astype(int)
    all_z_score.chromEnd = all_z_score.chromEnd.astype(int)
    all_z_score[['chrom', 'chromStart', 'chromEnd', 'score']].to_csv(
        '%s_fit.bedgraph' % (exp),
        index=False, sep=' ', header=False)
    print('create %s_fit.bedgraph' % (exp))


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('name', help='input type')
    parser.add_argument('cromsize_path', help='Chromosome sizes file')
    parser.add_argument('-norm_type', help='substring appears in the norm files')
    parser.add_argument('file', help='bigwig input', nargs='+')
    args = parser.parse_args()
    print('Input files:\n' + '\n'.join([f for f in args.file if (args.norm_type is None) or not (args.norm_type in f)]))

    all_bw_by_type = [BigWigFile(file=open(f, 'rb')) for f in args.file if
                      (args.norm_type is None) or not (args.norm_type in f)]
    all_bw = {args.name: all_bw_by_type}

    if args.norm_type is not None:
        print('Norm files:\n' + '\n'.join([f for f in args.file if args.norm_type in f]))

        norm_bw = [BigWigFile(file=open(f, 'rb')) for f in args.file if args.norm_type in f]
        all_bw['norm'] = norm_bw

    chrom_sizes = pd.read_csv(args.cromsize_path, sep='\t', names=['chrom', 'size'])
    generate_bigwig_all(all_bw, args.name, chrom_sizes)


if __name__ == '__main__':
    main()
