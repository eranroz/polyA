"""
This script calculates the zscore for each transcript, for log(data)-log(norm_data) or for log(data)
if no norm data is provided.

data = mean of all input bigwig files which doesn't contain norm in their names
norm_data = mean of all input bigwig files which contain norm in their names
"""

import numpy as np
import pandas as pd
from bx.bbi.bigwig_file import BigWigFile

from cds_vs_tUTR import WINDOW_PSEUDO_COUNT
from utils import read_known_genes

WINDOW_SIZE = 50  # 35
LOG_SCALE = True


def generate_bigwig_all(bw, exp, chrom_sizes, mean_sel='cds'):
    global CHROM_SIZES_PATH

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
    for chrom, chrom_size in chrom_sizes.itertuples(False):
        print(chrom)
        # to avoid memory overflow
        chrom_data = np.concatenate(
            [win_arr(get_as_arrays_simple(bw, chrom, i - 100000, i)) for i in range(100000, chrom_size, 100000)] + [
                win_arr(get_as_arrays_simple(bw, chrom, chrom_size - chrom_size % 100000, chrom_size))])
        exp_dict[chrom] = chrom_data

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
    z_score_dict = dict()
    count_dict = dict()
    for chrom in exp_dict.keys():
        z_score_dict[chrom] = np.zeros_like(exp_dict[chrom])
        count_dict[chrom] = np.zeros_like(exp_dict[chrom])

    all_transcripts_padded = all_transcripts_padded.loc[all_transcripts_padded.chrom.isin(exp_dict.keys())]
    print('going over transcripts')
    for chrom, cdsStart, cdsEnd, txStart, txEnd, strand, exonSt, exonEn in all_transcripts_padded[
        ['chrom', 'cdsStart', 'cdsEnd', 'txStart', 'txEnd', 'strand',
         'exonStarts', 'exonEnds']].itertuples(False):
        # ex_windows_pos = np.arange(txStart, txEnd)

        exon_wins = np.concatenate([np.arange(st, en, dtype=int) for st, en in zip(exonSt, exonEn)])

        if mean_sel == 'cds':
            cds_pos = [np.arange(st, en, dtype=int) for st, en in
                       zip(exonSt, exonEn) if en < cdsEnd and st > cdsStart]
            if len(cds_pos) == 0:
                continue
            cds_pos = np.concatenate(cds_pos)
            cds_vals = exp_dict[chrom][cds_pos]
            mean_val, std_val = np.mean(cds_vals), np.std(cds_vals)
        else:
            mean_val, std_val = np.mean(exp_dict[chrom][exon_wins]), np.std(exp_dict[chrom][exon_wins])
        if std_val == 0:
            continue  # ignore transcripts where with zero variance - not real/not expressed
        std_val = max(std_val, 0.1)
        tx_score = (exp_dict[chrom][exon_wins] - mean_val) / std_val

        z_score_dict[chrom][exon_wins] += tx_score
        count_dict[chrom][exon_wins] += 1
    print('end transcripts')
    # write to bedgraph
    all_z_score = []
    for chrom, chrom_vals in z_score_dict.items():
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
        '%s.bedgraph' % (exp),
        index=False, sep=' ', header=False)
    print('create %s.bedgraph' % (exp))


RPKM_NORM = dict()


def get_as_arrays_simple(bw_list, chrom, start, end, include_raw=False, pseudocount=WINDOW_PSEUDO_COUNT):
    global RPKM_NORM
    all_ex_datas = dict()
    ex_types = bw_list.keys()
    real_type = [k for k in ex_types if k != 'norm'][0]
    for ex_type in ex_types:
        tmp_arr = np.zeros(end - start, dtype=float)
        sum_arr = np.zeros(end - start, dtype=float)
        for bw_i, bw in enumerate(bw_list[ex_type]):
            sum_arr += np.nan_to_num(bw.get_as_array(chrom, start, end, tmp_arr))
        if ex_type not in RPKM_NORM:
            RPKM_NORM[ex_type] = 1
        all_ex_datas[ex_type] = sum_arr
    if 'norm' in ex_types:
        data = np.log2(all_ex_datas[real_type] + pseudocount) - np.log2(all_ex_datas['norm'] + pseudocount)
    else:
        data = np.log2(all_ex_datas[real_type] + pseudocount)
    if include_raw:
        return data, all_ex_datas
    return data


def calc_rpkm_files(files, chrom_sizes):
    sum_reads = 0
    for f in files:
        for chrom, chrom_size in chrom_sizes.itertuples(False):
            bw_cont = f.get(chrom, 0, chrom_size)
            if bw_cont is None: continue
            _, _, reads = zip(*iter(bw_cont))
            sum_reads += sum(reads)
    return 1e9 / sum_reads


def main():
    global RPKM_NORM
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('name', help='input type')
    parser.add_argument('cromsize_path', help='Chromosome sizes file')
    parser.add_argument('-norm_type', help='substring appears in the norm files')
    parser.add_argument('-rpkm', help='rpkm')
    parser.add_argument('-mean_type', choices=['cds', 'exons'], default='exons')

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
    if args.rpkm:
        RPKM_NORM[args.name] = calc_rpkm_files(all_bw[args.name], chrom_sizes)
        RPKM_NORM['norm'] = calc_rpkm_files(all_bw['norm'], chrom_sizes)
        print(RPKM_NORM[args.name])
    else:
        RPKM_NORM[args.name] = 1
        RPKM_NORM['norm'] = 1
    print('Calcing zscore')

    generate_bigwig_all(all_bw, args.name, chrom_sizes, mean_sel=args.mean_type)


if __name__ == '__main__':
    main()
