"""
This is simplified version where we compare CDS to 3UTR.
We don't try to estimate the position of change/use HMM but rather to highlight interesting transcripts
"""
import numpy as np
import pandas as pd
from bx.bbi.bigwig_file import BigWigFile
from scipy.stats import norm
from scipy.stats import normaltest

from utils import read_known_genes

WINDOW_JUMPS = 20
WINDOW_SIZE = 50  # 35
WINDOW_PSEUDO_COUNT = 5
LOG_SCALE = True
MIN_STD = 0.01  # for fraction
SWAP_CDS_UTR = False


def get_exon_values_windows(transcript_values, exon_starts, exon_ends, window_jumps=WINDOW_JUMPS,
                            window_size=WINDOW_SIZE):
    _ex_windows_pos = [win for exStart, exEnd in zip(exon_starts, exon_ends)
                       for win in range(exStart, exEnd - window_size, window_jumps)]
    _ex_windows = np.concatenate([np.arange(win, win + window_size) for win in _ex_windows_pos])
    try:
        _ex_windows_values = transcript_values.values[:, _ex_windows].reshape(
            (transcript_values.shape[0], _ex_windows.shape[0] // window_size, window_size)).mean(-1)
        _ex_windows_values = pd.DataFrame(_ex_windows_values, index=transcript_values.index, columns=_ex_windows_pos)
    except:
        _ex_windows_values = transcript_values[:, _ex_windows].reshape(
            (transcript_values.shape[0], _ex_windows.shape[0] // window_size, window_size))
        _ex_windows_values = _ex_windows_values.mean(-1)
    return _ex_windows_values, _ex_windows_pos


def calc_fdr(p_vals, alpha=0.05):
    n_hypo = p_vals.shape[-1]
    correction = 1 + np.argmax((np.sort(p_vals, -1) <= alpha) * (np.arange(1, n_hypo + 1) / n_hypo), -1)
    p_vals_corrected = np.clip(p_vals / (correction / n_hypo)[:, np.newaxis], 0, 1)
    return p_vals_corrected, correction / n_hypo


def calc_fdr_for_all_transcripts(all_bw, sum_normal_by_total=False, inverse=False,
                                 p_calc=None, output_file=None):
    global SWAP_CDS_UTR

    if output_file is None:
        suffixes = np.array(['inserse', 'norm'])
        suffixes_selection = np.array([inverse, sum_normal_by_total])
        output_file = 'cds_vs_end_fdr{}.csv'.format(
            '_'.join(suffixes[suffixes_selection].tolist()) + '_%s' % ('log' if LOG_SCALE else 'nolog'))

    transcripts = read_known_genes('hg19')
    # skip transcripts with no cds
    transcripts = transcripts.loc[transcripts.cdsStart < transcripts.cdsEnd]
    # skip transcript in special chromosomes
    transcripts = transcripts.loc[~transcripts.chrom.isin([x for x in transcripts.chrom.unique() if '_' in x])]

    transcripts = transcripts.loc[np.where(transcripts.strand == '+',
                                           transcripts.cdsEnd < (
                                               transcripts.txEnd - transcripts.txStart - WINDOW_SIZE),
                                           (transcripts.cdsStart > WINDOW_SIZE))]

    all_transcripts_padded = transcripts[
        ['name', 'chrom', 'strand', 'txStart', 'txEnd', 'cdsStart', 'cdsEnd', 'exonStarts', 'exonEnds']]

    output_res_score = []

    for transcripts_key, transcripts_group in all_transcripts_padded.groupby(['chrom', 'txStart', 'txEnd']):
        chrom, tx_start, tx_end = transcripts_key
        transcript_raw_values = get_raw(all_bw, chrom, tx_start, tx_end)
        transcript_values = get_as_pd_simple(all_bw, chrom, tx_start, tx_end)

        for name, strand, cdsStart, cdsEnd, exon_starts, exon_ends in transcripts_group[
            ['name', 'strand', 'cdsStart', 'cdsEnd', 'exonStarts', 'exonEnds']].itertuples(index=False):
            ex_windows_values, ex_windows_pos = get_exon_values_windows(transcript_values, exon_starts, exon_ends)

            # skip transcripts with zero std /  no reads
            if np.any(np.std(ex_windows_values, 1) == 0):
                continue
            exon_descriptor = pd.concat(
                [transcript_raw_values[st:end] for st, end in zip(exon_starts, exon_ends)]).describe()

            ex_windows_values_cds = ex_windows_values.loc[:, (ex_windows_pos > cdsStart) & (ex_windows_pos < cdsEnd)]
            ex_windows_values_ncds = ex_windows_values.loc[:,
                                     (ex_windows_pos > cdsEnd) if strand == '+' else (ex_windows_pos < cdsStart)]
            # compute p-vals
            if ex_windows_values_ncds.shape[1] == 0:
                continue

            if SWAP_CDS_UTR:
                p_vals = pd.DataFrame([norm(end_vals.mean(), np.maximum(end_vals.std(), MIN_STD)).cdf(cds_vals)
                                       for cds_vals, end_vals in
                                       zip(ex_windows_values_cds.values, ex_windows_values_ncds.values)],
                                      index=ex_windows_values.index)
            else:
                p_vals = pd.DataFrame([norm(cds_vals.mean(), np.maximum(cds_vals.std(), MIN_STD)).cdf(end_vals)
                                       for cds_vals, end_vals in
                                       zip(ex_windows_values_cds.values, ex_windows_values_ncds.values)],
                                      index=ex_windows_values.index)

            if np.any(np.isnan(p_vals.values)):
                raise Exception

            # normality test
            try:
                norm_pvals = [normaltest(ex_windows_values_cds.loc[ex])[1] for ex in p_calc.keys()]
            except:
                norm_pvals = [-1 for ex in p_calc.keys()]

            p_vals = np.array([p_calc_func(p_vals.loc[ex_type]) for ex_type, p_calc_func in p_calc.items()])
            p_vals[np.isnan(p_vals)] = 1
            # p_vals = np.abs(p_calc[:, np.newaxis] - p_vals)
            p_fdr_vals_high, bh_threshold_high = calc_fdr(p_vals, 0.05)
            corrections_high = np.sum(p_fdr_vals_high <= 0.05, -1)
            p_fdr_vals_low, bh_threshold_low = calc_fdr(p_vals, 0.01)
            corrections_low = np.sum(p_fdr_vals_low <= 0.01, -1)

            tx_computed = [name] + corrections_high.tolist() + p_fdr_vals_high.min(
                -1).tolist() + corrections_low.tolist() + p_fdr_vals_low.min(-1).tolist() + norm_pvals + [
                              p_vals.shape[1]]
            tx_computed += exon_descriptor.values.tolist()
            output_res_score.append(tx_computed)

    output_res = pd.DataFrame(output_res_score)
    # full_output_res = pd.DataFrame(full_output_res)
    descriptior_cols = ['count', 'mean', 'std', '0%', '25%', '50%', '75%', '100%']
    output_res.columns = ['name'] + [format_t % (ex, criteria) for criteria in [5, 1] for format_t in
                                     ['%s_num_pass%ifdr', '%s_pvals_%ifdr'] for ex in p_calc.keys()] + \
                         ['%s_norm' % ex for ex in p_calc.keys()] + ['tail'] + descriptior_cols
    output_res_merged = pd.merge(output_res, transcripts, on='name')
    sort_value = '3prime_pvals_1fdr'
    if sort_value not in output_res.columns:
        sort_value = '%s_pvals_1fdr' % list(p_calc.keys())[0]
    output_res_merged[
        ['name', 'geneSymbol', 'description', 'chrom', 'txStart', 'txEnd'] +
        ['%s_pvals_%ifdr' % (ex, 1) for ex in p_calc.keys()]
        + ['%s_pvals_%ifdr' % (ex, 5) for ex in p_calc.keys()]
        + ['%s_num_pass%ifdr' % (ex, 1) for ex in p_calc.keys()]
        + ['%s_num_pass%ifdr' % (ex, 5) for ex in p_calc.keys()] + ['%s_norm' % ex for ex in p_calc.keys()]
        + ['tail'] + descriptior_cols].sort(sort_value).to_csv(output_file, index=False)


def get_as_pd_simple(bw_list, chrom, start, end):
    all_ex_datas = dict()
    ex_types = bw_list.keys()
    real_type = [k for k in ex_types if k != 'norm'][0]
    for ex_type in ex_types:
        tmp_arr = np.zeros(end - start, dtype=float)
        sum_arr = np.zeros(end - start, dtype=float)
        for bw_i, bw in enumerate(bw_list[ex_type]):
            sum_arr += np.nan_to_num(bw.get_as_array(chrom, start, end, tmp_arr))
        all_ex_datas[ex_type] = np.log2((sum_arr + WINDOW_PSEUDO_COUNT))
    if 'norm' in ex_types:
        data = all_ex_datas[real_type] - all_ex_datas['norm']
    else:
        data = all_ex_datas[real_type]

    return pd.DataFrame(np.array([data]), index=[[k for k in bw_list.keys() if 'norm' not in k][0]])


def get_raw(bw_list, chrom, start, end):
    ex_types = bw_list.keys()
    real_type = [k for k in ex_types if k != 'norm'][0]

    tmp_arr = np.zeros(end - start, dtype=float)
    sum_arr = np.zeros(end - start, dtype=float)
    for bw_i, bw in enumerate(bw_list[real_type]):
        sum_arr += np.nan_to_num(bw.get_as_array(chrom, start, end, tmp_arr))

    return pd.Series(sum_arr)


def main():
    global SWAP_CDS_UTR
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('name', help='input type')
    parser.add_argument('--fold_up', help='Whether we consider fold up or fold down',
                        action='store_true')
    parser.add_argument('--swap_cds', help='Whether UTR should be larger than cds (3prime) or smaller (TEX/3prime)',
                        action='store_false')
    parser.add_argument('-norm_type', help='substring appears in the norm files')

    parser.add_argument('file', help='bigwig input', nargs='+')

    args = parser.parse_args()
    if args.norm_type:
        all_bw_by_type = [BigWigFile(file=open(f, 'rb')) for f in args.file if args.norm_type not in f]
        norm_files = [BigWigFile(file=open(f, 'rb')) for f in args.file if args.norm_type in f]
        dict_experiments = {
            args.name: all_bw_by_type,
            'norm': norm_files
        }

    else:
        all_bw_by_type = [BigWigFile(file=open(f, 'rb')) for f in args.file]
        dict_experiments = {args.name: all_bw_by_type}
    out_file = args.name + '.csv'
    if args.fold_up:
        p_val_from_cdf = {args.name: lambda x: 1 - x}
    else:
        p_val_from_cdf = {args.name: lambda x: x}
    if not args.swap_cds:
        SWAP_CDS_UTR = True

    calc_fdr_for_all_transcripts(dict_experiments, p_calc=p_val_from_cdf, output_file=out_file)


if __name__ == '__main__':
    main()
