"""
What we have:
* Total RNA-Seq         ------\(poly A site)/--
* TEX                   -------\________
* Biotin capping 3'     ________/----------
* 5' pull down          ---------\__________

Sample transcripts:
uc002iqt.3
uc010wmk.1
"""
import os

import hmm_kit.HMMModel as hmm
import numpy as np
import pandas as pd
from bx.bbi.bigwig_file import BigWigFile
from hmm_kit.bwiter import bw_iter_log
from scipy.stats import multivariate_normal
from scipy.stats import norm
from scipy.stats import ttest_ind, skew

from cds_vs_tUTR import WINDOW_PSEUDO_COUNT
from utils import read_known_genes

WINDOW_JUMPS = 20
WINDOW_SIZE = 50  # 35

DEBUG = False


def get_exon_values_windows(transcript_values, exon_starts, exon_ends, window_jumps=WINDOW_JUMPS,
                            window_size=WINDOW_SIZE):
    _ex_windows_pos = [win for exStart, exEnd in zip(exon_starts, exon_ends)
                       for win in range(exStart, exEnd - window_size, window_jumps)]
    _ex_windows = np.concatenate([np.arange(win, win + window_size) for win in _ex_windows_pos])

    _ex_windows_values = transcript_values[_ex_windows].reshape(
        (_ex_windows.shape[0] // window_size, window_size))
    _ex_windows_values = _ex_windows_values.mean(-1)
    return _ex_windows_values, _ex_windows_pos


def calc_fdr(p_vals, alpha=0.05):
    # fdr = p_vals * len(p_vals) / rankdata(p_vals)
    # fdr[fdr > 1] = 1
    n_hypo = p_vals.shape[-1]
    correction = 1 + np.argmax((np.sort(p_vals, -1) <= alpha) * (np.arange(1, n_hypo + 1) / n_hypo), -1)
    p_vals_corrected = np.clip(p_vals / (correction / n_hypo)[:, np.newaxis], 0, 1)
    return p_vals_corrected, correction / n_hypo


def hmm_frag_transcripts(all_bw, experiment, feature_extractors={}):
    print('Started %s' % experiment)

    output_file = 'breakpoint_score_{}.csv'.format(experiment)

    transcripts = read_known_genes('hg19')
    # skip transcripts with no cds
    transcripts = transcripts.loc[transcripts.cdsStart < transcripts.cdsEnd]

    # skip transcripts with no 3UTR
    transcripts = transcripts.loc[np.where(transcripts.strand == '+',
                                           transcripts.cdsEnd < (
                                               transcripts.txEnd - transcripts.txStart - 2 * WINDOW_SIZE),
                                           (transcripts.cdsStart > 2 * WINDOW_SIZE))]

    # skip transcript in special chromosomes
    transcripts = transcripts.loc[~transcripts.chrom.isin([x for x in transcripts.chrom.unique() if '_' in x])]

    all_transcripts_padded = transcripts[
        ['name', 'chrom', 'strand', 'txStart', 'txEnd', 'cdsStart', 'cdsEnd', 'exonStarts', 'exonEnds']]

    output_res_score = []

    init_transitions = np.array([
        [0, 0.5, 0.5],
        [0, 0.99, 0.01],
        [0, 0, 1]
    ])

    extra_features = feature_extractors.keys()
    skip_wrong_isoforms = True
    nan_annotation_mask = True
    real_types = [experiment for experiment in all_bw.keys() if experiment != 'norm']
    for transcripts_key, transcripts_group in all_transcripts_padded.groupby(['chrom', 'txStart', 'txEnd']):
        chrom, tx_start, tx_end = transcripts_key

        all_ex_datas, raw_ex_datas = dict(), dict()

        for exp_type, exp_list in all_bw.items():
            tmp_arr = np.zeros(tx_end - tx_start, dtype=float)
            sum_arr = np.zeros(tx_end - tx_start, dtype=float)
            for bw_i, bw in enumerate(exp_list):
                sum_arr += np.nan_to_num(bw.get_as_array(chrom, tx_start, tx_end, tmp_arr))
            raw_ex_datas[exp_type] = sum_arr
            all_ex_datas[exp_type] = np.log2((sum_arr + WINDOW_PSEUDO_COUNT))
        if 'norm' in all_bw.keys():
            all_ex_datas = dict([(ex_type, all_ex_datas[ex_type] - all_ex_datas['norm']) for ex_type in real_types
                                 if ex_type != 'norm'])

        for name, strand, cdsStart, cdsEnd, exon_starts, exon_ends in transcripts_group[
            ['name', 'strand', 'cdsStart', 'cdsEnd', 'exonStarts', 'exonEnds']].itertuples(index=False):
            ex_windows_values, ex_windows_pos = zip(*iter([get_exon_values_windows(all_ex_datas[exp_type],
                                                                                   exon_starts, exon_ends)
                                                           for exp_type in real_types]))
            ex_windows_values = np.array(ex_windows_values)
            ex_windows_pos = np.array(ex_windows_pos[0])
            # skip transcripts with zero std /  no reads
            if np.any(np.std(ex_windows_values, -1) == 0):
                continue

            ex_windows_values_cds = np.compress((ex_windows_pos > cdsStart) & (ex_windows_pos < cdsEnd),
                                                ex_windows_values, -1)

            UTR_selector = (ex_windows_pos > cdsEnd) if strand == '+' else (ex_windows_pos < cdsStart)

            if (np.sum(UTR_selector) < 2) or np.all(ex_windows_values_cds) == 0:
                continue

            ex_windows_values_exp = np.array([get_exon_values_windows(raw_ex_datas[exp], exon_starts, exon_ends)[0]
                                              for exp in real_types])
            ex_windows_values_norm = get_exon_values_windows(raw_ex_datas['norm'], exon_starts, exon_ends)[0]
            if np.any(np.sum(ex_windows_values_exp > 0, 1) < 2):
                continue

            if skip_wrong_isoforms and exon_starts.shape[0] > 1:
                exon_map = np.split(ex_windows_values_norm, np.where(
                    (np.array(ex_windows_pos)[:, np.newaxis] == exon_starts[np.newaxis, :]).any(1))[0][1:])
                # skip wrong transcripts: those for which the exons aren't transcribed
                exons_max_map = [np.log1p(exon.max()) for exon in exon_map]
                if (norm(*norm.fit(exons_max_map)).pdf(exons_max_map) < 0.01).any():
                    print('Skipping %s - wrong isoform' % name)
                    continue

            ex_windows_values_ncds = np.compress(UTR_selector, ex_windows_values, -1)
            if nan_annotation_mask:

                # mask the start/end of transcripts
                base_threshold = ex_windows_values_norm > 0
                true_between = np.argwhere(base_threshold)[[0, -1], 0]


                true_between_diff = np.array(true_between, copy=True)

                if ex_windows_values_norm[(ex_windows_pos > cdsStart) & (ex_windows_pos < cdsEnd)].std() > 0:
                    true_between_exons = \
                        np.argwhere(norm(*norm.fit(
                            ex_windows_values_norm[(ex_windows_pos > cdsStart) & (ex_windows_pos < cdsEnd)])).cdf(
                            ex_windows_values_norm) > 0.01)[
                            [0, -1], 0]
                    true_between_diff[0] = max(true_between_diff[0], true_between_exons[0])
                    true_between_diff[1] = min(true_between_diff[1], true_between_exons[1])
                real_read_mask = (np.arange(ex_windows_values_norm.shape[0]) >= true_between_diff[0]) & \
                                 (np.arange(ex_windows_values_norm.shape[0]) <= true_between_diff[-1])

            is_coding_expresssed = np.any(real_read_mask & (ex_windows_pos > cdsStart) & (ex_windows_pos < cdsEnd))
            if skip_wrong_isoforms and not is_coding_expresssed:
                print('Skipping - the coding region is not expressed')
                continue
            if strand == '+':
                exon_selector = ex_windows_pos > cdsStart
            else:
                exon_selector = ex_windows_pos < cdsEnd

            if nan_annotation_mask:
                exon_selector &= real_read_mask
            exon_data = np.compress(exon_selector, ex_windows_values, axis=-1)
            exon_data_raw = np.compress(exon_selector, ex_windows_values_exp, axis=-1)

            if strand == '-':
                exon_data = np.fliplr(exon_data)
                exon_data_raw = np.fliplr(exon_data_raw)

            coding_mean = np.mean(ex_windows_values_cds[ex_windows_values_cds != 0])
            if np.isnan(coding_mean):
                print('Skipping %s - all zeros' % name)
                continue
            cov_data = np.cov(exon_data)
            if exon_data.ndim != 2:
                exon_data = np.array([exon_data])
            try:
                hmm_model = hmm.GaussianHMM(init_transitions,
                                            [(np.mean(ex_windows_values_cds, axis=-1), cov_data),
                                             # (coding_mean + 2 * cov_data, cov_data)
                                             (np.mean(ex_windows_values_ncds, axis=-1), cov_data)
                                             ],
                                            cov_sharing=(0, 0), cov_model='shared', min_std=cov_data)

                new_model, model_likelihood = bw_iter_log(exon_data, hmm_model, stop_condition=3)
                # model_likelihood = 0
                # new_model = hmm_model
                state_seq, path_likelihood = new_model.viterbi(exon_data, True)
                p_body = np.compress(state_seq == 0, exon_data, axis=-1).mean(axis=-1)
                p_tail = np.compress(state_seq == 1, exon_data, axis=-1).mean(
                    axis=-1)
            except:
                print('%s failed to baum Welsch' % name)
                continue

            if not np.any(state_seq == 1) or np.all(state_seq == 1):
                break_point = -1  # invalid
                ttest_pval = 1  # there is no two populations
                kstest_pval = 1
                extra_features = feature_extractors.keys()
                extra_features_vals = [0 for feature in extra_features
                                       for part in ['body', 'tail']]
                in_threeUTR = False
                llr_test = -np.inf
                freq_significant_down = -1
                freq_significant_up = -1
            else:
                break_point = np.where(state_seq == 1)[0][0]

                ttest_pval = ttest_ind(*np.split(exon_data, [break_point], axis=-1), axis=1)[1].max()
                null_likelihood = multivariate_normal.logpdf(exon_data.T, exon_data.mean(axis=-1),
                                                             np.cov(exon_data)).sum()
                llr_test = path_likelihood - null_likelihood
                extra_features_vals = [feature_extractors[feature](exon_data_raw[exp_i][part]) for feature in
                                       extra_features
                                       # exon_data
                                       for part in [slice(break_point), slice(break_point, None)] for exp_i, exp in enumerate(real_types)]

                # cdf
                # see also Hotelling's T-squared?
                data_min_mean = exon_data-exon_data.mean(1)[:, np.newaxis]
                mahalanobolis_distance = (np.dot(data_min_mean.T, np.cov(exon_data))*data_min_mean.T).sum(1)
                norm_body = norm(*norm.fit(mahalanobolis_distance[break_point:]))
                freq_significant_down = (calc_fdr(norm_body.cdf([mahalanobolis_distance[break_point:]]))[0] < 0.05).mean()
                freq_significant_up = (calc_fdr(norm_body.sf([mahalanobolis_distance[break_point:]]))[0] < 0.05).mean()

                if strand == '+':
                    break_point = ex_windows_pos[exon_selector][break_point]
                    in_threeUTR = break_point > cdsEnd
                else:
                    break_point = ex_windows_pos[exon_selector][-break_point]
                    in_threeUTR = break_point < cdsStart

            # break_point in genome = txStart + break_point
            tx_calced = [name, break_point, model_likelihood, path_likelihood, ttest_pval, llr_test,
                         freq_significant_up, freq_significant_down, in_threeUTR, p_body,
                         p_tail] + extra_features_vals
            output_res_score.append(tx_calced)
    extra_cols = ['%s_%s_%s' % (f, part, exp) for f in extra_features for part in ['body', 'tail'] for exp in
                  real_types]
    output_res_score = pd.DataFrame(output_res_score,
                                    columns=['name', 'bp', 'model_l', 'path_l', 't_test', 'llr_test',
                                             'freq_down', 'freq_up',
                                             'bp in UTR', 'p1', 'p2'] + extra_cols)
    output_res_score = pd.merge(transcripts, output_res_score, on='name')
    break_point_columns = ['name', 'geneSymbol', 'chrom', 'strand', 'txStart', 'txEnd', 'bp', 'model_l', 'path_l',
                           't_test', 'llr_test', 'freq_down', 'freq_up', 'bp in UTR', 'p1',
                           'p2'] + extra_cols
    output_res_score = output_res_score[break_point_columns]

    if os.path.exists(output_file):
        input('Are you sure you want to override %s? (Press CTRL+C if not)' % output_file)
    output_res_score.sort(['bp in UTR'], ascending=[True]).to_csv(output_file, index=False)
    # create bed file for visualization
    bed_data = output_res_score[['name', 'txStart', 'bp', 'chrom', 'strand', 'llr_test']]
    bed_data.rename(columns={'llr_test': 'score'}, inplace=True)
    bed_data['chromStart'] = (bed_data.txStart + bed_data.bp - 100).astype(int)
    bed_data['chromEnd'] = (bed_data.txStart + bed_data.bp + 100).astype(int)
    bed_data.loc[(bed_data.score > 0.05), 'score'] = 10
    bed_data.loc[(bed_data.score <= 0.05) & (bed_data.score > 0.01), 'score'] = 500
    bed_data.loc[(bed_data.score < 0.01), 'score'] = 999
    bed_data['score'] = 1  # np.where(output_res_score.llr_test < 0.01, 1000, 90)

    with open(output_file.replace('.csv', '.bed'), 'w') as bed_fd:
        bed_fd.write('track name=%s description="hmm_frag_polya %s" useScore=1\n' % (experiment, experiment))
        bed_data[['chrom', 'chromStart', 'chromEnd', 'name', 'score', 'strand']].to_csv(bed_fd,
                                                                                        index=False, sep=' ',
                                                                                        header=False, mode='a')

    print('Finished %s' % experiment)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('name', help='input type')
    parser.add_argument('-norm_type', help='substring appears in the norm files')

    parser.add_argument('file', help='bigwig input', nargs='+')
    parser.add_argument('-groups', help='How to group to channels', nargs='+')
    parser.add_argument('--data_dir', help='Directory to loop for input files', default='')
    args = parser.parse_args()
    files = [os.path.join(args.data_dir, f) for f in args.file]
    print(
        'Input files:\n\t' + '\n'.join([f for f in files if (args.norm_type is None) or not (args.norm_type in f)]))

    if args.groups:
        all_bw_by_type = dict((g, [BigWigFile(file=open(f, 'rb'))
                                   for f in files if g in f and args.norm_type not in f]) for g in args.groups)
    else:
        all_bw_by_type = {args.name: [BigWigFile(file=open(f, 'rb'))
                                      for f in files if
                                      (args.norm_type is None) or not (args.norm_type in f)]}

    if args.norm_type is not None:
        print('Norm files:\n\t' + '\n\t'.join([f for f in files if args.norm_type in f]))
        norm_bw = [BigWigFile(file=open(f, 'rb')) for f in files if args.norm_type in f]
        all_bw_by_type['norm'] = norm_bw

    print('Running fragmentation for all transcripts')
    feature_extractors = {
        'mean': np.mean,
        'std': np.std,
        'skewness': skew,
        'max': np.max,
        'count': lambda x: x.shape[0]
    }

    hmm_frag_transcripts(all_bw_by_type, args.name, feature_extractors=feature_extractors)


if __name__ == '__main__':
    main()
