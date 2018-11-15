"""
What we have:
* Total RNA-Seq         ------\(poly A site)/--
* TEX                   -------\________
* Biotin capping 3'     ________/----------
* 5' pull down          ---------\__________

Here we fit of RNA-seq in experiments based on control, and use continuous Gaussian HMM to model the bias (fit error).

PS: This approach seems to more properly handle low coverage areas compared to evaluating the RATIO (experiment/Control)
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from bx.bbi.bigwig_file import BigWigFile
from scipy.stats import ttest_ind, ks_2samp, skew
from hmm_kit import HMMModel as hmm
from hmm_kit.bwiter import bw_iter_log
from utils import read_known_genes

WINDOW_JUMPS = 20
WINDOW_SIZE = 50  # 35
DISCRETE = False
DEBUG = False


def get_exon_values_windows(transcript_values, exon_starts, exon_ends, window_jumps=WINDOW_JUMPS,
                            window_size=WINDOW_SIZE):
    _ex_windows_pos = np.array([win for exStart, exEnd in zip(exon_starts, exon_ends)
                       for win in range(exStart, exEnd - window_size, window_jumps)])
    _ex_windows = np.concatenate([np.arange(win, win + window_size) for win in _ex_windows_pos])

    _ex_windows_values = transcript_values[_ex_windows].reshape(
        (_ex_windows.shape[0] // window_size, window_size))
    _ex_windows_values = _ex_windows_values.mean(-1)
    return _ex_windows_values, _ex_windows_pos


def calc_fdr(p_vals, alpha=0.05):
    n_hypo = p_vals.shape[-1]
    correction = 1 + np.argmax((np.sort(p_vals, -1) <= alpha) * (np.arange(1, n_hypo + 1) / n_hypo), -1)
    p_vals_corrected = np.clip(p_vals / (correction / n_hypo)[:, np.newaxis], 0, 1)
    return p_vals_corrected, correction / n_hypo


def hmm_frag_transcripts(all_bw, experiment, feature_extractors={}):
    global DISCRETE
    from sklearn.linear_model import LinearRegression
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
    init_emission = np.array([
        [1, 0],  # start
        [0.9, 0.1],  # body
        [0.1, 0.9]  # tail
    ])

    extra_features = feature_extractors.keys()
    skip_wrong_isoforms = True
    nan_annotation_mask = True

    for transcripts_key, transcripts_group in all_transcripts_padded.groupby(['chrom', 'txStart', 'txEnd']):
        chrom, tx_start, tx_end = transcripts_key
        # get the read coverage for both treat (experiment) and control (norm)
        exp_values = np.nansum([v.get_as_array(chrom, tx_start, tx_end) for v in all_bw[experiment]], 0)
        norm_values = np.nansum([v.get_as_array(chrom, tx_start, tx_end) for v in all_bw['norm']], 0)

        for name, strand, cdsStart, cdsEnd, exon_starts, exon_ends in transcripts_group[
            ['name', 'strand', 'cdsStart', 'cdsEnd', 'exonStarts', 'exonEnds']].itertuples(index=False):
            # filter to exonic regions, and transform to windows
            ex_windows_values_exp, ex_windows_pos = get_exon_values_windows(np.nan_to_num(exp_values), exon_starts,
                                                                            exon_ends)
            ex_windows_values_norm, _ = get_exon_values_windows(np.nan_to_num(norm_values), exon_starts,
                                                                exon_ends)
            # skip transcripts with zero std /  no reads
            if np.std(ex_windows_values_norm) == 0 or np.std(ex_windows_values_exp) == 0 or \
                            np.sum(ex_windows_values_exp > 0) < 2 or np.all(ex_windows_values_norm == 0):
                continue

            log_ex_windows_values_exp = np.log2(ex_windows_values_exp + 1)
            log_norm_windows_values = np.log2(ex_windows_values_norm + 1)

            if skip_wrong_isoforms and exon_starts.shape[0] > 1:
                exon_map = np.split(log_ex_windows_values_exp, np.where(
                    (np.array(ex_windows_pos)[:, np.newaxis] == exon_starts[np.newaxis, :]).any(1))[0][1:])
                # skip wrong transcripts: those for which the exons aren't transcribed
                exons_max_map = [exon.max() for exon in exon_map]
                if (norm(*norm.fit(exons_max_map)).pdf(exons_max_map) < 0.01).any():
                    print('Skipping %s - wrong isoform' % name)
                    continue

            UTR_selector = (ex_windows_pos > cdsEnd) if strand == '+' else (ex_windows_pos < cdsStart)
            # we could use coding selector but coding may be short or long
            lin_model = LinearRegression().fit(np.array([log_norm_windows_values]).T, log_ex_windows_values_exp)

            ex_windows_values = log_ex_windows_values_exp - np.maximum(
                lin_model.predict(np.array([log_norm_windows_values]).T), 0)

            ex_windows_values_cds = ex_windows_values[(ex_windows_pos > cdsStart) & (ex_windows_pos < cdsEnd)]
            ex_windows_values_ncds = ex_windows_values[UTR_selector]
            if nan_annotation_mask:
                # mask the start/end of transcripts
                base_threshold = ex_windows_values_norm > 0
                true_between = np.argwhere(base_threshold)[[0, -1], 0]

                true_between_diff = np.array(true_between, copy=True)

                if log_norm_windows_values[(ex_windows_pos > cdsStart) & (ex_windows_pos < cdsEnd)].std() > 0:
                    true_between_exons = \
                        np.argwhere(norm(*norm.fit(
                            log_norm_windows_values[(ex_windows_pos > cdsStart) & (ex_windows_pos < cdsEnd)])).cdf(
                            log_norm_windows_values) > 0.01)[
                            [0, -1], 0]
                    true_between_diff[0] = max(true_between_diff[0], true_between_exons[0])
                    true_between_diff[1] = min(true_between_diff[1], true_between_exons[1])
                real_read_mask = (np.arange(log_norm_windows_values.shape[0]) >= true_between_diff[0]) & \
                                 (np.arange(log_norm_windows_values.shape[0]) <= true_between_diff[-1])

            is_coding_expressed = np.any(real_read_mask & (ex_windows_pos > cdsStart) & (ex_windows_pos < cdsEnd))
            if skip_wrong_isoforms and not is_coding_expressed:
                print('Skipping - the coding region is not expressed')
                continue
            if strand == '+':
                exon_selector = ex_windows_pos > cdsStart
                if nan_annotation_mask:
                    exon_selector &= real_read_mask
                exon_data = ex_windows_values[exon_selector]
                exon_data_raw = {'exp': ex_windows_values_exp[exon_selector],
                                 'norm': ex_windows_values_norm[exon_selector],
                                 'ratio': np.log2(1 + ex_windows_values_exp[exon_selector]) - np.log2(
                                     1 + ex_windows_values_norm[exon_selector])}
            else:
                exon_selector = ex_windows_pos < cdsEnd
                if nan_annotation_mask:
                    exon_selector &= real_read_mask
                exon_data = ex_windows_values[exon_selector][::-1]
                exon_data_raw = {'exp': ex_windows_values_exp[exon_selector][::-1],
                                 'norm': ex_windows_values_norm[exon_selector][::-1],
                                 'ratio': (np.log2(1 + ex_windows_values_exp[exon_selector]) - np.log2(
                                     1 + ex_windows_values_norm[exon_selector]))[::-1]
                                 }

            coding_mean = np.mean(ex_windows_values_cds[ex_windows_values_cds != 0])
            if np.isnan(coding_mean):
                print('Skipping %s - all zeros' % name)
                continue
            cov_data = np.cov(exon_data)

            try:
                if DISCRETE:
                    hmm_model = hmm.DiscreteHMM(init_transitions, init_emission)
                    exon_data_discrete = np.array(exon_data > np.percentile(ex_windows_values_cds, 90), dtype=int)
                    new_model, model_likelihood = bw_iter_log(exon_data_discrete, hmm_model, stop_condition=3)
                    state_seq, path_likelihood = new_model.viterbi(exon_data_discrete, True)
                    p_body = new_model.emission[1, 1]
                    p_tail = new_model.emission[2, 1]
                else:
                    hmm_model = hmm.GaussianHMM(init_transitions,
                                                [(np.mean(ex_windows_values_cds), cov_data),
                                                 (np.mean(ex_windows_values_ncds), cov_data)
                                                 ],
                                                cov_sharing=[0, 0], cov_model='shared', min_std=cov_data)
                    new_model, model_likelihood = bw_iter_log(np.array([exon_data]), hmm_model, stop_condition=3)

                    fb_res = new_model.forward_backward_log(np.array([exon_data]), True)
                    p_tail_transition = np.exp(
                        fb_res.forward[:-1, 0] + np.log(hmm_model.state_transition[1, 2]) + np.log(
                            new_model.get_emission()[1:, np.array([exon_data])])[1, 1:] + fb_res.backward[1:,
                                                                                          1] - fb_res.model_p)
                    tail_posterior = fb_res.state_p[:, 1]
                    p_tail_transition_interval = (p_tail_transition > 0.05) & (p_tail_transition < 0.95)
                    p_tail_transition_interval = np.concatenate(
                        [[False], p_tail_transition_interval])  # add dummy False to have same dimension
                    tail_posterior_interval = (tail_posterior > 0.05) & (tail_posterior < 0.95)

                    state_seq, path_likelihood = new_model.viterbi(np.array([exon_data]), True)
                    p_body = np.mean(exon_data[state_seq == 0])
                    p_tail = np.mean(exon_data[state_seq == 1])
            except Exception as e:
                print('%s transcript failed - skipping, error:' % name)
                print(e)
                continue

            if not np.any(state_seq == 1) or np.all(state_seq == 1):
                # 1 state fits the transcript
                break_point = -1  # invalid
                ttest_pval = 1  # there is no two populations
                kstest_pval = 1
                extra_features = feature_extractors.keys()
                extra_features_vals = [0 for feature in extra_features
                                       for part in ['body', 'tail'] for d in [None, None, None]]
                in_threeUTR = False
                llr_test = -np.inf
                freq_significant_down = -1
                freq_significant_up = -1
                tail_interval_pos = np.array([-1])
            else:
                # the transcript is fitted by 2 states
                break_point = np.where(state_seq == 1)[0][0]
                ex_windows_pos = np.array(ex_windows_pos)

                _, kstest_pval = ks_2samp(exon_data[:break_point], exon_data[break_point:])
                _, ttest_pval = ttest_ind(exon_data[:break_point],
                                          exon_data[break_point:])  # maybe exclude after canonical breakpoint?
                null_likelihood = norm(*norm.fit(exon_data)).logpdf(exon_data).sum()
                llr_test = path_likelihood - null_likelihood
                extra_features_vals = [feature_extractors[feature](d[part]) for feature in extra_features
                                       # exon_data
                                       for part in [slice(break_point), slice(break_point, None)] for d in
                                       [exon_data_raw['exp'], exon_data_raw['norm'],
                                        exon_data_raw['ratio']]]  # exon_data

                norm_body = norm(*new_model.get_emission().mean_vars[0][0])
                # cdf
                freq_significant_down = (calc_fdr(norm_body.cdf(exon_data[break_point:]))[0] < 0.05).mean()
                # survival function
                freq_significant_up = (calc_fdr(norm_body.sf(exon_data[break_point:]))[0] < 0.05).mean()

                if strand == '+':
                    break_point = ex_windows_pos[exon_selector][break_point]
                    in_threeUTR = break_point > cdsEnd
                    transition_interval_pos = ex_windows_pos[exon_selector][p_tail_transition_interval]
                    tail_interval_pos = ex_windows_pos[exon_selector][tail_posterior_interval]
                else:
                    break_point = ex_windows_pos[exon_selector][-break_point]
                    in_threeUTR = break_point < cdsStart
                    transition_interval_pos = ex_windows_pos[exon_selector][::-1][p_tail_transition_interval]
                    tail_interval_pos = ex_windows_pos[exon_selector][::-1][tail_posterior_interval]

            if not p_tail_transition_interval.any():
                transition_interval_pos = np.array([-1])
            if not tail_posterior_interval.any():
                tail_interval_pos = np.array([-1])
            # break_point in genome = txStart + break_point
            tx_calced = [name, break_point, model_likelihood, path_likelihood, kstest_pval, ttest_pval, llr_test,
                         freq_significant_up, freq_significant_down, in_threeUTR, p_body,
                         p_tail, transition_interval_pos.min(), transition_interval_pos.max(), tail_interval_pos.min(),
                         tail_interval_pos.max()] + extra_features_vals
            output_res_score.append(tx_calced)
    extra_cols = ['%s_%s_%s' % (f, part, d) for f in extra_features for part in ['body', 'tail'] for d in
                  ['data', 'control', 'norm_data']]
    output_res_score = pd.DataFrame(output_res_score,
                                    columns=['name', 'bp', 'model_l', 'path_l', 'ks_test', 't_test', 'llr_test',
                                             'freq_down', 'freq_up',
                                             'bp in UTR', 'p1', 'p2',
                                             'AB_interval_min', 'AB_interval_max', 'B_interval_min',
                                             'B_interval_max'] + extra_cols)
    output_res_score = pd.merge(transcripts, output_res_score, on='name')
    break_point_columns = ['name', 'geneSymbol', 'chrom', 'strand', 'txStart', 'txEnd', 'bp', 'model_l', 'path_l',
                           'ks_test', 't_test', 'llr_test', 'freq_down', 'freq_up', 'bp in UTR', 'p1',
                           'p2', 'AB_interval_min', 'AB_interval_max', 'B_interval_min', 'B_interval_max'] + extra_cols
    output_res_score = output_res_score[break_point_columns]
    output_res_score['max_coverage'] = output_res_score[['max_body_data', 'max_tail_data']].values.max(1)
    output_res_score['is_good'] = 2 ** output_res_score['max_coverage'] > 100
    if os.path.exists(output_file):
        input('Are you sure you want to override %s? (Press CTRL+C if not)' % output_file)

    # interesting transcripts are those for which the transition is within the 3'UTR and significant
    output_res_score.sort_values(['bp in UTR', 'ks_test'], ascending=[False, True]).to_csv(output_file, index=False)
    # create bed file for visualization
    bed_data = output_res_score[['name', 'txStart', 'bp', 'chrom', 'strand', 'ks_test']]
    bed_data.rename(columns={'ks_test': 'score'}, inplace=True)
    bed_data['chromStart'] = (bed_data.txStart + bed_data.bp - 100).astype(int)
    bed_data['chromEnd'] = (bed_data.txStart + bed_data.bp + 100).astype(int)
    bed_data.loc[(bed_data.score > 0.05), 'score'] = 10
    bed_data.loc[(bed_data.score <= 0.05) & (bed_data.score > 0.01), 'score'] = 500
    bed_data.loc[(bed_data.score < 0.01), 'score'] = 999
    bed_data['score'] = np.where(output_res_score.ks_test < 0.01, 1000, 90)
    # output as bed file for visualization in UCSC genome browser
    with open(output_file.replace('.csv', '.bed'), 'w') as bed_fd:
        bed_fd.write('track name=%s description="hmm_frag_fit %s" useScore=1\n' % (experiment, experiment))
        bed_data[['chrom', 'chromStart', 'chromEnd', 'name', 'score', 'strand']].to_csv(bed_fd,
                                                                                        index=False, sep=' ',
                                                                                        header=False, mode='a')

    print('Finished %s' % experiment)


def main():
    global DISCRETE
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('name', help='Experiment name (input type)')
    parser.add_argument('-norm_type', help='Substring appears in the control files')
    parser.add_argument('-non_discrete', dest='discrete', action='store_false', help='use discrete of continuous HMM')

    parser.add_argument('file', help='bigwig files as input', nargs='+')
    args = parser.parse_args()
    DISCRETE = args.discrete
    print(
        'Input files:\n\t' + '\n\t'.join(
            [f for f in args.file if (args.norm_type is None) or not (args.norm_type in f)]))

    all_bw_by_type = [BigWigFile(file=open(f, 'rb')) for f in args.file if
                      (args.norm_type is None) or not (args.norm_type in f)]
    all_bw = {args.name: all_bw_by_type}

    if args.norm_type is not None:
        print('Norm files:\n\t' + '\n\t'.join([f for f in args.file if args.norm_type in f]))
        norm_bw = [BigWigFile(file=open(f, 'rb')) for f in args.file if args.norm_type in f]
        all_bw['norm'] = norm_bw

    print('Running fragmentation for all transcripts')
    feature_extractors = {
        'mean': np.mean,
        'std': np.std,
        'skewness': skew,
        'max': np.max,
        'count': lambda x: x.shape[0]
    }

    hmm_frag_transcripts(all_bw, args.name, feature_extractors=feature_extractors)


if __name__ == '__main__':
    main()
