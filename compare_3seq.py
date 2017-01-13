import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

GENOME = 'hg19'
TRANSCRIPTS_HMM_OUTPUT_FILE = ''  # for example: ''breakpoint_score_U2OS_TEX_fit_hmm.csv'


def histogram_dist_hmm_alphapeaks(peaks):
    tex_data = pd.read_csv(TRANSCRIPTS_HMM_OUTPUT_FILE)
    tex_data = tex_data.loc[tex_data['bp in UTR']]
    tex_data.bp = tex_data.bp + tex_data.txStart  # absolute values

    # add UCSC name in instead of refseq name
    knownToRefSeq = pd.read_csv('knownToRefSeq.hg19.txt.gz', sep='\t', compression='gzip',
                                names=['name', 'refSeqName'])
    peaks = pd.merge(peaks, knownToRefSeq, left_on='closest_transcript', right_on='refSeqName')
    peaks_hmm = pd.merge(peaks, tex_data, left_on='name', right_on='name')
    peaks_hmm['dist_hmm_peak_start'] = peaks_hmm.bp - peaks_hmm.start
    peaks_hmm['dist_hmm_peak_end'] = peaks_hmm.bp - peaks_hmm.end
    peaks_hmm['dist_hmm_peak'] = np.where(
        np.abs(peaks_hmm['dist_hmm_peak_start']) < np.abs(peaks_hmm['dist_hmm_peak_end']),
        peaks_hmm['dist_hmm_peak_start'], peaks_hmm['dist_hmm_peak_end'])
    peaks_hmm.loc[(peaks_hmm.bp < peaks_hmm.end) & (peaks_hmm.bp > peaks_hmm.start), 'dist_hmm_peak'] = 0
    min_dist_idx = peaks_hmm.groupby('name', as_index=False).apply(lambda x: x.dist_hmm_peak.abs().idxmin())
    argmin_peak = peaks_hmm.ix[min_dist_idx]
    xticks = np.arange(0, 5000, 100)
    y = np.array([(argmin_peak.dist_hmm_peak <= i).sum() for i in xticks])

    print(['dist', 'vals'])
    print(np.array([xticks, y]).T)

    plt.figure()
    _ = plt.hist(argmin_peak.dist_hmm_peak, bins=np.arange(-1000, 1000, 50))
    plt.title('Distance between HMM point and 3\'seq cleavage point (%i transcripts; bp in 3UTR)' %
              argmin_peak.shape[0])

    plt.savefig('dist_hmm_bp_from_threeseq.pdf')


def read_refseq(genome):
    ref_genes = pd.read_csv('data/refGene.%s.txt' % genome, sep='\t')
    ref_genes.exonStarts = ref_genes.exonStarts.apply(lambda x: np.array(x.rstrip(',').split(','), dtype=int))
    ref_genes.exonEnds = ref_genes.exonEnds.apply(lambda x: np.array(x.rstrip(',').split(','), dtype=int))
    ref_genes['mRnaId'] = ref_genes.name
    ref_genes = ref_genes.loc[
        (ref_genes.cdsStart != ref_genes.cdsEnd)]  # filter out such transcripts e.g use only transcripts with coding
    return ref_genes


def read_bw_peak_counts():
    contBwData = pd.read_csv('data/alphaAmanitine/ContPeaks.tab', sep='\t',
                             names=['name', 'sizeCont', 'coveredCont', 'sumCont', 'mean0Cont', 'meanCont', 'minCont',
                                    'maxCont'], index_col='name')
    contTexBwData = pd.read_csv('data/alphaAmanitine/contTexPeaks.tab', sep='\t',
                                names=['name', 'sizeContTex', 'coveredContTex', 'sumContTex', 'mean0ContTex',
                                       'meanContTex', 'minContTex', 'maxContTex'], index_col='name')
    alphaTexBwData = pd.read_csv('data/alphaAmanitine/aAmanitineTex.tab', sep='\t',
                                 names=['name', 'sizeAmanitineTex', 'coveredAmanitineTex', 'sumAmanitineTex',
                                        'mean0AmanitineTex', 'meanAmanitineTex', 'minAmanitineTex', 'maxAmanitineTex'],
                                 index_col='name')
    alphaBwData = pd.read_csv('data/alphaAmanitine/aAmanitinePeaks.tab', sep='\t',
                              names=['name', 'sizeAmanitine', 'coveredAmanitine', 'sumAmanitine', 'mean0Amanitine',
                                     'meanAmanitine', 'minAmanitine', 'maxAmanitine'], index_col='name')
    return pd.concat([contBwData, contTexBwData, alphaTexBwData, alphaBwData], axis=1)


peaks = pd.read_csv('data/peaks_exp.txt', sep='\t')
peaks = peaks.loc[peaks.distance == 0]  # get rid of peaks out of known transcripts annotations

# combine all the replications
peaks['Cont'] = peaks[['1-YMC_S1', '2-YMC_S2', '3-YMC_S3']].sum(1)
peaks['aAmanitine'] = peaks[['4-YMC_S4', '5-YMC_S5', '6-YMC_S6']].sum(1)
peaks['contTex'] = peaks[['7-YMC_S7', '8-YMC_S8', '9-YMC_S9']].sum(1)
peaks['aAmanitineTex'] = peaks[['10-YMC_S10', '11-YMC_S11', '12-YMC_S12']].sum(1)
peaks.drop(['1-YMC_S1', '2-YMC_S2', '3-YMC_S3', '4-YMC_S4', '5-YMC_S5', '6-YMC_S6', '7-YMC_S7',
            '8-YMC_S8', '9-YMC_S9', '10-YMC_S10', '11-YMC_S11', '12-YMC_S12'], axis=1, inplace=True)

# for peak with overlapping transcripts - split to multiple rows, each for each transcript
split_by_col = 'closest_transcript'  # 'associated_gene'
splittedp = peaks[split_by_col].str.split(';').apply(pd.Series, 1).stack()
splittedp.index = splittedp.index.droplevel(-1)
splittedp.name = split_by_col
del peaks[split_by_col]
peaks = peaks.join(splittedp)

peaks = peaks.groupby('closest_transcript').filter(lambda g: len(g) > 1)
histogram_dist_hmm_alphapeaks(peaks)  # histogram of distances (delta HMM and 3'seq)

transcripts = read_refseq('hg19')  # read_known_genes('hg19')


peaks_metadata = pd.merge(transcripts, peaks, left_on=['mRnaId', 'strand'],
                          right_on=['closest_transcript', 'strand'])  # add metadata
print('Missing transcripts (those are probably less stable and were dropped from latest UCSC DB): ')
print('%i/%i' % (len(set(peaks.closest_transcript.unique().tolist()) - set(peaks_metadata.mRnaId.unique().tolist())),
                 len(set(peaks.closest_transcript.unique().tolist()))))
del peaks_metadata['mRnaId']
print('Number of peaks (with all possible transcripts): %i' % peaks_metadata.shape[0])
# get rid of peaks that doesn't overlap  the transcript
peaks_metadata = peaks_metadata.loc[(peaks_metadata.end > peaks_metadata.txStart) &
                                    (peaks_metadata.start < peaks_metadata.txEnd)]

print('Filtering peaks:\n\t%i - peaks after removing peaks not in annotated transcripts' % peaks_metadata.shape[0])

# annotation for posing Pol II?
peaks_metadata['in5UTR'] = np.where(peaks_metadata.strand == '+', peaks_metadata.start < peaks_metadata.cdsStart,
                                    peaks_metadata.end > peaks_metadata.cdsEnd)
peaks_metadata = peaks_metadata.loc[~peaks_metadata.in5UTR]
print('\t%i - peaks after removing peaks in 5UTR (RNA PolII Posing)' % peaks_metadata.shape[0])
peaks_metadata = peaks_metadata.groupby('closest_transcript').filter(lambda g: len(g) > 1)
print('\t%i - peaks after removing peaks in which the transcript has only 1 peak' % peaks_metadata.shape[0])

# it may start before and end after. so we use conservative definition
peaks_metadata['in3UTR'] = np.where(peaks_metadata.strand == '+', peaks_metadata.start > peaks_metadata.cdsEnd,
                                    peaks_metadata.end < peaks_metadata.cdsStart)
group_by_analysis = 'closest_transcript'  # associated_gene
selected_cols = ['closest_transcript', 'associated_gene', 'chrom', 'strand', 'txStart', 'txEnd', 'peak_id', 'start',
                 'end', 'Cont', 'aAmanitine', 'contTex', 'aAmanitineTex']
peaks_metadata = pd.DataFrame(peaks_metadata.loc[peaks_metadata['in3UTR'], selected_cols])
print('\t%i - peaks after selecting only peaks from 3UTR' % peaks_metadata.shape[0])
peaks_metadata['peak_rank'] = peaks_metadata.groupby(group_by_analysis)['Cont'].transform(
    lambda x: x.rank(method='first', ascending=False))
peaks_metadata['peak_pos_rank'] = 0
peaks_metadata.loc[peaks_metadata.strand == '+', 'peak_pos_rank'] = \
    peaks_metadata.loc[peaks_metadata.strand == '+'].groupby(group_by_analysis)['start'].transform(
        lambda x: x.rank(ascending=False))
peaks_metadata.loc[peaks_metadata.strand == '-', 'peak_pos_rank'] = \
    peaks_metadata.loc[peaks_metadata.strand == '-'].groupby(group_by_analysis)['start'].transform(
        lambda x: x.rank(ascending=True))

peaks_metadata['Distal'] = peaks_metadata.peak_pos_rank == 1
peaks_metadata['Proximal'] = peaks_metadata.groupby('closest_transcript')['peak_pos_rank'].transform(
    lambda x: x.idxmax()) == peaks_metadata.index

# all the transcripts for which we have both distal and proximal in the 3UTR
prixmal_distal_by_pos = peaks_metadata.loc[peaks_metadata.Distal | peaks_metadata.Proximal].groupby(
    'closest_transcript').filter(lambda g: len(g) == 2).sort('closest_transcript')
print('\t%i - peaks after selecting only those with at least 2 peaks in the 3UTR' % prixmal_distal_by_pos.shape[0])

prixmal_distal_by_pos = pd.merge(prixmal_distal_by_pos.loc[prixmal_distal_by_pos.Distal],
                                 prixmal_distal_by_pos.loc[prixmal_distal_by_pos.Proximal],
                                 on=['closest_transcript', 'associated_gene', 'strand', 'chrom', 'txStart', 'txEnd'],
                                 suffixes=('Distal', 'Proxy'))

peaks_bw_counts = read_bw_peak_counts()

prixmal_distal_by_pos = pd.merge(prixmal_distal_by_pos, peaks_bw_counts[
    ['maxCont', 'maxContTex', 'maxAmanitine', 'maxAmanitineTex', 'meanCont', 'meanContTex', 'meanAmanitine',
     'meanAmanitineTex']], left_on='peak_idProxy', right_index='name')
prixmal_distal_by_pos = pd.merge(prixmal_distal_by_pos, peaks_bw_counts[
['maxCont', 'maxContTex', 'maxAmanitine', 'maxAmanitineTex', 'meanCont', 'meanContTex', 'meanAmanitine',
     'meanAmanitineTex']], left_on='peak_idDistal', right_index='name', suffixes=('Proxy', 'Distal'))
del prixmal_distal_by_pos['DistalProxy']
del prixmal_distal_by_pos['ProximalProxy']


# ProxiTex     DistalTex            ProxiCont       DistalCont          a/b                 CanonicalEnrichment
# a*body     a*[Canonical]      b*body     b*[Canonical+Tail]      ProxiTex/ProxiCont

# 1-([Canonical]/[body]) / ([tail+canonical]/[body]) = [tail]/[tail+Canonical]

# ([Canonical]+[body]/[body]) / ([tail+canonical]+[body]/[body]) =  [Canonical]+[body]/[tail+canonical]+[body]

prixmal_distal_by_pos[
    'DistalProxyRatioTex'] = prixmal_distal_by_pos.maxContTexDistal / prixmal_distal_by_pos.maxContTexProxy
prixmal_distal_by_pos[
    'DistalProxyRatioControl'] = prixmal_distal_by_pos.maxContDistal / prixmal_distal_by_pos.maxContProxy
prixmal_distal_by_pos[
    'DistalProxyRatioTexAlpah'] = prixmal_distal_by_pos.maxAmanitineTexDistal / prixmal_distal_by_pos.maxAmanitineTexProxy
prixmal_distal_by_pos[
    'DistalProxyRatioAlpha'] = prixmal_distal_by_pos.maxAmanitineDistal / prixmal_distal_by_pos.maxAmanitineProxy

prixmal_distal_by_pos['TailEnrichment'] = 1 - (
    (prixmal_distal_by_pos.maxContTexDistal / prixmal_distal_by_pos.maxContTexProxy) / (
        prixmal_distal_by_pos.maxContDistal / prixmal_distal_by_pos.maxContProxy))
prixmal_distal_by_pos['TailAlphaEnrichment'] = 1 - (
    (prixmal_distal_by_pos.maxAmanitineTexDistal / prixmal_distal_by_pos.maxAmanitineTexProxy) / (
        prixmal_distal_by_pos.maxAmanitineDistal / prixmal_distal_by_pos.maxAmanitineProxy))

prixmal_distal_by_pos['TailEnrichmentPseudo'] = 1 - (((
                                                          prixmal_distal_by_pos.maxContTexDistal + prixmal_distal_by_pos.maxContTexProxy) / (
                                                          prixmal_distal_by_pos.maxContTexProxy)) / ((
                                                                                                         prixmal_distal_by_pos.maxContDistal + prixmal_distal_by_pos.maxContProxy) / (
                                                                                                         prixmal_distal_by_pos.maxContProxy)))
prixmal_distal_by_pos['TailAlphaEnrichmentPseudo'] = 1 - (((
                                                               prixmal_distal_by_pos.maxAmanitineTexProxy + prixmal_distal_by_pos.maxAmanitineTexDistal) / (
                                                               prixmal_distal_by_pos.maxAmanitineTexProxy)) / ((
                                                                                                                   prixmal_distal_by_pos.maxAmanitineDistal + prixmal_distal_by_pos.maxAmanitineProxy) / (
                                                                                                                   prixmal_distal_by_pos.maxAmanitineProxy)))

from scipy.stats import ttest_rel

# drop -inf
ttest_pval = \
    ttest_rel(
        *prixmal_distal_by_pos[['TailEnrichment', 'TailAlphaEnrichment']].replace(-np.inf, np.nan).dropna().values.T)[
        1]

prixmal_distal_by_pos[['closest_transcript', 'associated_gene', 'chrom', 'startDistal',
                       'endDistal', 'startProxy', 'endProxy', 'maxContTexDistal', 'maxContTexProxy', 'maxContDistal',
                       'maxContProxy',
                       'maxAmanitineDistal', 'maxAmanitineProxy', 'maxAmanitineTexDistal', 'maxAmanitineTexProxy',
                       'TailEnrichment', 'TailAlphaEnrichment']].rename(columns={
    'closest_transcript': 'transcript', 'associated_gene': 'gene',
    'maxContTexDistal': 'Distal_TEX',
    'maxContDistal': 'Distal_Cont',
    'maxContProxy': 'Prox_Cont',
    'maxContTexProxy': 'Prox_TEX',
    'maxAmanitineDistal': 'Distal_aAmanitine',
    'maxAmanitineProxy': 'Proxy_aAmanitine',
    'maxAmanitineTexDistal': 'Distal_TEX_aAmanitine',
    'maxAmanitineTexProxy': 'Proxy_TEX_aAmanitine'
}).to_csv('proximalDistalNew.csv', float_format='%.3f', index=False,
          sep='\t')

plt.hist([prixmal_distal_by_pos['TailEnrichment'].replace(-np.inf, np.nan).dropna().values,
          prixmal_distal_by_pos['TailAlphaEnrichment'].replace(-np.inf, np.nan).dropna().values],
         bins=np.arange(-3, 1, 0.1))
plt.clf()
plt.title(
    'Post transcriptional cleavage: $\\alpha$-Amanitine incubation increase the affect of TEX on proximal peak enrichment (p=%.2e)' % (
        ttest_rel(*prixmal_distal_by_pos[['TailEnrichment', 'TailAlphaEnrichment']].replace(-np.inf,
                                                                                            np.nan).dropna().values.T)[
            1]))
plt.hist(prixmal_distal_by_pos['TailEnrichment'].replace(-np.inf, np.nan).dropna().values, bins=np.arange(-3, 1.1, 0.1))
plt.hist(prixmal_distal_by_pos['TailAlphaEnrichment'].replace(-np.inf, np.nan).dropna().values,
         bins=np.arange(-3, 1.1, 0.1))
plt.xlabel('Tail enrichment')
plt.ylabel('#Transcripts')
plt.legend(['TEX - $\mu$ = %.3f' % prixmal_distal_by_pos['TailEnrichment'].replace([-np.inf], np.nan).dropna().mean(),
            '$\\alpha$Amanitine - $\mu$ = %.3f' % prixmal_distal_by_pos['TailAlphaEnrichment'].replace([-np.inf],
                                                                                                       np.nan).dropna().mean()])


plt.style.use('ggplot')

xxx = prixmal_distal_by_pos[
    ['DistalProxyRatioControl', 'DistalProxyRatioTex', 'DistalProxyRatioAlpha', 'DistalProxyRatioTexAlpah']].replace(
    np.inf, np.nan).dropna()

plt.boxplot(
    [xxx['DistalProxyRatioControl'].values, xxx['DistalProxyRatioTex'].values,
     xxx['DistalProxyRatioAlpha'].values, xxx['DistalProxyRatioTexAlpah'].values], showfliers=False, notch=True)

plt.xticks(np.arange(4) + 1, ['Control', 'TEX', 'alpha', 'Alpha+TEX'])
plt.ylabel('Distal/Proximal')
plt.legend()

prixmal_distal_by_pos['CanonicalEnrichment'] = (
                                                   prixmal_distal_by_pos.contTexDistal / prixmal_distal_by_pos.ContDistal) * (
                                                   prixmal_distal_by_pos.ContProxy / prixmal_distal_by_pos.maxContProxy)
# (DistalTex/DistalCont)*(b/a) = ([Canonical]/[Canonical+Tail])
plt.clf()
plt.title('$\frac{Canonical}{Cnonical+Tail}$')
plt.hist(((prixmal_distal_by_pos.contTexDistal / prixmal_distal_by_pos.ContDistal) * (
    prixmal_distal_by_pos.ContProxy / prixmal_distal_by_pos.contTexProxy)).dropna().values, bins=np.arange(0, 2, 0.1))

#

xContProxy = np.array([prixmal_distal_by_pos.ContProxy.values, np.ones_like(prixmal_distal_by_pos.ContProxy.values)]).T
xContDistal = np.array(
    [prixmal_distal_by_pos.ContDistal.values, np.ones_like(prixmal_distal_by_pos.ContDistal.values)]).T
cont_to_tex_proxy = np.linalg.lstsq(xContProxy, prixmal_distal_by_pos.contTexProxy.values)[0]
cont_to_tex_distal = np.linalg.lstsq(xContDistal, prixmal_distal_by_pos.contTexDistal.values)[0]

plt.clf()
plt.scatter(prixmal_distal_by_pos.ContProxy.values, prixmal_distal_by_pos.contTexProxy.values, c='red', s=3)
plt.scatter(prixmal_distal_by_pos.ContDistal.values, prixmal_distal_by_pos.contTexDistal.values, c='blue', s=3)

plt.plot(np.arange(10, 500), np.dot(cont_to_tex_proxy, np.array([np.arange(10, 500), np.ones(490)])), c='red')
plt.plot(np.arange(10, 500), np.dot(cont_to_tex_distal, np.array([np.arange(10, 500), np.ones(490)])), c='blue')

plt.legend(['Proxy', 'Distal'])
plt.xlabel('Control')
plt.ylabel('TEX')
plt.xlim((10, 500))
plt.ylim((10, 500))
