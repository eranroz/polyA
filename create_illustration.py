"""
This is a helper script to create nice plots of HMM cleavage points.
"""
from utils import read_known_genes


def plot_example(gene_name, bp_data, data, cont, label='TEX', downsample=200):
    import numpy as np
    from matplotlib import pyplot as plt

    plt.style.use('ggplot')
    plt.rcParams['font.family'] = 'Arial'

    bp_data = bp_data.loc[bp_data.name == gene_name].iloc[0]
    genes = read_known_genes('hg19', gene_name)

    genes = genes.iloc[0]
    data_chrom = np.nan_to_num(data.get_as_array(genes.chrom, genes.txStart, genes.txEnd))
    cont_chrom = np.nan_to_num(cont.get_as_array(genes.chrom, genes.txStart, genes.txEnd))
    normdata = np.log2(np.nan_to_num(data_chrom) + 1) - np.log2(np.nan_to_num(cont_chrom) + 1)

    pos = np.concatenate([np.arange(st, en) for st, en in zip(genes.exonStarts, genes.exonEnds)])

    cdPos = np.argmax(pos > genes.cdsStart) if genes.strand == '-' else np.argmax(pos > genes.cdsEnd)

    xx, yy = np.meshgrid(np.arange(len(pos)), np.arange(-np.abs(normdata).max(), np.abs(normdata).max()))
    bp_in_exons = np.argmax(pos >= bp_data.bp)

    bp_in_exons_smoothed = np.around(np.argmax(pos >= bp_data.bp) / downsample)
    smooted_norm = normdata[pos][:(len(pos) // downsample) * downsample].reshape((len(pos) // downsample),
                                                                                 downsample).mean(1)
    if genes.strand == '+':
        mean_body, std_body = np.mean(smooted_norm[:bp_in_exons_smoothed]), np.std(smooted_norm[:bp_in_exons_smoothed])
        mean_tail, std_tail = np.mean(smooted_norm[bp_in_exons_smoothed:]), np.std(smooted_norm[bp_in_exons_smoothed:])
    else:
        mean_body, std_body = np.mean(smooted_norm[bp_in_exons_smoothed:]), np.std(smooted_norm[bp_in_exons_smoothed:])
        mean_tail, std_tail = np.mean(smooted_norm[:bp_in_exons_smoothed]), np.std(smooted_norm[:bp_in_exons_smoothed])

    pdf_body = np.abs(
        yy[:, 0] - mean_body) / std_body
    pdf_tail = np.abs(
        yy[:, 0] - mean_tail) / std_tail
    if bp_data.strand == '+':
        zz = np.column_stack([np.repeat(pdf_body[:, np.newaxis], np.argmax(bp_data.bp < pos), axis=1),
                              np.repeat(pdf_tail[:, np.newaxis], len(pos) - np.argmax(bp_data.bp < pos), axis=1)])
    else:
        zz = np.column_stack([np.repeat(pdf_tail[:, np.newaxis], np.argmax(bp_data.bp < pos), axis=1),
                              np.repeat(pdf_body[:, np.newaxis], len(pos) - np.argmax(bp_data.bp < pos), axis=1)])

    from matplotlib import gridspec
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 4, 4])
    # f, axarr = plt.subplots(2, sharex=True)
    axarr = plt.subplot(gs[0]), plt.subplot(gs[1]), plt.subplot(gs[2])
    axarr[0].set_title(
        '%s %s(%s:%i-%i)' % (bp_data.geneSymbol, bp_data.strand, bp_data.chrom, bp_data.txStart, bp_data.txEnd))
    # cds are heavier
    cdsPos = np.arange(np.argmax(pos >= genes.cdsStart), np.argmax(pos >= genes.cdsEnd))
    nonCds = np.concatenate(
        [np.arange(0, np.argmax(pos >= genes.cdsStart)), np.arange(np.argmax(pos >= genes.cdsEnd), pos.max() + 1)])
    axarr[0].plot(np.arange(0, len(pos)), np.ones_like(pos), linewidth=10, c='#000099')
    axarr[0].plot(cdsPos, np.ones_like(cdsPos), linewidth=20, c='#000099')
    axarr[0].set_yticks([])
    axarr[0].set_xticks([])
    axarr[0].patch.set_visible(False)

    if genes.strand == '+':
        axarr[0].annotate('5\' ', (0, 1), (10, 20), textcoords='offset points')
        axarr[0].annotate('3\' ', (pos.max(), 1), (-10, 20), textcoords='offset points')
    else:
        axarr[0].annotate('3\' ', (0, 1), (10, 20), textcoords='offset points')
        axarr[0].annotate('5\' ', (pos.max(), 1), (-10, 20), textcoords='offset points'),
        # data
    axarr[1].plot(np.arange(0, (len(pos) // downsample) * downsample, downsample),
                  np.log2(data_chrom[pos][:(len(pos) // downsample) * downsample].reshape(
                      (len(pos) // downsample),
                      downsample).mean(
                      1)) + 1, label=label, c='#009900', linewidth=3)
    axarr[1].plot(np.arange(0, (len(pos) // downsample) * downsample, downsample),
                  np.log2(cont_chrom[pos][:(len(pos) // downsample) * downsample].reshape(
                      (len(pos) // downsample),
                      downsample).mean(
                      1) + 1), label='Control', c='#000099', linewidth=3)

    axarr[1].legend(loc='upper left', frameon=False, fontsize=10)
    axarr[1].set_ylabel('# $log_2$')
    axarr[1].patch.set_visible(False)
    axarr[1].set_xticks([])
    # ratio
    axarr[2].patch.set_visible(False)
    axarr[2].contourf(xx, yy, zz, levels=np.arange(6), alpha=0.3, cmap=plt.cm.Blues_r)
    axarr[2].plot(np.arange(0, (len(pos) // downsample) * downsample, downsample),
                  normdata[pos][:(len(pos) // downsample) * downsample].reshape((len(pos) // downsample),
                                                                                downsample).mean(
                      1), linewidth=5, label='%s/Control' % label)
    axarr[2].legend(loc='upper left', fontsize=10, frameon=False)
    axarr[2].set_ylabel('')
    axarr[2].set_ylabel('# $log_2$')
    axarr[2].patch.set_visible(False)
    axarr[2].set_xticks([])


def main():
    import argparse
    import pandas as pd
    from bx.bbi.bigwig_file import BigWigFile

    parser = argparse.ArgumentParser()
    parser.add_argument('gene')
    parser.add_argument('cell', default='293')
    parser.add_argument('exp', default='TEX')
    parser.add_argument('transcript', help='Transcript name')
    parser.add_argument('cont_bigwig', help='Path to control bigwig')
    parser.add_argument('treat_bigwig', help='Path to treated experiment bigwig')
    parser.add_argument('hmm_output_file', help='Path to HMM output file')

    parser.add_argument('--downsample', default=200)
    args = parser.parse_args()

    cont = BigWigFile(open(args.cont_bigwig, 'rb'))
    data = BigWigFile(open(args.treat_bigwig, 'rb'))
    bp_data = pd.read_csv(args.hmm_output_file)

    plot_example(args.transcript, bp_data=bp_data, data=data, cont=cont, label=args.exp,
                 downsample=args.downsample)


if __name__ == '__main__':
    main()
