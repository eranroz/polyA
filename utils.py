import numpy as np
import pandas as pd


def read_known_genes(genome, relative_positions=True):
    """Reads annotation files derived from UCSC
    :param genome: Genome version (such as hg19, hg38)
    :param relative_positions: Whether exon positions are relative to TSS or chromosomal positions
    :return: dataframe of transcripts
    """
    column_names = ['name', 'chrom', 'strand', 'txStart', 'txEnd', 'cdsStart', 'cdsEnd', 'exonCount', 'exonStarts',
                    'exonEnds']
    _transcripts = pd.read_csv('data/knownGene.%s.txt.gz' % genome, sep='\t', compression='gzip',
                               names=column_names, header=None, usecols=list(range(len(column_names))))

    kg_xref = pd.read_csv('data/kgXreg.%s.txt.gz' % genome, sep='\t', compression='gzip',
                          names=['name', 'geneSymbol', 'description'], header=None, usecols=[0, 4, 7])
    _transcripts = pd.merge(_transcripts, kg_xref, on='name', how='left')

    _transcripts.exonStarts = _transcripts.exonStarts.apply(lambda x: np.array(x.rstrip(',').split(','), dtype=int))
    _transcripts.exonEnds = _transcripts.exonEnds.apply(lambda x: np.array(x.rstrip(',').split(','), dtype=int))

    # relative to TSS
    if relative_positions:
        _transcripts.exonStarts = _transcripts.exonStarts - _transcripts.txStart
        _transcripts.exonEnds = _transcripts.exonEnds - _transcripts.txStart

        _transcripts.cdsStart -= _transcripts.txStart
        _transcripts.cdsEnd -= _transcripts.txStart

    return _transcripts
