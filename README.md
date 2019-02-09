This repository contains the source code for analysis for analysis of TEX, 5-CAP-PD and 3-PD experiments.

# Install
1. Install required python packages:
`pip install -r requirements.txt`
2. Get required metadata:
   * In directory data run: `get_data.sh`
   * Alternatively download manually the following files to data directory:
     * knownGene.hg19.txt.gz - http://hgdownload.cse.ucsc.edu/goldenpath/hg19/database/knownGene.txt.gz
     * kgXreg.hg19.txt.gz - http://hgdownload.cse.ucsc.edu/goldenpath/hg19/database/kgXref.txt.gz

# Usage
* hmm_frag_fit.py - Fits a 2-state Gaussian HMM to each transcript
  * usage: `hmm_frag_fit.py [-norm_type CONTROL] [-non_discrete] EXP_NAME bigwig1 [bigwig2 ...]`
    * EXP_NAME: Name of experiment (input type)
    * bigwigs: bigwig files of RNA-seq (both treated and control) as input
    * -norm_type NORM_TYPE: Substring appears in the control files
    * -non_discrete: Whether to use continuous HMM or discrete
  * Usage example: 
    ```
    python hmm_frag_fit.py U2OS_5prime_fit_hmm -non_discrete -norm_type Total data/*_TEX.bigwig data/*_Total.bigwig
    ```

## Supplementary scripts
* comapre_3seq.py - For comparing HMM points to peaks in 3'-seq
* cds_vs_tUTR.py - Compare CDS and 3UTR
* hmm_frag_multichannel.py - Similar to hmm_frag_fit.py but works on a multichannel
* create_illustration - illustrate the output based on the HMM output and the read data (bigwigs)
* scripts for generating debug tracks:
  * zscore_track.py
  * regression_fit_track.py

# Additonal infomation
* Malka, Y., Steiman-Shimony, A., Rosenthal, E., Argaman, L., Cohen-Daniel, L., Arbib, E., Margalit, H., Kaplan, T. & Berger, M. (2017). [Post-transcriptional 3´-UTR cleavage of mRNA transcripts generates thousands of stable uncapped autonomous RNA fragments](https://doi.org/10.1038/s41467-017-02099-7). Nature communications, 8(1), 2029.
* [GSE84068](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE84068) - The samples contain the original bigwig files used by the above paper.
