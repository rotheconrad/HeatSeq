# HeatSeq: cluster munitions for microbial genomics

![HeatSeq Logo](https://github.com/rotheconrad/HeatSeq/blob/main/images/HeatSeq_logo.png)


Quickly and easily predict and visualize distance based clusters for all vs. all ANI, AAI, Mash distance and others.

This tool uses hierarchical clustering to generate distance based cluster predictions for genomic (ANI, AAI) and metagenomic data (beta distance, MASH, Simka), or any other all vs. all distance type matrix, and it builds a clustered heatmap with column labels to highlight predicted cluster assignments and other user provided metadata. It outputs a seaborn clustermap in vector based PDF format along with separate PDF legend corresponding to each metadata row. It also outputs a metadata tsv file with the predicted cluster assignments for each cluster threshold for each genome, metagenome, or sample along with a separate tsv file specifying the color assignments. These output files may be modified by the user and input back into the code to customize the final figure output.

# Table of Contents

1. [Usage](#usage)
	1. [Case One: Default](#case-one-default)
	1. [Case Two: User Defined](#case-two-user-defined)
	1. [Case Three: Custom Metadata](#case-three-custom-metadata)
1. [Examples](#examples)
	1. [fastANI](#fastANI)
	1. [ANI](#ani)
	1. [AAI](#aai)
	1. [Mash](#mash)
	1. [Simka](#simka)
	1. [Distance](#distance)
1. [Software Dependencies](#software-dependencies)
1. [How to Cite](#how-to-cite)
1. [Future Improvements](#future-improvements)

![Final Figure Example](https://github.com/rotheconrad/HeatSeq/blob/main/images/clustermap_example.png)

# Usage

Their are three initial options:

1. Predict clusters for local minimumns in the distance distribution (default).
2. Predict clusters for user defined values (-user).
    e.g. -user 95 97.5 98.5 99.5
3. Skip cluster prediction and use user defined metadata.
    e.g. -m meta_data.tsv -c meta_colors.tsv

And several additional parameters:

1. -type, Specify data type. One of fastANI (default), ANI, AAI, Mash, Simka, Distance
	1. fastANI (default) - this takes the longform fastANI output and parses it into a distance matrix.
	1. ANI - this is for a square all vs. all matrix of ANI values with 100's down the diagonal.
	1. AAI - this is for a square all vs. all matrix of AAI values with 100's down the diagonal.
	1. Mash - this is for a Mash distance matrix.
	1. Simka - this is for a Simka distance matrix.
	1. Distance - this is any square distance matrix with values 0-1.
1. -dmin, Specify the minimum value for colors in the figure.
1. -dmax, Specify the maximum value for colors in the figure.
1. -metric, set different cluster metrics. choose from:
	1. euclidean, seuclidean, correlation, hamming, jaccard, braycurtis
	1. See webpage/docs for [scipy.spatial.distance.pdist](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html) for additional options and more info.
1. -method, set different cluster method choose from:
    1. single, complete, average, weighted, centroid, median, ward
    1. See webpage/docs for [scipy.cluster.hierarchy.linkage](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html) for more info.
1. -no, Create clustermap of the data without cluster prediction or metadata.

```bash
# print the help menu
python HeatSeq.py -h
```

## Case One: Default

```bash
python HeatSeq.py -i -o
```

## Case Two: User Defined

```bash
python HeatSeq.py -i -o -user
```

## Case Three: Custom Metadata

```bash
python HeatSeq.py -i -o -m -c
```

([Return to Table of Contents](#table-of-contents))

# Examples

## fastANI

```bash
python HeatSeq.py -i -o
```

## ANI

```bash
python HeatSeq.py -i -o
```

## AAI

```bash
python HeatSeq.py -i -o
```

## Mash

```bash
python HeatSeq.py -i -o
```

## Simka

```bash
python HeatSeq.py -i -o
```

## Distance

```bash
python HeatSeq.py -i -o
```

([Return to Table of Contents](#table-of-contents))

# Software Dependencies

## External dependencies

- [FastANI](https://github.com/ParBLiSS/FastANI)
- [Prodigal](https://github.com/hyattpd/Prodigal)
- [BLAST+](https://blast.ncbi.nlm.nih.gov/Blast.cgi?PAGE_TYPE=BlastDocs&DOC_TYPE=Download)
- [Diamond]()
- [MMseqs2](https://github.com/soedinglab/MMseqs2)
- [EggNog-mapper](https://github.com/eggnogdb/eggnog-mapper)
- [COGclassifier](https://github.com/moshi4/COGclassifier/) (optional alternative to EggNog Mapper)
- [Python](https://www.python.org/) version 3.6+ (for all custom code in this workflow)

#### References

1. Jain C, Rodriguez-R LM, Phillippy AM, Konstantinidis KT, Aluru S. High throughput ANI analysis of 90K prokaryotic genomes reveals clear species boundaries. Nature communications. 2018 Nov 30;9(1):1-8.
1. Hyatt D, Chen GL, LoCascio PF, Land ML, Larimer FW, Hauser LJ. Prodigal: prokaryotic gene recognition and translation initiation site identification. BMC bioinformatics. 2010 Dec;11(1):1-1.
1. Camacho C, Coulouris G, Avagyan V, Ma N, Papadopoulos J, Bealer K, Madden TL. BLAST+: architecture and applications. BMC bioinformatics. 2009 Dec;10(1):1-9.
1. Buchfink B, Reuter K, Drost HG. Sensitive protein alignments at tree-of-life scale using DIAMOND. Nature methods. 2021 Apr;18(4):366-8.
1. Steinegger M, Söding J. MMseqs2 enables sensitive protein sequence searching for the analysis of massive data sets. Nature biotechnology. 2017 Nov;35(11):1026-8.
1. Cantalapiedra CP, Hernández-Plaza A, Letunic I, Bork P, Huerta-Cepas J. eggNOG-mapper v2: functional annotation, orthology assignments, and domain prediction at the metagenomic scale. Molecular biology and evolution. 2021 Dec 1;38(12):5825-9.
1. Huerta-Cepas J, Szklarczyk D, Heller D, Hernández-Plaza A, Forslund SK, Cook H, Mende DR, Letunic I, Rattei T, Jensen LJ, von Mering C. eggNOG 5.0: a hierarchical, functionally and phylogenetically annotated orthology resource based on 5090 organisms and 2502 viruses. Nucleic acids research. 2019 Jan 8;47(D1):D309-14.
1. Sanner MF. Python: a programming language for software integration and development. J Mol Graph Model. 1999 Feb 1;17(1):57-61.

## Required packages for Python

- [pandas](https://pandas.pydata.org/) 
- [numpy](https://numpy.org/)
- [scipy](https://scipy.org/)
- [statsmodels](https://www.statsmodels.org)
- [matplotlib](https://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)
- [lmfit](https://lmfit.github.io/lmfit-py/)
- [datashader](https://datashader.org/)
- [pygam](https://pygam.readthedocs.io/)

*Python and all packages can be easily installed with conda or pip. Prodigal, BLAST+ and MMseqs2 can also be installed easily with [Conda](https://docs.conda.io/en/latest/miniconda.html). Just search "conda install name"*

#### References

1. Van Rossum G, Drake FL. Python 3 Reference Manual. Scotts Valley, CA: CreateSpace; 2009.
1. McKinney W, others. Data structures for statistical computing in python. In: Proceedings of the 9th Python in Science Conference. 2010. p. 51–6.
1. Harris CR, Millman KJ, van der Walt SJ, Gommers R, Virtanen P, Cournapeau D, et al. Array programming with NumPy. Nature. 2020;585:357–62.
1. Virtanen P, Gommers R, Oliphant TE, Haberland M, Reddy T, Cournapeau D, et al. SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature Methods. 2020;17:261–72.
1. Seabold S, Perktold J. Statsmodels: Econometric and statistical modeling with python. InProceedings of the 9th Python in Science Conference 2010 Jun 28 (Vol. 57, No. 61, pp. 10-25080).
1. Hunter JD. Matplotlib: A 2D graphics environment. Computing in science & engineering. 2007;9(3):90–5.
1. Waskom ML. Seaborn: statistical data visualization. Journal of Open Source Software. 2021 Apr 6;6(60):3021.
1. Newville M, Stensitzki T, Allen DB, Rawlik M, Ingargiola A, Nelson A. LMFIT: Non-linear least-square minimization and curve-fitting for Python. Astrophysics Source Code Library. 2016 Jun:ascl-1606.
1. James A. Bednar, Jim Crist, Joseph Cottam, and Peter Wang (2016). "Datashader: Revealing the Structure of Genuinely Big Data", 15th Python in Science Conference (SciPy 2016).
1. Servén D., Brummitt C. (2018). pyGAM: Generalized Additive Models in Python. Zenodo. DOI: 10.5281/zenodo.1208723

([Return to Table of Contents](#table-of-contents))

# How to Cite

If you use any part of this workflow in your research please cite the following manuscript:

PLACEHOLDER FOR MANUSCRIPT CITATION AND LINK

# Future Improvements

1. Add support for other All vs. All ANI formats and tools (skani, sourmash, etc).
1. More efficient All vs. All RBM sequence alignment for nucleotides.
1. Filter MMSeqs2 Gene clusters for improved true ortholog assignments.
1. Compute number of unique alleles for each gene cluster.
1. Compute pN/pS for each gene cluster.

([Return to Table of Contents](#table-of-contents))