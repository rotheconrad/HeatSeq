# HeatSeq: cluster munitions for microbial genomics

![HeatSeq Logo](https://github.com/rotheconrad/HeatSeq/blob/main/images/HeatSeq_logo.jpg)


Quickly and easily predict and visualize distance based clusters for all vs. all ANI, AAI, Mash distance and others. Excellent for exploring distance based genomic clades or sample groups. Developed to investigate intraspecies clades such as genomovars and phylogroups.

This tool uses hierarchical clustering to generate distance based cluster predictions for genomic (ANI, AAI) and metagenomic (beta distance, MASH, Simka) data, or any other all vs. all distance type matrix, and it builds a clustered heatmap with column labels to highlight predicted cluster assignments and other user provided metadata. It outputs a seaborn clustermap in vector based PDF format along with separate PDF legend corresponding to each metadata row. It also outputs a metadata tab separated value (tsv) file with the predicted cluster assignments for each cluster threshold for each genome, metagenome, or sample along with a separate tsv file specifying the color assignments. These output files may be modified by the user and input back into the code to customize the final figure output.

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

### Example output

#### Finished heatmap figure

![Final Figure Example](https://github.com/rotheconrad/HeatSeq/blob/main/images/clustermap_example.png)

#### Predicted clusters table (see: *files/example_predicted_clusters.tsv*)

The genome/sample order in the *predicted_clusters.tsv* file matches the order of the clustermap. The outputs also include a *meta_colors.tsv* file where a color has been assigned to each predicted cluster for each distance threshold, and a *distmat.tsv* which is the distance matrix reordered to match the clustermap.

**genomes**|**clade-0 (96.0% ANI; t=0.04)**|**clade-1 (97.4% ANI; t=0.026)**|**clade-2 (98.0% ANI; t=0.02)**|**clade-3 (98.7% ANI; t=0.013)**|**clade-4 (99.2% ANI; t=0.008)**|**clade-5 (99.8% ANI; t=0.002)**
:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:
gnm\_1383800v1|c0-006|c1-007|c2-008|c3-013|c4-032|c5-039
gnm\_1389944v1|c0-006|c1-007|c2-008|c3-019|c4-048|c5-059
gnm\_1380692v1|c0-006|c1-007|c2-008|c3-019|c4-048|c5-059
gnm\_1380608v1|c0-006|c1-007|c2-008|c3-019|c4-048|c5-059
gnm\_1380652v1|c0-006|c1-007|c2-008|c3-019|c4-048|c5-059
gnm\_1388647v1|c0-006|c1-007|c2-008|c3-019|c4-048|c5-059
gnm\_1388747v1|c0-006|c1-007|c2-008|c3-019|c4-048|c5-059
gnm\_1383334v1|c0-006|c1-007|c2-008|c3-024|c4-053|c5-066
gnm\_1388961v1|c0-006|c1-007|c2-008|c3-024|c4-053|c5-066
gnm\_1389954v1|c0-006|c1-007|c2-008|c3-024|c4-053|c5-066
gnm\_1389962v1|c0-006|c1-007|c2-008|c3-024|c4-053|c5-066
gnm\_1389099v1|c0-006|c1-007|c2-008|c3-020|c4-049|c5-061
gnm\_1383380v1|c0-006|c1-007|c2-008|c3-020|c4-049|c5-060

# Usage

Their are three initial options:

1. Predict clusters for local minimumns in the distance distribution (default).
2. Predict clusters for user defined values (-user).
    e.g. -user 95 97.5 98.5 99.5
3. Skip cluster prediction and use user defined metadata.
    e.g. -m meta_data.tsv -c meta_colors.tsv

And several additional parameters:

1. -dtype, Specify data type. One of fastANI (default), ANI, AAI, Mash, Simka, Distance
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

### Case One: Default

The default case uses [scipy.stats.gaussian_kde](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html) and [scipy.signal.find_peaks](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html) to estimate local minimums and maximums in the distance distribution. The local minimums are used as distance thresholds during the hierarchical clustering process.

```bash
python HeatSeq.py -i files/01_example_fastANI_allV.tsv -o tests/01_example_default
```

#### KDE with peaks (local maximums) and valleys (local minimums)

![Final Figure Example](https://github.com/rotheconrad/HeatSeq/blob/main/images/local_minimums_example.png)

### Case Two: User Defined

The user defined case allows the local minimum estimates to be amended, appended, or changed entirely based on pure curiosity and speculation. In the example below we amended and appended the local minimums that were identified in case one using some intra-species level distance thresholds.

```bash
python HeatSeq.py -i files/01_example_fastANI_allV.tsv -o tests/02_example_default -user 96 97.4 98 98.7 99.2 99.8
```

### Case Three: Custom Metadata

The custom metadata case allows the user to modify the *predicted_clusters.tsv* and the *meta_colors.tsv* files output from the case two example to include additional metadata categories and/or alter the colors. In this example we've added niche and clermontyping assignments for each genome and we've added color assignments for each unique niche or clermontyping category.

```bash
python HeatSeq.py -i files/01_example_fastANI_allV.tsv -o tests/03_example_default -m files/01_example_predicted_clusters.tsv -c files/01_example_meta_colors.tsv
```

([Return to Table of Contents](#table-of-contents))

# Examples

### fastANI

fastANI is the default case and was utilized in the case examples above. Here we show examples with some optional parameters

```bash
# set min and max
python HeatSeq.py -i files/01_example_fastANI_allV.tsv -o tests/xa_example_default -dmin 90 -dmax 99

# no metadata or predicted clusters
python HeatSeq.py -i files/01_example_fastANI_allV.tsv -o tests/xb_example_default -no True

# different method and metric
python HeatSeq.py -i files/01_example_fastANI_allV.tsv -o tests/xc_example_default -method centroid -metric correlation -user 99.5 98.5 97.5 96.5
```

### ANI

This option is to accommodate ANI values estimated with other tools. Arrange the data into a square matrix and format at a tsv file.

```bash
python HeatSeq.py -i files/01_example_fastANI_allV.tsv -o tests/04_example_ANI -dtype ANI
```

*this option has incomplete example and testing*

### AAI

This options alters the value range for hierarchical clustering and the heatmap to accommodate the lower percentage range of AAI estimates. Any AAI tool may be used. Arrange the data into a square matrix and format at a tsv file.

```bash
python HeatSeq.py -i files/05_example_AAI_allV.aai -o tests/05_example_AAI -dtype AAI
```

### Mash

This option takes the output from Mash. It alters the value range for hierarchical clustering and the heatmap to fit 0-1.

Mash releases: https://github.com/marbl/Mash/releases
Mash docs: https://mash.readthedocs.io/

```bash
# first exucute all vs all comparison using mash
# step 1. create sketch of all genomes or metagenomes
# my files contains metagenomes or genomes in fastq or fasta format.
mkdir mySketches
for f in myFiles/*fasta; do n=`basename $f | cut -d. -f1`; ./mash sketch -o mySketches/$n $f; done
# step 2. concatenate sketches in single file
./mash paste allSketches mySketches/*.msh
# step 3. compute all vs all dist with allSketches
./mash dist allSketches.msh allSketches.msh -t > files/06_example_mash_allV.tsv

# step 4. create the heatmap from the mash distance matrix.
python HeatSeq.py -i files/06_example_mash_allV.tsv -o tests/06_example_MASH -dtype Mash -user 0.6 0.8 1.0
```

### Simka

This option takes the output from Simka. It alters the value range for hierarchical clustering and the heatmap to fit 0-1. 

```bash
python HeatSeq.py -i files/07_example_simka_allV.txt -o tests/07_example_SIMKA -dtype Simka -user 0.5 0.7 0.9
```

### Distance

Any square matrix with values 0-1 may be used. Matrix file should be formatted as a tsv file.

*this option has incomplete example and testing*

```bash
python HeatSeq.py -i files/08_example_distance_allV.tsv -o tests/08_example_distance -dtype Distance
```

([Return to Table of Contents](#table-of-contents))

# Software Dependencies

### External dependencies

Code in this repo uses only Python and Python packages which are listed below. However, it requires the user to independently compute distance (ANI, AAI, Mash etc.) with an external tool prior to running this code.

*Use fastANI, enveomics, Mash, Simka etc.*

### Required packages for Python

- [Python](https://www.python.org/) version 3.6+
- [pandas](https://pandas.pydata.org/) 
- [numpy](https://numpy.org/)
- [scipy](https://scipy.org/)
- [matplotlib](https://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)

*Python and all packages can be easily installed with conda or pip. Prodigal, BLAST+ and MMseqs2 can also be installed easily with [Conda](https://docs.conda.io/en/latest/miniconda.html). Just search "conda install name"*

#### References

1. Sanner MF. Python: a programming language for software integration and development. J Mol Graph Model. 1999 Feb 1;17(1):57-61.
1. Van Rossum G, Drake FL. Python 3 Reference Manual. Scotts Valley, CA: CreateSpace; 2009.
1. McKinney W, others. Data structures for statistical computing in python. In: Proceedings of the 9th Python in Science Conference. 2010. p. 51–6.
1. Harris CR, Millman KJ, van der Walt SJ, Gommers R, Virtanen P, Cournapeau D, et al. Array programming with NumPy. Nature. 2020;585:357–62.
1. Virtanen P, Gommers R, Oliphant TE, Haberland M, Reddy T, Cournapeau D, et al. SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature Methods. 2020;17:261–72.
1. Hunter JD. Matplotlib: A 2D graphics environment. Computing in science & engineering. 2007;9(3):90–5.
1. Waskom ML. Seaborn: statistical data visualization. Journal of Open Source Software. 2021 Apr 6;6(60):3021.

([Return to Table of Contents](#table-of-contents))

# How to Cite

If you use any part of this workflow in your research please cite the following manuscript:

PLACEHOLDER FOR MANUSCRIPT CITATION AND LINK

# Future Improvements

1. Add additional parsing support for outputs from other tools to save the user from creating the distance matrix.
1. Add support for checkM or other tools to create the option of selecting a cluster representative for each predicted cluster.
1. Add cluster analysis tools and figures to evaluate the effectiveness of clustering at each distance threshold. Silhouette plots etc.

([Return to Table of Contents](#table-of-contents))