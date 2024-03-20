#!/usr/bin/env python

'''Predict subpopulation ANI clusters and build heatmap with metadata.

This tool takes the fastANI_allV.ani output from Part 01, Step 02
as input. It has a few options and ultimately returns an ANI matrix
as a hierarchical clustered heatmap.

Their are three initial options:

1) Predict ANI clusters for local minimumns in the ANI distribution (default).
2) Predict ANI clusters for user defined values (-user).
    e.g. -user 95 97.5 98.5 99.5
3) Skip cluster prediction and use user defined metadata.
    e.g. -m meta_data.tsv -c meta_colors.tsv

The end result is a publication ready figure of an ANI clustermap with 
ANI cluster and/or user defined metadata in vector PDF format.

It will also write out a tsv file with genome name and predicted clusters
if options 1 or 2 were used.

This script requires the following packages:

    * argparse
    * pandas
    * matplotlib
    * seaborn

For option 3, the user defined metadata requires the following two files:

    1) A tab separated metadata file with a row for each genome in the
    ANI file with the exact genome name, and columns for each meta value.
    File should include a header.
            example:
                Genome\tSite\tNiche\tPhylogroup
                Genome1\tA\tSheep\tB1
                Genome2\tB\tCow\tE

    2) A tab separated file of unique meta catagories and colors
            example:
                Site1\t#ffffb3
                Site2\t#377eb8
                Species1\t#ff7f00
                Species2\t#f781bf
                Temp1\t#4daf4a
                Temp2\t#3f1gy5

* NOTE: The color ranges for the heatmap accommodate to 90% ANI difference
        The code will need modification for more diverse datasets.

To set different cluster method choose from:
    single, complete, average, weighted, centroid, median, ward
    *See webpage/docs for scipy.cluster.hierarchy.linkage for more info

To set different cluster metrics choose from:
    euclidean, seuclidean, correlation, hamming, jaccard, braycurtis
    *See webpage/docs for scipy.spatial.distance.pdist for additional 
     options and more info

To use different data types set -dtype to on of the following:
    * fastANI - (default) for longform fastANI output.
    * ANI - For all vs. all square matrix of any ANI data.
    * AAI - For all vs. all square matrix of any AAI data.
    * Mash - For an all vs. all square matrix output by Mash
    * Simka - For an all vs. all square matrix output by Simka
    * Distance - For any other all vs. all square matrix range 0-1.

-------------------------------------------
Author :: Roth Conrad w/ thanks to Dorian J. Feistel
Email :: rotheconrad@gatech.edu
GitHub :: https://github.com/rotheconrad
Date Created :: March 2024
License :: GNU GPLv3
Copyright 2024 Roth Conrad
All rights reserved
-------------------------------------------
'''

import argparse, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from decimal import Decimal
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.cluster.hierarchy import fclusterdata
from scipy.spatial.distance import pdist, squareform

def parse_ANI_file(infile, dmin, dmax):

    ''' creates a square distance matrix of 100 - ANI '''

    dist = {} # keep ANI distance of each genome pair
    gdict = {} # keep set of genome names
    data = {} # build dict to convert to dataframe (distance matrix)
    ani_array = []

    # read the file, compute distance 100 - ANI, store data in dist dict
    with open(infile, 'r') as file:
        # header = file.readline() # ANI output should not have a header
        for line in file:
            X = line.rstrip().split('\t')
            # genomes in pair
            g1 = X[0].split('/')[-1].split('.')[0]
            g2 = X[1].split('/')[-1].split('.')[0]
            # get ANI
            ani = round(float(X[2]), 2)
            # get second genome length (total kmers)
            g2size = int(X[4])
            # filter min and max ANI values
            if dmin and ani < dmin: continue
            if not g1 == g2:
                if dmax and ani > dmax: continue
            # add ani to array
            ani_array.append(ani)
            # convert to ANI distance
            #d = 100 - ani
            d = ani
            # store distance by genome pair name
            gpair = '-'.join(sorted([g1, g2]))
            if gpair in dist:
                gsize = dist[gpair][1]
                if g2size > gsize:
                    dist[gpair] = [d, g2size]
            else:
                dist[gpair] = [d, g2size]
            # add all genome names to gdict to get the complete set of names
            gdict[g1] = ''
            gdict[g2] = ''

    # get the sorted genome name list
    glist = sorted(list(gdict.keys()))
    #data['rows'] = glist # add rows or index names to dict.
    # iterate through genome list and dictionary to create square matrix df
    for g1 in glist:
        data[g1] = []
        for g2 in glist:
            key = '-'.join(sorted([g1, g2]))
            d = dist.get(key, [np.nan, 0])[0]
            data[g1].append(d)

    df = pd.DataFrame(data, index=glist, columns=glist).dropna()
    df = df[df.index]

    if len(df) == 0:
        print(
            '\n\n'
            '\tERROR: The dataframe is empty.\n'
            '\tIts possible that dmin or dmax may be too stringent.\n'
            '\tTry adjusting the parameters or alternately,\n'
            '\tyou could modify your input to remove unwanted genomes.'
            '\n\n'
            )
        sys.exit(1)

    return df, ani_array


def get_valleys(ani_array, outpre):

    # use peak_finder algorithm to find local minimums (valleys)

    kde = gaussian_kde(ani_array, bw_method=0.15)
    x = np.linspace(min(ani_array), max(ani_array), 1000)
    y = kde(x)
    yp = find_peaks(y) # returns array (peak indices in x, {properties dict})
    yn = y * -1 # inverse of y to find the valleys
    yv = find_peaks(yn) # these are the valley position
    pks = yp[0] # peaks
    vls = yv[0] # valleys
    local_maximums = [round(i,1) for i in sorted(x[pks], reverse=True)]
    local_minimums = [round(i,1) for i in sorted(x[vls], reverse=True)]

    max_txt = 'Local maximums: ' + ', '.join([str(i) for i in local_maximums])
    min_txt = 'Local minimums: ' + ', '.join([str(i) for i in local_minimums])
    line = f'{max_txt}\n{min_txt}'

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='KDE')
    plt.scatter(x[pks], y[pks], color='red', label='Peaks')
    plt.scatter(x[vls], y[vls], color='blue', label='Valleys', marker='v')
    plt.title('ANI KDE with Peaks and Valleys')
    plt.xlabel('ANI')
    plt.ylabel('Density')
    ax = plt.gca()
    plt.text(0.01, 0.99, line, horizontalalignment='left',
     verticalalignment='top', transform=ax.transAxes)
    plt.legend()
    
    plt.savefig(f'{outpre}_Local_minimums.pdf')

    return local_minimums


def predict_clusters(df, d_thresholds, metric, method, units):

    # initialize dicts for meta data and colors
    metadf = {}
    cdict = {}

    genomes = df.columns.tolist()
    # prepare ANI and AAI
    if units == 100:
        if sum(np.diag(df)) > 0:
            # convert ANI or AAI percent to distance
            df = (100-df)/100
        sorted_threshold = sorted(d_thresholds)
        distance_thresholds = [round((100 - i)/100, 4) for i in sorted_threshold]
    # prepare 0-1 distance Mash, Simka etc
    if units == 1:
        sorted_threshold = sorted(d_thresholds, reverse=True)
        distance_thresholds = sorted_threshold

    # convert to condensed matrix
    # for some reason the condensed matrix gives (triangle) gives different
    # results than the uncondensed matrix (square)
    # squareform converts between condensed and uncondensed
    cond_matrix = squareform(df.to_numpy())

    for n, d in enumerate(distance_thresholds):
        label = f'clade-{n} ({sorted_threshold[n]}% ANI; t={d})'
        
        Z = linkage(cond_matrix, metric=metric, method=method)
        clusters = fcluster(Z=Z, t=d, criterion='distance')
        #clusters = fclusterdata(
                            #X=df, t=d, criterion='distance',
                            #metric=metric, method=method
                            #)
        named = [f'c{n:01}-{i:03}' for i in clusters]
        metadf[label] = named

    # prepare metadf as dataframe
    metadf = pd.DataFrame(metadf, index=genomes)

    # populate cdict with color for each unique metadf value
    count = 0
    for m in metadf.columns:
        count += 1
        unique_clusters = sorted(metadf[m].unique())
        colors = sns.color_palette("Paired", len(unique_clusters)).as_hex()
        if count % 2 == 0: colors.reverse()
        for i,u in enumerate(unique_clusters):
            cdict[u] = colors[i]

    return metadf, cdict


def parse_colors(metacolors):

    ''' reads meta colors file into a dict of {meta value: color} '''

    cdict = {}

    with open(metacolors, 'r') as file:
        for line in file:
            X = line.rstrip().split('\t')
            mval = X[0]
            color = X[1]
            cdict[mval] = color

    return cdict


def get_custom_cmap(d_array, dmin, dmax, units):

    # define colors
    # Read these as colors per whole ANI value
    # eg: [100 red, 99 orange, 98 yellow, 97 green, ] or
    # eg: [0 red, 0.1 orange, 0.2 yellow, 0.3 green, ]
    colors = [
              '#e41a1c', '#ff7f00', '#ffff33', '#006d2c', '#80cdc1',
              '#377eb8', '#e78ac3', '#984ea3', '#bf812d', '#bababa',
              '#252525'
              ]

    # get min and max
    if not dmin:
        dmin = np.min(d_array)
        print(f'\n\n\tData minimum: {dmin}')
    if not dmax: 
        dmax = np.max(d_array)
        print(f'\n\n\tData maximum: {dmax}')

    if units == 100:
        # one color per 1% ANI range
        d_range = int(np.ceil(dmax) - np.floor(dmin) + 1)
        if d_range > 11: d_range = 11
        cmap = sns.blend_palette(reversed(colors[:d_range]), as_cmap=True)

    elif units == 1:
        dmax = float(Decimal(dmax).quantize(Decimal("1.0"), 'ROUND_UP'))
        dmin = float(Decimal(dmin).quantize(Decimal("1.0"), 'ROUND_DOWN'))

        # one color per 0.1 distance range
        step = 0.1
        d_range = len(np.arange(dmin, dmax+step, step))
        if d_range == 4:
            d_range += 3
        elif d_range == 3:
            d_range += 4
        elif d_range == 2:
            d_range += 5
        elif d_range == 1:
            d_range += 6

        cmap = sns.blend_palette(colors[:d_range], as_cmap=True)

    return cmap, dmin, dmax


def plot_clustred_heatmap(
            df, outpre, cmap, metadf, cdict, dmin, dmax, metric, method
            ):

    # build the plot with metadata
    if not metadf.empty:
        # Build legend
        for meta in metadf.columns:
            labels = metadf[meta].unique()

            # create columns for larger metadata sets
            cols = int(np.ceil(len(labels)/20))

            fig, ax = plt.subplots(figsize=(cols*4,10))
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)

            for label in labels:
                ax.bar(
                    0, 0, color=cdict[label], label=label, linewidth=0
                    )

            ax.legend(
                title=meta, title_fontsize='xx-large', loc="center",
                frameon=False, markerscale=5, fontsize='xx-large', ncol=cols
                )
            plt.savefig(f'{outpre}_Legend_{meta}.pdf')
            plt.close()

        # Build the clustermap
        colordf = metadf.replace(cdict) # swap colors in for values
        g = sns.clustermap(
                        df, figsize=(16,9), cmap=cmap,
                        xticklabels=True,
                        yticklabels=True,
                        col_colors=colordf,
                        vmin=dmin, vmax=dmax,
                        metric=metric, method=method
                        )


    # build the plot without metadata
    else:
        g = sns.clustermap(
                        df, figsize=(16,9), cmap=cmap,
                        xticklabels=True,
                        yticklabels=True,
                        vmin=dmin, vmax=dmax,
                        metric=metric, method=method
                        )

    # reorder the distance matrix and write to file
    orows = df.columns[g.dendrogram_row.reordered_ind] # row order
    ocols = df.index[g.dendrogram_col.reordered_ind] # col order
    df = df.reindex(orows)
    df = df[ocols]
    df.to_csv(f'{outpre}_distmat.tsv', sep='\t')

    metadf = metadf.reindex(orows)
    metadf.to_csv(f'{outpre}_predicted_clusters.tsv', sep='\t')

    with open(f'{outpre}_meta_colors.tsv', 'w') as cout:
        for meta, color in cdict.items():
            cout.write(f'{meta}\t{color}\n')

    g.cax.set_xlabel('ANI (%)', rotation=0, labelpad=10, fontsize=12)
    # remove labels and tick marks
    g.ax_heatmap.set_xlabel('')
    g.ax_heatmap.set_ylabel('')
    g.ax_heatmap.set_xticklabels([])
    g.ax_heatmap.set_yticklabels([])
    # If you want to also remove the ticks
    g.ax_heatmap.tick_params(axis='x', which='both', length=0)
    g.ax_heatmap.tick_params(axis='y', which='both', length=0)
    # write file
    g.savefig(f'{outpre}_clustermap.pdf')
    plt.close()

    return True


def main():

    # Configure Argument Parser
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
        )
    parser.add_argument(
        '-i', '--input_file',
        help='Please specify the input file name!',
        metavar=':',
        type=str,
        required=True
        )
    parser.add_argument(
        '-o', '--output_file_prefix',
        help='Please specify the prefix to use for output file names!',
        metavar=':',
        type=str,
        required=True
        )
    parser.add_argument(
        '-user', '--user_clusters',
        help='(Optional) Input ANI value list for cluster predction (eg 95, 97.5, 99.5)!',
        metavar=':',
        type=float,
        nargs='+',
        required=False,
        default=None
        )
    parser.add_argument(
        '-m', '--meta_data_file',
        help='(Optional) Please specify a meta data file!',
        metavar=':',
        type=str,
        required=False,
        default=None
        )
    parser.add_argument(
        '-c', '--meta_colors_file',
        help='(Optional) Please specify a meta colors file!',
        metavar=':',
        type=str,
        required=False,
        default=None
        )
    parser.add_argument(
        '-no', '--no_meta_data',
        help='(Optional) Set -no True to build basic seaborn clustermap!',
        metavar=':',
        type=str,
        required=False,
        default=None
        )
    parser.add_argument(
        '-dmin', '--minimum_distance',
        help='(Optional) Minimum distance for plot color (Default: data min)!',
        metavar=':',
        type=float,
        required=False,
        default=None
        )
    parser.add_argument(
        '-dmax', '--maximum_distance',
        help='(Optional) Maximum distance for plot color (Default: data max)!',
        metavar=':',
        type=float,
        required=False,
        default=None
        )
    parser.add_argument(
        '-metric', '--cluster_metric',
        help='(Optional) Specify cluster metric (default: euclidean)',
        metavar=':',
        type=str,
        required=False,
        default='euclidean'
        )
    parser.add_argument(
        '-method', '--cluster_method',
        help='(Optional) Specify cluster metric (default: average)',
        metavar=':',
        type=str,
        required=False,
        default='average'
        )
    parser.add_argument(
        '-dtype', '--data_type',
        help='(Optional) Specify data type (default: fastANI)',
        metavar=':',
        type=str,
        required=False,
        default='fastANI'
        )
    args=vars(parser.parse_args())

    # Do what you came here to do:
    print('\n\nRunning Script...')
    
    # define parameters
    infile = args['input_file']
    outpre = args['output_file_prefix']
    user_clusters = args['user_clusters']
    metadata = args['meta_data_file']
    metacolors = args['meta_colors_file']
    no_meta = args['no_meta_data']
    dmin = args['minimum_distance']
    dmax = args['maximum_distance']
    metric = args['cluster_metric']
    method = args['cluster_method']
    dtype = args['data_type']

    # read in the data.
    if dtype == 'fastANI':
        # read in all vs. all fastANI file.
        df, d_array = parse_ANI_file(infile, dmin, dmax)
        units = 100
    elif dtype == 'ANI' or dtype == 'AAI':
        df = pd.read_csv(infile, sep='\t', index_col=0)
        d_array = df.to_numpy().flatten()
        units = 100
    elif dtype == 'Mash' or dtype == 'Distance':
        df = pd.read_csv(infile, sep='\t', index_col=0)
        d_array = df.to_numpy().flatten()
        units = 1
    elif dtype == 'Simka':
        df = pd.read_csv(infile, sep=';', index_col=0)
        d_array = df.to_numpy().flatten()
        units = 1
    else:
        print(
            '\n\n\t\tERROR: please specify one of the following:\n'
            '\t\t\tfastANI, ANI, AAI, Mash, Simka, or Distance.\n\n'
            )

    # define min, max and cmap
    cmap, dmin, dmax = get_custom_cmap(d_array, dmin, dmax, units)

    ####################################################################

    if no_meta:
        # create the plot without meta data colors
        metadf, cdict = pd.DataFrame(), pd.DataFrame()
        _ = plot_clustred_heatmap(
            df, outpre, cmap, metadf, cdict, dmin, dmax, metric, method
            )

    elif metadata and metacolors:
        cdict = parse_colors(metacolors)
        metadf = pd.read_csv(metadata, sep='\t', index_col=0, header=0)
        metadf = metadf.reindex(df.index).dropna()
        # select only rows in metadf
        df = df[df.index.isin(metadf.index)]
        # select only columns in metadf
        df = df[metadf.index.tolist()]
        _ = plot_clustred_heatmap(
            df, outpre, cmap, metadf, cdict, dmin, dmax, metric, method
            )

    elif metadata:
        print('\n\nBoth meta data and meta colors are required to use them!')

    elif metacolors:
        print('\n\nBoth meta data and meta colors are required to use them!')

    elif user_clusters:
        # run clustering algo on user defined values
        metadf, cdict = predict_clusters(df, user_clusters, metric, method, units)

        _ = plot_clustred_heatmap(
            df, outpre, cmap, metadf, cdict, dmin, dmax, metric, method
            )

    else:
        # default case. find_peaks, get local minimums, predict clusters, plot
        predicted_clusters = get_valleys(d_array, outpre)
        metadf, cdict = predict_clusters(df, predicted_clusters, metric, method, units)

        _ = plot_clustred_heatmap(
            df, outpre, cmap, metadf, cdict, dmin, dmax, metric, method
            )

    print('\n\nComplete success space cadet! Hold on to your boots.\n\n')


if __name__ == "__main__":
    main()
