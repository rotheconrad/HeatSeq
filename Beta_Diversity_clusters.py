#!/usr/bin/env python

'''Predict clusters of similar samples and build heatmap with metadata.

This tool takes the distance from Simka or Mash output as input.
It has a few options and ultimately returns predicted clusters and a
hierarchical clustered heatmap as a vector based PDF.

Their are three initial options:

1) Predict clusters for local minimumns in the distance distribution (default).
2) Predict clusters for user defined distances (-user).
    e.g. -user 0.9 0.8 0.7 0.6 0.5
3) Skip cluster prediction and use user defined metadata.
    e.g. -m meta_data.tsv -c meta_colors.tsv

The end result is a publication ready figure of a distance clustermap with 
cluster and/or user defined metadata in vector PDF format.

It will also write out a tsv file with genome name and predicted clusters
if options 1 or 2 were used.

This script requires the following packages:

    * argparse
    * pandas
    * matplotlib
    * seaborn

For option 3, the user defined metadata requires the following two files:

    1) A tab separated metadata file with a row for each sample in the
    input file with the exact sample name, and columns for each meta value.
    File should include a header.
            example:
                Sample\tSite\tNiche\tPhylogroup
                Sample1\tA\tSheep\tB1
                Sample2\tB\tCow\tE

    2) A tab separated file of unique meta catagories and colors
            example:
                Site1\t#ffffb3
                Site2\t#377eb8
                Species1\t#ff7f00
                Species2\t#f781bf
                Temp1\t#4daf4a
                Temp2\t#3f1gy5

* NOTE: The color ranges for the heatmap accommodate 0 - 1 range.

To set different cluster method (default: average) choose from:
    single, complete, average, weighted, centroid, median, ward
    *See webpage/docs for scipy.cluster.hierarchy.linkage for more info

To set different cluster metrics (default: euclidean) choose from:
    euclidean, seuclidean, correlation, hamming, jaccard, braycurtis
    *See webpage/docs for scipy.spatial.distance.pdist for additional 
     options and more info

-------------------------------------------
Author :: Roth Conrad w/ thanks to Dorian Feistel
Email :: rotheconrad@gatech.edu
GitHub :: https://github.com/rotheconrad
Date Created :: March 2024
License :: GNU GPLv3
Copyright 2024 Roth Conrad
All rights reserved
-------------------------------------------
'''

import argparse
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


def get_valleys(d_array, outpre):

    # use peak_finder algorithm to find local minimums (valleys)

    kde = gaussian_kde(d_array, bw_method=0.15)
    x = np.linspace(min(d_array), max(d_array), 1000)
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


def predict_clusters(df, d_thresholds, metric, method):

    # initialize dicts for meta data and colors
    metadf = {}
    cdict = {}

    genomes = df.columns.tolist()
    # convert ANI and thresholds to ANI distance
    #df = (100-df)/100
    sorted_threshold = sorted(d_thresholds, reverse=True)
    distance_thresholds = sorted_threshold

    # convert to condensed matrix
    # for some reason the condensed matrix gives (triangle) gives different
    # results than the uncondensed matrix (square)
    # squareform converts between condensed and uncondensed
    cond_matrix = squareform(df.to_numpy())

    for n, d in enumerate(distance_thresholds):
        label = f'clade-{n} (d={sorted_threshold[n]})'
        
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


def custom_cmap(bmin, bmax):

    # one color per 0.1 distance range
    step = 0.1
    d_range = len(np.arange(bmin, bmax+step, step))
    if d_range == 4:
        d_range += 3
    elif d_range == 3:
        d_range += 4
    elif d_range == 2:
        d_range += 5
    elif d_range == 1:
        d_range += 6

    # Read these as colors per 0.1 increment
    # eg: [0 red, 0.1 orange, 0.2 yellow, 0.3 green, ]

    colors = [
              '#e41a1c', '#ff7f00', '#ffff33', '#006d2c', '#80cdc1',
              '#377eb8', '#e78ac3', '#984ea3', '#bf812d', '#bababa',
              '#252525'
              ]

    cmap = sns.blend_palette(colors[:d_range], as_cmap=True)

    return cmap


def plot_clustred_heatmap(
            df, outpre, cmap, metadf, cdict, bmin, bmax, metric, method
            ):

    # build the plot with metadata
    if not metadf.empty:
        # Build legend
        for meta in metadf.columns:
            labels = metadf[meta].unique()

            # create columns for larger metadata sets
            cols = np.ceil(len(labels)/20)

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
                        vmin=bmin, vmax=bmax,
                        metric=metric, method=method
                        )


    # build the plot without metadata
    else:
        g = sns.clustermap(
                        df, figsize=(16,9), cmap=cmap,
                        xticklabels=True,
                        yticklabels=True,
                        vmin=bmin, vmax=bmax,
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

    g.cax.set_xlabel('Distance', rotation=0, labelpad=10, fontsize=12)
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
        help='Input distance list for cluster predction (eg 0.9 0.8 0.7 0.6)!',
        metavar=':',
        type=float,
        nargs='+',
        required=False,
        default=None
        )
    parser.add_argument(
        '-m', '--meta_data_file',
        help='Please specify a meta data file!',
        metavar=':',
        type=str,
        required=False
        )
    parser.add_argument(
        '-c', '--meta_colors_file',
        help='Please specify a meta colors file!',
        metavar=':',
        type=str,
        required=False
        )
    parser.add_argument(
        '-no', '--no_meta_data',
        help='Set -no True to build basic seaborn clustermap of distances!',
        metavar=':',
        type=str,
        required=False,
        default=None
        )
    parser.add_argument(
        '-min', '--minimum_beta_distance',
        help='Minimum distance to include. Removes data below. (Default: 0.0)!',
        metavar=':',
        type=float,
        required=False,
        default=0.0
        )
    parser.add_argument(
        '-max', '--maximum_beta_distance',
        help='Maximum distance to include. Removes data above. (Default: max)!',
        metavar=':',
        type=float,
        required=False,
        default=None
        )
    parser.add_argument(
        '-metric', '--cluster_metric',
        help='Cluster metric (default: euclidean)',
        metavar=':',
        type=str,
        required=False,
        default='euclidean'
        )
    parser.add_argument(
        '-method', '--cluster_method',
        help='Cluster metric (default: average)',
        metavar=':',
        type=str,
        required=False,
        default='average'
        )
    parser.add_argument(
        '-beta', '--beta_diversity_type',
        help='Use -beta Simka or -beta Mash',
        metavar=':',
        type=str,
        required=True,
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
    bmin = args['minimum_beta_distance']
    bmax = args['maximum_beta_distance']
    metric = args['cluster_metric']
    method = args['cluster_method']
    beta = args['beta_diversity_type']

    # read in the data.
    if beta == 'Simka':
        df = pd.read_csv(infile, sep=';', index_col=0)
        d_array = df.to_numpy().flatten()
    elif beta == 'Mash':
        df = pd.read_csv(infile, sep='\t', index_col=0)
        d_array = df.to_numpy().flatten()
    else:
        print('\n\n\t\tERROR: please specify -beta Simka or -beta Mash\n\n')
    
    # get bmax from the data
    if not bmax:
        dmax = np.max(d_array)
        bmax = float(Decimal(dmax).quantize(Decimal("1.0"), 'ROUND_UP'))
    if not bmin:
        dmin = np.min(d_array)
        bmin = float(Decimal(dmin).quantize(Decimal("1.0"), 'ROUND_DOWN'))

    print(f'\n\n\tData min and max: {bmin}, {bmax}')

    # create custom color map for the heatmap gradient
    cmap = custom_cmap(bmin, bmax)
    
    if no_meta:
        # create the plot without meta data colors
        metadf, cdict = pd.DataFrame(), pd.DataFrame()
        _ = plot_clustred_heatmap(
            df, outpre, cmap, metadf, cdict, bmin, bmax, metric, method
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
            df, outpre, cmap, metadf, cdict, bmin, bmax, metric, method
            )

    elif metadata:
        print('\n\nBoth meta data and meta colors are required to use them!')

    elif metacolors:
        print('\n\nBoth meta data and meta colors are required to use them!')

    elif user_clusters:
        # run clustering algo on user defined values
        metadf, cdict = predict_clusters(df, user_clusters, metric, method)

        _ = plot_clustred_heatmap(
            df, outpre, cmap, metadf, cdict, bmin, bmax, metric, method
            )

    else:
        # default case. find_peaks, get local minimums, predict clusters, plot
        predicted_clusters = get_valleys(d_array, outpre)
        metadf, cdict = predict_clusters(df, predicted_clusters, metric, method)

        _ = plot_clustred_heatmap(
            df, outpre, cmap, metadf, cdict, bmin, bmax, metric, method
            )

    print('\n\nComplete success space cadet! Hold on to your boots.\n\n')


if __name__ == "__main__":
    main()
