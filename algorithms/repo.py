#!/usr/bin/env python

import argparse
import sys
import numpy as np
import pandas as pd

from miscs import get_data
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin

def main():
    # args
    parser = argparse.ArgumentParser()
    # parsing different files
    parser.add_argument("-c", "--coupling", help="extra file for coupling", type=str)
    parser.add_argument("-i", "--ignore", help="except columns from csv", action='append')
    parser.add_argument("-s", "--suffix", help="output suffix for input files input.output.csv",
                        type=str, default=".output.csv")
    parser.add_argument("files", nargs="*")

    args = parser.parse_args()

    if not len(args.files):
        print("Please provide a list of csv\n %s input.csv" % sys.argv[0])
        return

    values, meta = None, None
    headers = []
    filename = args.files[0]
    # reading inputs
    try:
        values, meta, headers = get_data(filename, args.ignore)
    except Exception as e:
        print("error %s" % e)
        return

    if values.empty:
        print("empty file " % filename)

    X = values.to_numpy()
    M = None
    if not meta.empty:
        M = meta.to_numpy()

    model = KMeans(n_clusters=3, n_init=10)
    model.fit(X)

    # output with centroid
    centers = np.sort(model.cluster_centers_, axis=0)
    labels = pairwise_distances_argmin(X,centers)

    # creating output file with the new classes
    # saving file
    def gen():
        if M.any():
            for m, var, label in zip(M, X, labels):
                row = list(m) + list(var) + [label]
                print("columna", row)
                yield row
        else:
            for var, label in zip(X, labels):
                row = list(var) + [label]
                yield row

    columns = headers + ["robot-type"]
    if args.ignore:
        print("aqui no anda")
        columns = args.ignore + columns
    print("veamos" , columns)

    out = pd.DataFrame(list(gen()), columns=columns)
    outputfile = filename + args.suffix
    out.to_csv(outputfile, index=False)
    print("take a look to %s" % outputfile)


if __name__ == '__main__':
    if not main():
        sys.exit(1)