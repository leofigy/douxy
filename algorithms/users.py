#!/usr/bin/env python
import sys,os
import logging
import pandas as pd

from miscs import hello
from sklearn.model_selection import train_test_split


# sample header 
# author     string
# added      integer
# deleted    integer
# commits    integer

def main():
    hello()
    # args
    if len(sys.argv) < 2:
        print("Usage: %s <user_file>" % sys.argv[0])
        return
    
    filename = sys.argv[1]
    data = None

    try:
        # weird encoding ... 
        data = pd.read_csv(filename, encoding='iso-8859-1')
    except Exception as e:
        print("Invalid input %s" % e)
        return

    if data.empty:
        print("empty file")
        return

    # train set 
    # validation set
    # 
    train, test = train_test_split(data, test_size=0.1)
    train, val  = train_test_split(train, test_size=0.1)

    print('Train set %d' % len(train))
    print('Validation set %d' % len(val))
    print('Test %d', len(test))


if __name__ == '__main__':
    if not main():
        sys.exit(1)
