from __future__ import print_function
import collections
import numpy as np
import tensorflow as tf
import zipfile
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from scipy import ndimage
from multiprocessing import Pool
from six.moves import cPickle as pickle
from six.moves.urllib.request import urlretrieve

csv = pd.read_csv('children_de_DE.csv')
csv = csv[["imageURL", "sku", "imageId"]]
csv["imageURL"] = csv["imageURL"] + "&fit=inside|128:128"
data = np.array(csv)

def maybe_load(url, h, force=False):
    if not os.path.exists("images"):
        os.mkdir("images")
    filename = get_name(h)
    if force or not os.path.exists(filename):
        try:
            filename, _ = urlretrieve(url, filename)
        except KeyboardInterrupt:
            sys.exit(0)
        except:
            print(url)
        
def _maybe_load(arr):
    maybe_load(arr[0], arr[2])

def get_name(h):
    return "images/"+h+".jpg"

with Pool(8) as p:
    p.map(_maybe_load, data)

#for i, img in enumerate(data):
#    _maybe_load(img)
#    if (i%100==0):
#        print(i)
