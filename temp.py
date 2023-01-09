import pandas as pd
import numpy as np
import os
import sys

def main():
    with open('clusters', 'r') as f:
        clusters_values = f.read()

    clusters_values = clusters_values.replace('\n', '').replace('-1','')
    with open('clusters_list', 'w') as f:
        f.write(clusters_values)


if __name__ == "__main__":
    main()
