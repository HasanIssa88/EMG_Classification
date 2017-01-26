"""Show how to use pandas, to group files by elements from the file-name

author: ThH
ver:    1.0
date:   Jan-2017
"""

import os
import pandas as pd
import numpy as np

# Get a list of the files starting with "subj"
data_dir = 'data'
dirs = os.listdir(data_dir)
sub_dirs = [dir for dir in dirs if dir.startswith('subj')]

# Make a pandas DataFrame
df = pd.DataFrame({'names':sub_dirs})

# Add the labels, from the file-names
df['type'] = df.names.str.split('[_.]').str.get(5)
df['gtype'] = df.names.str.split('[_.]').str.get(3)
df['subj'] = df.names.str.split('[_.]').str.get(1)

# Show two ways how to extract a group of files
print(df.names[(df.subj=='01') * (df.gtype=='pre')])
print(df.names[np.logical_and(df.subj=='01',df.gtype=='pre')])

input('Thanks for using programs by Thomas ;)')
