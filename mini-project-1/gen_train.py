# Provided a directory name, the script writes the names of all the images in the directory into a txt file along with their labels. The label is assigned using the first part of the name.

# Assumes images are of jpg format and name is of the form : <class>_<imgname>.jpg

import sys
import glob

try:
    dir_loc = sys.argv[1]

except IndexError:
    print("Usage: python gen_train.py <dir_location>")
    sys.exit(1)

ls = glob.glob(dir_loc + "/*.jpg")

f = open('train.txt','w')

for l in ls:
    f.write(l + " " + l.split("/")[2].split("_")[0] + "\n")

f.close()
