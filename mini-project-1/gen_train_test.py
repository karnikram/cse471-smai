# Generates train, test, and gt files using the image file names in the provided directory.
# Assumes names are of the form : <label>_<name>.jpg

import sys
import glob

try:
    dir_loc = sys.argv[1]

except IndexError:
    print('Correct usage: python gen_train_test.py <img_directory_path>')
    sys.exit(1)

ls = glob.glob(dir_loc + "/*.jpg")
n = len(ls)

train_file = open('train.txt','w')
test_file = open('test.txt','w')
gt_file = open('gt.txt','w')

for i,l in enumerate(ls):
    if(i <= 0.8 * n):
        train_file.write(l + " " + l.split("/")[2].split("_")[0] + "\n")

    else:
        test_file.write(l + "\n")
        gt_file.write(l.split("/")[2].split("_")[0] + "\n")

train_file.close()
test_file.close()
gt_file.close()
