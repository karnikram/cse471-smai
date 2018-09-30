import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


train_path = "./train.txt"
test_path = "./test.txt"
gt_path = "./gt.txt"
pred_path = "./pred.txt"

# run classifier and store output in pred_path
script_path = "./linear_classifier.py"
os.system("python3 {} {} {} > {}".format(script_path, train_path, test_path, pred_path))

"""
We will compute precision, accuracy and F1 score.
For that we require that all the output labels are valid.

For example, if train file contains only three labels : alice, bob and abc
and if your output is some string other than these three labels then it will be considered invalid and you will get zero marks.
"""
# collect all labels of train file
# ---------------------------------
with open(train_path, "r") as f:
    lines = f.readlines()

label_set = set()
for l in lines:
    img_path, label = l.strip().split()
    label_set.add(label)
# ---------------------------------


# validate the predictions
# ---------------------------------
with open(pred_path, "r") as f:
    pred_labels = f.readlines()
    pred_labels = [l.strip() for l in pred_labels]

with open(gt_path, "r") as f:
    gt_labels = f.readlines()
    gt_labels = [l.strip() for l in gt_labels]

for gtl in gt_labels:
    assert(gtl in label_set)

for predl in pred_labels:
    if predl not in label_set:
        print("Invalid output")
        print("All scores zero")
        exit(0)
# ---------------------------------


# evaluate the predictions
# ---------------------------------
mp = {l : idx for idx, l in enumerate(label_set)}
gt_labels = [mp[l] for l in gt_labels]
pred_labels = [mp[l] for l in pred_labels]

print(f1_score(gt_labels, pred_labels, average="macro"))
print(precision_score(gt_labels, pred_labels, average="macro"))
print(recall_score(gt_labels, pred_labels, average="macro"))
# ---------------------------------
