import math
import pandas as pd
import time
import sys
import matplotlib.pyplot as plt
import numpy as np
from random import randint
from random import shuffle
import pickle
import classify as cs

start_time = time.time()

no_of_rows = int(raw_input('Calculation size?\n'))
restrict_occ_prop = float(raw_input('Proportion covered by occupations?\n'))
if restrict_occ_prop == 0:
    soc_level = int(raw_input('Which level of SOC to classify?\n'))
    soc_level_dict = {1:1000, 2:100, 3:10, 4:1}
    soc_div = soc_level_dict[soc_level]
additional_features = int(raw_input('Include additional features?\n'))
no_of_boot_samples = int(raw_input('No of bootstrap samples?\n'))
no_of_cross_val = int(raw_input('No of cross validations?\n'))

sql = 'SELECT * FROM v_records'

injury_records = cs.InjuryRecords(sql, no_of_rows)
print 'Train:', injury_records.train
print 'Test:', injury_records.test

occ_data = injury_records.records
train = injury_records.train

if restrict_occ_prop != 0:
    occ_data['ip_occ_grp'] = cs.restrict_occ(occ_data['ip_occupation'], restrict_occ_prop)
else:
    occ_data['ip_occ_grp'] = occ_data.ip_occupation.apply(lambda d: math.floor(d/soc_div))
occ_data.sic_b3 = occ_data.sic_b3.apply(lambda d: math.floor(d/1000))

occ_dict = cs.OccDict(occ_data['ip_occ_grp'])
occ_data.ip_occ_text = occ_data.ip_occ_text.apply(lambda d: cs.clean_occ(d))
features = occ_data.ip_occ_text.apply(cs.bag_of_words)
occ_codes = occ_data['ip_occ_grp'].values.tolist()

if additional_features == 1:
    row = 0
    for g in features:
        g.update(dict({'sic': occ_data['sic_b3'][row]}))
        g.update(dict({'sex': occ_data['ip_gender'][row]}))
        g.update(dict({'age': occ_data['ip_age'][row]}))
        g.update(dict({'severity': occ_data['severity'][row]}))
        row += 1

feature_set = [(features[x], occ_codes[x]) for x in xrange(len(features))]

cross_val_accuracy = np.empty([no_of_cross_val])
no_dict_entries = occ_dict.no_records
cross_val_con_matrix = np.empty([no_dict_entries, no_dict_entries, no_of_cross_val])

toolbar_width = no_of_cross_val
print "Performing cross validation:"
sys.stdout.write("[%s]" % (" " * toolbar_width))
sys.stdout.flush()
sys.stdout.write("\b" * (toolbar_width + 1))

for x in xrange(no_of_cross_val):
    if x > 0:
        shuffle(feature_set)

    test_set, train_set = feature_set[train:], feature_set[:train]

    bag_classifier = cs.BaggingClassifier(train_set, test_set, occ_dict, no_of_boot_samples)
    cross_val_accuracy[x] = bag_classifier.confusion_matrix.accuracy
    cross_val_con_matrix[:, :, x] = bag_classifier.confusion_matrix.cm_column
    sys.stdout.write("-")
    sys.stdout.flush()

sys.stdout.write("\n")

test_classifier = bag_classifier.model[0]
save_classifier = open('classifier.pickle', 'wb')
pickle.dump(test_classifier, save_classifier)

match_occ_in_top_3=0

for x in xrange(len(test_set)):
    a = test_classifier.prob_classify(test_set[x][0])
    c = [(b, a.prob(b)) for b in a.samples()]
    d = [(e[0]) for e in sorted(c, key=lambda pb: pb[1], reverse=True)][:3]
    if test_set[x][1] in d:
        match_occ_in_top_3 = match_occ_in_top_3 + 1

cross_val_mean = np.mean(cross_val_accuracy)
cross_val_ci =  1.96 * np.std(cross_val_accuracy)
accuracy = 'Cross validated accuracy: %.2f +/- %.2f' % (cross_val_mean, cross_val_ci)

top_3_acc = float(match_occ_in_top_3)/float(injury_records.test)

top_3_accuracy = 'Accuracy of top 3 choices: %.2f' % (top_3_acc)

accuracy = 'Cross validated accuracy: %.2f (Top 3: %.2f)' % (cross_val_mean, top_3_acc)

print accuracy
print top_3_accuracy

cross_val_avg_con_matrix = np.mean(cross_val_con_matrix, axis=2)

occ_labels = [int(occ_dict.dict[x]) for x in xrange(no_dict_entries)]

fig = plt.figure(figsize=(5,5))
fig.suptitle(accuracy, fontsize=14)

ax = fig.add_subplot(111)
cax = ax.matshow(cross_val_avg_con_matrix, cmap='Blues', vmin=0, vmax=1)
ax.xaxis.set_ticks(np.arange(0,no_dict_entries,1))
ax.xaxis.set_ticklabels(occ_labels, rotation='vertical', fontsize=8)
ax.set_xlabel('Predicted')

ax.xaxis.set_label_position('top')
ax.yaxis.set_ticks(np.arange(0,no_dict_entries,1))
ax.yaxis.set_ticklabels(occ_labels, fontsize=8)
ax.set_ylabel('Actual')
fig.colorbar(cax)
fig.savefig('con_matrix.png')

comp_time = time.time() - start_time
m, s = divmod(comp_time, 60)
h, m = divmod(m, 60)
print "Execution Time: %d:%02d:%02d" % (h, m, s)
plt.show()
