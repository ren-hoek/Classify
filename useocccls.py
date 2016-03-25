import pickle
from classify import clean_occ, bag_of_words, occ_lk_dict

saved_classifier = open('classifier.pickle')
classifier = pickle.load(saved_classifier)
saved_classifier.close()
soc_dict = occ_lk_dict()

print classifier.confusion_matrix.cm_np
stop = False
while stop is False:
    job_title = raw_input('Please give job title:\n')
    if job_title != 'quit()':
        print job_title
        clean_job_title = clean_occ(job_title)
        features = bag_of_words(clean_job_title)
        soc_code =  classifier.classify(features)
        print features
        soc_label = soc_dict[soc_code]
        print soc_code,"-",soc_label
        raw_input('Press any key')
    else:
        stop = True
        print 'quitting...'
