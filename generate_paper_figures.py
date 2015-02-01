import random
import optunity.metrics as metrics
from semisup_metrics import *
from matplotlib import pyplot as plt
import pickle
import csv

num_pos = 2000
num_neg = 0
num_unl = 10000
true_pfrac = 0.2
guessed_pfrac = 0.2

# use the bootstrap to estimate CI on rank CDF of known positives?
# if False, the DKW band will be used instead (typically looser)
bootstrap = True
nboot = 2000
ci_width = 0.95

# write results to csv files?
num = 3
write_csv = False

# config for figure 6 in the paper
#dist_config = {'mean_pos': 1.0, 'sigma_pos': 1.0,
#               'mean_neg': 0.0, 'sigma_neg': 1.0}

# config for figure 7 in the paper
#dist_config = {'mean_pos': 2.0, 'sigma_pos': 2.0,
#               'mean_neg': 0.0, 'sigma_neg': 1.0}

# config for figure 8 in the paper
dist_config = {'mean_pos': 1.0, 'sigma_pos': 0.3,
               'mean_neg': 0.0, 'sigma_neg': 1.0}

def generate_decvals(num, mean, sigma):
    return [random.normalvariate(mean, sigma) for _ in range(num)]

generate_pos_class_decvals = lambda num: generate_decvals(num, dist_config['mean_pos'], dist_config['sigma_pos'])
generate_neg_class_decvals = lambda num: generate_decvals(num, dist_config['mean_neg'], dist_config['sigma_neg'])

def load_resvm_data(fname, setting, labeled_frac_pos, labeled_frac_neg, fp_frac=0.0, fn_frac=0.0):
    data = pickle.load(open(fname, 'r'))
    true_labels = data['labels'].tolist()
    labels = true_labels[:]
    decision_values = list(data[setting]['resvm'])

    total_pos = float(sum(true_labels))
    total_neg = float(len(true_labels) - total_pos)
    labeled_pos = total_pos * labeled_frac_pos
    labeled_neg = total_neg * labeled_frac_neg
    pos_idx = [idx for idx, lab in enumerate(true_labels) if lab]
    neg_idx = [idx for idx, lab in enumerate(true_labels) if not lab]
    relabel_pos_idx = random.sample(pos_idx, int(total_pos - labeled_pos))
    relabel_neg_idx = random.sample(neg_idx, int(total_neg - labeled_neg))
    for i in relabel_pos_idx:
        labels[i] = None
    for i in relabel_neg_idx:
        labels[i] = None

    # add label noise
#    pos_idx = set(pos_idx).difference(set(relabel_pos_idx))
#    neg_idx = set(neg_idx).difference(set(relabel_neg_idx))
    print(len(pos_idx) * fp_frac)
    print(len(neg_idx))
    fp_idx = random.sample(neg_idx, int(len(pos_idx) * fp_frac))
    fn_idx = random.sample(pos_idx, int(len(neg_idx) * fn_frac))
    for i in fp_idx:
        labels[i] = True
    for i in fn_idx:
        labels[i] = False

    return labels, true_labels, decision_values

##################################
#   GENERATE DATA SET
##################################

num_neg_in_unl = int(num_unl * (1 - true_pfrac))
num_pos_in_unl = int(num_unl * true_pfrac)
labels = [True] * num_pos + [None] * num_pos_in_unl + [False] * num_neg + [None] * num_neg_in_unl
true_labels = [True] * (num_pos + num_pos_in_unl) + [False] * (num_neg + num_neg_in_unl)
decision_values = generate_pos_class_decvals(num_pos + num_pos_in_unl) + generate_neg_class_decvals(num_neg + num_neg_in_unl)

#labels, true_labels, decision_values = load_resvm_data('experiments/sensit_1_0.pkl', 'semi', 0.1, 0.0)
##labels, true_labels, decision_values = load_resvm_data('experiments/sensit_1_0.pkl', 'pu', 0.1, 0.1)
#guessed_pfrac = float(true_labels.count(True) - labels.count(True)) / labels.count(None)
#true_pfrac = guessed_pfrac

##################################
#   COMPUTE TRUE ROC AND PR CURVES
##################################

sort_labels, sort_dv = zip(*sorted(zip(labels, decision_values), key=op.itemgetter(1), reverse=True))
known_pos_ranks = [idx for idx, lab in enumerate(sort_labels) if lab]
known_pos_ecdf = compute_ecdf_curve(known_pos_ranks)
auc_true, roc_true = metrics.roc_auc(true_labels, decision_values, return_curve=True)
auc_neg, curve_neg = metrics.roc_auc(labels, decision_values, return_curve=True)

pr_auc, pr_true = metrics.pr_auc(true_labels, decision_values, return_curve=True)
pr_neg_auc, pr_neg = metrics.pr_auc(labels, decision_values, return_curve=True)

ranks = list(range(len(labels)))

##################################
#   COMPUTE APPROXIMATIONS
##################################

print('Computing bounds.')
print('+ Computing CI bounds on CDF of known positives.')
if bootstrap:
    cdf_bounds = bootstrap_ecdf_bounds(labels, decision_values,
                                    pmap=optunity.pmap, nboot=nboot, ci_width=ci_width)
else:
    cdf_bounds = dkw_bounds(labels, decision_values, ci_width=ci_width)

print('+ Computing contingency tables.')
tables = compute_contingency_tables(labels=sort_labels, decision_values=sort_dv,
                                    reference_lb=cdf_bounds.lower,
                                    reference_ub=cdf_bounds.upper,
                                    beta=guessed_pfrac, presorted=True)

print('+ Computing ROC curves.')
roc_bounds = roc_bounds(labels, decision_values, beta=guessed_pfrac, tables=tables, cdf_bounds=cdf_bounds)
auc_lower = auc(roc_bounds.lower)
auc_upper = auc(roc_bounds.upper)

print('+ Computing PR curves.\n')
pr_bounds = pr_bounds(labels, decision_values, beta=guessed_pfrac, tables=tables, cdf_bounds=cdf_bounds)
auc_lower = auc(pr_bounds.lower)
auc_upper = auc(pr_bounds.upper)

##################################
#   DRAW PLOTS
##################################

print('Plotting.')
print('+ Plotting rank distributions.')
plt.figure(1)
plt.plot(ranks, map(cdf_bounds.lower, ranks), color='blue', linewidth=5, alpha=0.4)
plt.plot(ranks, map(cdf_bounds.upper, ranks), color='red', linewidth=5, alpha=0.4)
plt.plot(*zip(*known_pos_ecdf), color='black', linewidth=2)
plt.xlabel('rank')
plt.ylabel('TPR')
plt.draw()

print('+ Plotting ROC curves.')
plt.figure(2)
plt.plot(*zip(*roc_true), color='black', linewidth=2)
plt.plot(*zip(*roc_bounds.lower), color='blue')
plt.plot(*zip(*roc_bounds.upper), color='red')
plt.plot(*zip(*curve_neg), color='magenta')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.draw()

print('+ Plotting PR curves.')
plt.figure(3)
plt.plot(*zip(*pr_true), color='black', linewidth=2)
plt.plot(*zip(*pr_bounds.lower), color='blue')
plt.plot(*zip(*pr_bounds.upper), color='red')
plt.plot(*zip(*pr_neg), color='magenta')
plt.xlabel('TPR')
plt.ylabel('Precision')
plt.show()

##################################
#   SAVE OUTPUT TO CSV FILE
##################################

if write_csv:

    xs = set(range(101))
    reference = [(0.0, float(x)) for x in xs]


    # clean PR curve
    def clean_curve(curve):
        new_curve = []

        it = 0
        while it < len(curve):
            new_curve.append(curve[it])
            while curve[it][0] == new_curve[-1][0]:
                it += 1
                if it == len(curve): break
            if not new_curve[-1] == curve[it - 1]:
                new_curve.append(curve[it - 1])
        return new_curve


    roc_neg = curve_neg
    curves = {'true': roc_true, 'lower': roc_bounds.lower, 'upper': roc_bounds.upper, 'neg': roc_neg}

    mindiff = 1e-3
    fname = 'roc_' + str(num) + '_'
    for name, curve in curves.items():
        curve = clean_curve(curve)
        with open(fname + name + '.csv', 'w') as f:
            w = csv.writer(f)

            w.writerow(['fpr', 'tpr'])
            lastfpr = -1
            for fpr, tpr in curve:
    #            if fpr < 1.0 and fpr % 0.001 < 5e-5: #min(fpr % 0.001, math.fabs(0.001 - fpr % 0.001)) < 1e-7:
    #                w.writerow([fpr, tpr])
                if math.fabs(fpr - lastfpr) > mindiff:
                    lastfpr = fpr
                    w.writerow([fpr, tpr])
                    if tpr == 1.0: break
            w.writerow([1.0] * 2)

    curves = {'true': pr_true, 'lower': pr_bounds.lower, 'upper': pr_bounds.upper, 'neg': pr_neg}
    fname = 'pr_' + str(num) + '_'
    for name, curve in curves.items():
        curve = clean_curve(curve)
        with open(fname + name + '.csv', 'w') as f:
            w = csv.writer(f)

            w.writerow(['tpr', 'precision'])
            w.writerow([0.0, 0.0])
            lasttpr = -1
            for tpr, precision in curve:
                if math.fabs(tpr - lasttpr) > mindiff:
                    lasttpr = tpr
    #            if tpr > 0.0 and tpr < 1.0 and tpr % 0.001 < 5e-5: #min(tpr % 0.001, math.fabs(0.001 - tpr % 0.001)) < 1e-7:
    #                w.writerow([tpr, precision])
                    w.writerow([tpr, precision])
