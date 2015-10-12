from __future__ import print_function
import operator as op
import random
import semisup_metrics as ss
import optunity.metrics as metrics
from matplotlib import pylab as plt
import pickle
import sys
import util

known_pos_frac = 0.05
known_neg_frac = 0.0

nboot = 2000
ci_width = 0.95

band = 'bootstrap'

print("This file allows you to recreate the figure 5 (based on covtype) and perform similar computations on another real data set (sensit).", file=sys.stdout)
print("You can also change the configuration (fraction of known positives, fraction of known negatives, beta).\n", file=sys.stdout)
print("Please specify which data set to use (a, b or c): (a) covtype, (b) sensit, (c) Gaussians.", file=sys.stdout)

choices = {'a': 'covtype', 'b': 'sensit', 'c': 'gaussians'}
choice = None
while not choice:
    value = sys.stdin.readline()[:-1]
    choice = choices.get(value, None)
    if not choice: print("Invalid input, please select a or b.", file=sys.stdout)

print("Leave blank if you want to use the configuration used to generate the paper's figures.")
line = sys.stdin.readline()[:-1]
if len(line):
    print("Specify the fraction of known positives or leave blank (default 0.05).")
    line = sys.stdin.readline()[:-1]
    if len(line): known_pos_frac = float(line)
    assert(0.0 <= known_pos_frac < 1.0)

    print("Specify the fraction of known negatives or leave blank (default 0.0).")
    line = sys.stdin.readline()[:-1]
    if len(line): known_neg_frac = float(line)
    assert(0.0 <= known_neg_frac < 1.0)

    print("Specify the type of confidence band to use (a or b) or leave blank: (a) bootstrap, (b) DKW")
    line = sys.stdin.readline()[:-1]
    if len(line) and line == 'b': band = 'DKW'
    else: band = 'bootstrap'

    print("Specify the width of the confidence band or leave blank (default 0.95).")
    line = sys.stdin.readline()[:-1]
    if len(line): ci_width = float(line)
    assert(0.0 < ci_width < 1.0)

    if band == 'bootstrap':
        print("Specify the width of the number of bootstrap resamples or leave blank (default 2000).")
        line = sys.stdin.readline()[:-1]
        if len(line): nboot = int(line)
        assert(nboot > 0)

print("Generating figures for %s." % choice, file=sys.stdout)
if band == 'bootstrap': print("Configuration: known_pos_frac=%1.3f, known_neg_frac=%1.3f, ci_width=%1.3f, estimating confidence band via bootstrap (nboot=%d)." % (known_pos_frac, known_neg_frac, ci_width, nboot))
else: print("Configuration: known_pos_frac=%1.3f, known_neg_frac=%1.3f, ci_width=%1.3f, estimating confidence band via DKW." % (known_pos_frac, known_neg_frac, ci_width))

def induce_partial_labels(labs, known_pos_frac, known_neg_frac):
    true_labels = list(labs)
    labels = list(labs)
    total_pos = float(sum(true_labels))
    total_neg = float(len(true_labels) - total_pos)
    labeled_pos = int(total_pos * known_pos_frac)
    labeled_neg = int(total_neg * known_neg_frac)
    pos_idx = [idx for idx, lab in enumerate(true_labels) if lab]
    neg_idx = [idx for idx, lab in enumerate(true_labels) if not lab]
    relabel_pos_idx = random.sample(pos_idx, int(total_pos - labeled_pos))
    relabel_neg_idx = random.sample(neg_idx, int(total_neg - labeled_neg))

    for i in relabel_pos_idx:
        labels[i] = None
    for i in relabel_neg_idx:
        labels[i] = None

    beta = float(true_labels.count(True) - labels.count(True)) / labels.count(None)
    return true_labels, labels, beta

print("+ loading data", file=sys.stdout)
if choice == 'gaussians':
    dist_config = {'mean_pos': 1.0, 'sigma_pos': 1.0,
                   'mean_neg': 0.0, 'sigma_neg': 1.0}

    num_pos = 2000
    num_neg = 0
    num_unl = 10000
    beta = 0.2
    guessed_pfrac = 0.2

    num_neg_in_unl = int(num_unl * (1 - beta))
    num_pos_in_unl = int(num_unl * beta)
    labels = [True] * num_pos + [None] * num_pos_in_unl + [False] * num_neg + [None] * num_neg_in_unl
    true_labels = [True] * (num_pos + num_pos_in_unl) + [False] * (num_neg + num_neg_in_unl)
    decision_values = util.generate_pos_class_decvals(num_pos + num_pos_in_unl, dist_config)
    decision_values += util.generate_neg_class_decvals(num_neg + num_neg_in_unl, dist_config)

else:
    with open('data.pkl', 'rb') as f:
        data = pickle.load(f)
        labels = list(data[choice]['labels'])
        decision_values = list(data[choice]['decision_values'])

    print("+ inducing partial labels at random", file=sys.stdout)
    # induce partial labels at random
    true_labels, labels, beta = induce_partial_labels(labels,
                                                    known_pos_frac,
                                                    known_neg_frac)
    # identify latent positives, which is ofcourse impossible in practice

lps = map(lambda x, y: x == True and y == None, true_labels, labels)
print("++ beta is now %1.3f" % beta)

print("+ computing rank CDF of known positives and confidence band")
sort_labels, sort_dv = zip(*sorted(zip(labels, decision_values),
                                    key=op.itemgetter(1), reverse=True))
known_pos_ranks = [idx for idx, lab in enumerate(sort_labels) if lab]
known_pos_ecdf = ss.compute_ecdf_curve(known_pos_ranks)

# compute rank CDF of latent positives (impossible in practice)
sort_lps, _ = zip(*sorted(zip(lps, decision_values),
                                key=op.itemgetter(1), reverse=True))
latent_pos_ranks = [idx for idx, lab in enumerate(sort_lps) if lab]
latent_pos_ecdf = ss.compute_ecdf_curve(latent_pos_ranks)

# true ROC curve and ROC curve assuming beta = 0
roc_auc_true, roc_true = metrics.roc_auc(true_labels, decision_values, return_curve=True)
roc_auc_neg, roc_neg = metrics.roc_auc(labels, decision_values, return_curve=True)

# true PR curve and PR curve assuming beta = 0
pr_auc_true, pr_true = metrics.pr_auc(true_labels, decision_values, return_curve=True)
pr_auc_neg, pr_neg = metrics.pr_auc(labels, decision_values, return_curve=True)

# compute bounds
if band == 'bootstrap':
    cdf_bounds = ss.bootstrap_ecdf_bounds(labels, decision_values, nboot=nboot, ci_width=ci_width)
else:
    cdf_bounds = ss.dkw_bounds(labels, decision_values, ci_width=ci_width)

print("+ computing contingency tables", file=sys.stdout)
# we use presorted labels & decision values for efficiency, but this is not necessary
# we use precomputed cdf_bounds, but again this is optional
tables = ss.compute_contingency_tables(labels=sort_labels, decision_values=sort_dv,
                                       reference_lb=cdf_bounds.lower,
                                       reference_ub=cdf_bounds.upper,
                                       beta=beta, presorted=True)


print('+ computing approximate ROC curves')
# we use precomputed contingency tables again for efficiency
roc_bounds = ss.roc_bounds(sort_labels, sort_dv, beta=beta, tables=tables, presorted=True)

print('+ computing approximate PR curves')
# we use precomputed contingency tables again for efficiency
pr_bounds = ss.pr_bounds(sort_labels, sort_dv, beta=beta, tables=tables, presorted=True)

def plot_proxy():
    p = plt.Rectangle((0, 0), 0, 0, color='blue', alpha=0.4)
    ax = plt.gca()
    ax.add_patch(p)
    return p

print('+ plotting results')
xs = list(range(len(labels)))
plt.figure(1)
plt.fill_between(xs, list(map(cdf_bounds.lower, xs)), list(map(cdf_bounds.upper, xs)), color='blue', alpha=0.4)
plt.plot(*zip(*known_pos_ecdf), color='black', linestyle='dashed', linewidth=2)
plt.plot(*zip(*latent_pos_ecdf), color='black', linewidth=2)
plot_proxy()
plt.xlabel('rank')
plt.ylabel('TPR')
plt.legend(['known positives', 'latent positives', 'expected region'], loc=4)
plt.title('Rank CDF')
plt.draw()

xs = [float(x) / 100 for x in range(101)]
roc_up = ss.zoh(*zip(*roc_bounds.upper))
roc_lo = ss.zoh(*zip(*roc_bounds.lower))
plt.figure(2)
plt.plot(*zip(*roc_true), color='black', linewidth=2)
plt.fill_between(xs, list(map(roc_lo, xs)), list(map(roc_up, xs)), color='blue', alpha=0.4)
plt.plot(*zip(*roc_neg), color='black', linestyle='dashed')
plot_proxy()
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend(['true curve', 'beta=0', 'expected region'], loc=4)
plt.title('Receiver Operating Characteristic curve')
plt.draw()

pr_up = ss.zoh(*zip(*pr_bounds.upper))
pr_lo = ss.zoh(*zip(*pr_bounds.lower))
plt.figure(3)
plt.plot(*zip(*pr_true), color='black', linewidth=2)
plt.plot(*zip(*pr_neg), color='black', linestyle='dashed')
plt.fill_between(xs, list(map(pr_lo, xs)), list(map(pr_up, xs)), color='blue', alpha=0.4)
plot_proxy()
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(['true curve', 'beta=0', 'expected region'], loc=1)
plt.title('Precision-Recall curve')
plt.show()

dist_to_lo = 0.01
dist_to_up = 0.01
betahat, evals = ss.estimate_beta(labels=sort_labels, decision_values=sort_dv,
                                  cdf_bounds=cdf_bounds, presorted=True,
                                  dist_to_lo=dist_to_lo, dist_to_up=dist_to_up)

print(betahat)


betas, los, ups = ss.beta_gap(labels=sort_labels, decision_values=sort_dv,
                              cdf_bounds=cdf_bounds, presorted=True,
                              dist_to_lo=dist_to_lo, dist_to_up=dist_to_up)


plt.figure(4)
plt.plot(betas, los, 'b--')
plt.plot(betas, ups, 'r--')
plt.plot([betahat.lower, betahat.lower], [min(los + ups), max(los + ups)], 'b', linewidth=3)
plt.plot([betahat.upper, betahat.upper], [min(los + ups), max(los + ups)], 'r', linewidth=3)
plt.plot([beta, beta], [min(los + ups), max(los + ups)], 'k', linewidth=3)
plt.plot([min(betas), max(betas)], [0, 0], 'k')
plt.plot(evals.lower, [0.0] * len(evals.lower), 'bx', markersize=10)
plt.plot(evals.upper, [0.0] * len(evals.upper), 'rx', markersize=10)
plt.xlabel(r'$\hat{\beta}$')
plt.ylabel('gap')
plt.legend(['lower gap', 'upper gap', r'$\hat{\beta}_{lo}$', r'$\hat{\beta}_{up}$', r'$\beta$'])
plt.axis('tight')
plt.show()
