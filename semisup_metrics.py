# This code is provided for ICML review purposes only. Please do not distribute.
#
# After review, everything will be made publicly available under a BSD license.

import numpy as np
import optunity
import operator as op
import math
import collections
import array

_lb_ub = collections.namedtuple("lb_ub", ["lower", "upper"])

class ContingencyTable(object):
    """Class to model a contingency table."""

    def __init__(self, TP=0, FP=0, TN=0, FN=0):
        self._TP = TP
        self._FP = FP
        self._TN = TN
        self._FN = FN

    def get_TP(self):
        return self._TP

    def set_TP(self, value):
        self._TP = value

    def del_TP(self):
        raise NotImplementedError

    def get_FP(self):
        return self._FP

    def set_FP(self, value):
        self._FP = value

    def del_FP(self):
        raise NotImplementedError


    def get_TN(self):
        return self._TN

    def set_TN(self, value):
        self._TN = value

    def del_TN(self):
        raise NotImplementedError

    def get_FN(self):
        return self._FN

    def set_FN(self, value):
        self._FN = value

    def del_FN(self):
        raise NotImplementedError

    TP = property(get_TP, set_TP, del_TP, "number of true positives")
    FP = property(get_FP, set_FP, del_FP, "number of false positives")
    TN = property(get_TN, set_TN, del_TN, "number of true negatives")
    FN = property(get_FN, set_FN, del_FN, "number of false negatives")

    @property
    def N(self):
        """The number of negatives (true labels)."""
        return self.TN + self.FP

    @property
    def P(self):
        """The number of positives (true labels)."""
        return self.TP + self.FN

    def __add__(self, other):
        return ContingencyTable(TP = self.TP + other.TP,
                                FP = self.FP + other.FP,
                                TN = self.TN + other.TN,
                                FN = self.FN + other.FN)

    def __str__(self):
        return "TP=%d, FP=%d, TN=%d, FN=%d" % (self.TP, self.FP, self.TN, self.FN)

def FPR(ct):
    """Computes false positive rate based on a contingency table.

    :param ct: a contingency table
    :type ct: ContingencyTable

    """
    return float(ct.FP) / ct.N

def TPR(ct):
    """Computes true positive rate based on a contingency table.

    :param ct: a contingency table
    :type ct: ContingencyTable

    """
    return float(ct.TP) / ct.P

def precision(ct):
    """Computes precision based on a contingency table.

    :param ct: a contingency table
    :type ct: ContingencyTable

    """
    return float(ct.TP) / (ct.TP + ct.FP)

def accuracy(ct):
    """Computes accuracy based on a contingency table.

    :param ct: a contingency table
    :type ct: ContingencyTable

    """
    return float(ct.TP + ct.TN) / (ct.N + ct.P)


def compute_ecdf_curve(ranking, maxrank=None):
    sort = sorted(ranking)
    if maxrank is None:
        maxrank = sort[-1]
    numpos = len(ranking)

    curve = [(rank, float(idx + 1) / numpos)
             for idx, rank in enumerate(sort)]

    if curve[0][0] > 0:
        curve.insert(0, (0, 0.0))
    curve.append((maxrank, 1.0))
    curve = list(zip(*_remove_duplicates(*zip(*curve))))
    return curve


def dkw_bounds(labels, decision_values, ci_width=0.95, presorted=False):
    """Computes the Dvoretzky-Kiefer-Wolfowitz confidence band for the empirical CDF
    at the given alpha level for the given ranking.

    :param ranks: a of ranks
    :type ranks: list
    :param alpha: alpha level to use
    :type alpha: float
    :param maxrank: the maximum rank, used to compute the empirical CDF
    :type maxrank: float
    :returns: a lower and upper bound on the CDF, both as list of (rank, TPR) tuples

    This confidence band is based on the DKW inequality:

        ..math:: P \left( \sup_x \left| F(x) - \hat(F)_n(X) \right| > \epsilon \right) \leq 2e^{-2n\epsilon^2}

    with:

        ..math:: \epsilon = \sqrt{\frac{1}{2n}ln\big(\frac{2}{\alpha}\big)}

    References:

    - A., Dvoretzky; Kiefer, J.; Wolfowitz, J. (1956). "Asymptotic minimax character of the sample distribution function and of the classical multinomial estimator". The Annals of Mathematical Statistics 27 (3): 642-669.

    - Massart, P. (1990). "The tight constant in the Dvoretzky-Kiefer-Wolfowitz inequality". The Annals of Probability: 1269-1283.

    """
    maxrank = len(labels)
    if presorted:
        sorted_labels = labels[:]
    else:
        sort_labels, _ = zip(*sorted(zip(labels, decision_values),
                                     key=op.itemgetter(1), reverse=True))
    known_pos_ranks = [idx for idx, lab in enumerate(sort_labels) if lab]
    ecdf = compute_ecdf_curve(known_pos_ranks)#, maxrank)

    alpha = 1.0 - ci_width
    epsilon = math.sqrt(math.log(2.0 / alpha) / (2 * len(known_pos_ranks)))

    new_ranks, TPRs = zip(*ecdf)
    lower = [max(0, TPR - epsilon) for TPR in TPRs]
    upper = [min(1, TPR + epsilon) for TPR in TPRs]
    return _lb_ub(lower=zoh(new_ranks, lower, presorted=True),
                  upper=zoh(new_ranks, upper, presorted=True))


def _bootstrap_ecdf_iter(ranks_arr, n, uniques):
    ecdf = compute_ecdf_curve(np.random.choice(ranks_arr, size=n, replace=True))

    TPRs = []
    ecdf_it = 0
    for rank in uniques:
        if rank < ecdf[ecdf_it][0]:
            if TPRs: TPRs.append(TPRs[-1])
            else: TPRs.append(0.0)
        else:
            TPRs.append(ecdf[ecdf_it][1])
            if TPRs[-1] == 1.0: break
            ecdf_it += 1
    TPRs.extend([1.0] * (len(uniques) - len(TPRs)))
    return TPRs

def _bootstrap_band(ranks, ci_width=0.95, nboot=1000, pmap=optunity.pmap):
    n = len(ranks)
    unique_ranks = sorted(set(ranks))
    ranks_arr = np.array(ranks)

    all_TPRs = pmap(lambda x: _bootstrap_ecdf_iter(ranks_arr, n, unique_ranks),
                    range(nboot))

    idx_lower = int(float(nboot) * (1 - ci_width) / 2)
    idx_upper = nboot - idx_lower
    ecdf_lower = []
    ecdf_upper = []

    for TPRs, rank in zip(zip(*all_TPRs), unique_ranks):
        TPRs = sorted(TPRs)
        ecdf_lower.append((rank, TPRs[idx_lower]))
        ecdf_upper.append((rank, TPRs[idx_upper]))

    return _lb_ub(lower=ecdf_lower, upper=ecdf_upper)

def _remove_duplicates(xs, ys, last=True):
    new_xs = []
    new_ys = []
    n = len(xs)
    idx = 0
    while idx < n:
        new_xs.append(xs[idx])
        new_ys.append(ys[idx])
        idx += 1
        while last and idx < n and new_xs[-1] == xs[idx]:
            new_ys[-1] = max(new_ys[-1], ys[idx])
            idx += 1

    return new_xs, new_ys


def zoh(xs, ys, presorted=False):
    """Returns a function of x that interpolates with zero-order hold between known values.

    :param xs: known x values
    :type xs: list of floats
    :param ys: known y values
    :type ys: list of floats
    :param presorted: whether xs are already sorted (ascending)
    :type presorted: bool
    :returns: f(x) which returns a linear interpolation of the known values

    If extrapolation is attempted, the bound on known values is returned (e.g. max(ys) or min(ys)).

    >>> f = zoh([0.0, 2.0], [-1.0, 1.0])
    >>> f(-1.0)
    -1.0
    >>> f(0.0)
    -1.0
    >>> f(1.0)
    -1.0
    >>> f(1.0)
    1.0
    >>> f(2.0)
    1.0

    """
    if not presorted:
        xs, ys = zip(*sorted(zip(xs, ys)))
        xs = array.array('f', xs)
        ys = array.array('f', ys)

    nx = len(xs)
    xs, ys = _remove_duplicates(xs, ys)

    def f(x):
        if f.xs[f.curr_idx] > x: f.curr_idx = 0
        idx = f.curr_idx

        if f.xs[0] > x: return f.ys[0]
        if f.xs[-1] < x: return f.ys[-1]
        while idx < nx and f.xs[idx] <= x:
            idx += 1

        return f.ys[idx - 1]

    f.curr_idx = 0
    f.xs = xs
    f.ys = ys
    return f

def bootstrap_ecdf_bounds(labels, decision_values,
                          ci_width=0.95, nboot=1000, pmap=map, presorted=False):
    """Returns CI bounds on the rank CDF of known positives based on the bootstrap.

    :param labels: the labels, such that True=known positive, False=known negative, None=unlabeled
    :type labels: iterable
    :param decision_values: the decision values
    :type decision_values: iterable

    :returns: two callables: lower and upper CI bound for each rank
    """

    if presorted: known_pos_ranks = [rank for rank, label in enumerate(labels) if label]
    else:
        sort_labels, sort_dv = zip(*sorted(zip(labels, decision_values),
                                           key=op.itemgetter(1),
                                           reverse=True))
        known_pos_ranks = [idx for idx, lab in enumerate(sort_labels) if lab]

    ecdf_bounds = _bootstrap_band(known_pos_ranks, ci_width=ci_width,
                                  nboot=nboot, pmap=pmap)

    return _lb_ub(lower=zoh(*zip(*ecdf_bounds.lower)),
                  upper=zoh(*zip(*ecdf_bounds.upper)))


def find_cutoff(ranks, n, last_cut, cutoff_rank):
    cut = last_cut
    while cut < n and ranks[cut] <= cutoff_rank: # TODO: insert bisection
        cut += 1
    return cut

def surrogates_contingency(num_unl_less, nunl, TPR, npos_in_unl, lower=True):
    # the number of surrogate positive labels that must be assigned
    theta = float(TPR) * npos_in_unl
    theta = math.floor(theta) if lower else math.ceil(theta)
    theta = int(theta)

    TP = min(num_unl_less, theta)

    # account for the fact that all remaining positives (if any) must fit in the tail
    ntail = nunl - num_unl_less
    TP = max(TP, npos_in_unl - ntail)

    FN = npos_in_unl - TP
    FP = num_unl_less - TP
    TN = nunl - npos_in_unl - FP
    return ContingencyTable(TP=TP, FP=FP, TN=TN, FN=FN)


def compute_contingency_tables(labels, decision_values, reference_lb=None,
                               reference_ub=None, ranks=None,
                               beta=0.0, presorted=False):
    """
    Computes an approximate contingency table at each rank, or at the specified ranks,
    corresponding to a lower and/or upper bound on false positive rate.

    :param labels: the labels, such that True=known positive, False=known negative, None=unlabeled
    :type labels: iterable
    :param decision_values: the decision values
    :type decision_values: iterable
    :param reference_lb: the lower bound reference rank distribution to approximate as function of rank
    :type reference_lb: callable
    :param reference_ub: the upper bound reference rank distribution to approximate as function of rank
    :type reference_ub: callable
    :param beta: fraction of positives in the unlabeled set
    :type beta: double
    :param lower: compute contingency tables for lower bound of FPR?
    :type lower: boolean
    :param upper: compute contingency tables for upper bound of FPR?
    :type upper: boolean
    :param ranks: compute contingency tables for specified ranks
    :type ranks: list
    :param presorted: are labels and decision values already sorted (by descending decision values)
    :type presorted: boolean

    """
    assert(0. <= beta <= 1.)

    if presorted: sorted_labels = labels
    else:
        sorted_labels = map(op.itemgetter(1),
                            sorted(zip(decision_values, labels),
                                   reverse=True))

    # ranks of known positives/negatives and unlabeled instances
    known_pos_ranks = []
    known_neg_ranks = []
    unlabeled_ranks = []
    for rank, label in enumerate(sorted_labels):
        if label == True: known_pos_ranks.append(rank)
        elif label == False: known_neg_ranks.append(rank)
        else: unlabeled_ranks.append(rank)

    # number of positives, negatives and unlabeled instances
    npos = len(known_pos_ranks)
    nneg = len(known_neg_ranks)
    nunl = len(unlabeled_ranks)
    npos_in_unl = int(beta * nunl)

    # cutoffs in each set to efficiently find the size of head(X,r)
    cut_neg = 0
    cut_unl = 0
    cut_pos = 0

    # sets of contingency tables
    cts_lower = []
    cts_upper = []

    # iterate over ranks
    if ranks: ranks[:] = sorted(ranks)
    else: ranks = list(range(len(labels)))
    for rank in ranks:
        ct = ContingencyTable()

        # find split in known positives and update contingency table
        if known_pos_ranks:
            cut_pos = find_cutoff(known_pos_ranks, npos, cut_pos, rank)
            ct.TP += cut_pos
            ct.FN += npos - cut_pos

        # find split in known negatives and update contingency table
        if known_neg_ranks:
            cut_neg = find_cutoff(known_neg_ranks, nneg, cut_neg, rank)
            ct.FP += cut_neg
            ct.TN += nneg - cut_neg

        # find labeling and corresponding cutoffs in U
        if unlabeled_ranks:
            cut_unl = find_cutoff(unlabeled_ranks, nunl, cut_unl, rank)

            if reference_lb is not None:
                TPR = reference_lb(rank)
                ct_lb = surrogates_contingency(cut_unl, nunl, TPR, npos_in_unl,
                                               lower=True)
                cts_lower.append(ct + ct_lb)

            if reference_ub is not None:
                TPR = reference_ub(rank)
                ct_ub = surrogates_contingency(cut_unl, nunl, TPR, npos_in_unl,
                                               lower=False)
                cts_upper.append(ct + ct_ub)

    return _lb_ub(lower=cts_lower, upper=cts_upper)

def get_contingency_tables(labels, decision_values, beta=0.0, cdf_bounds=None,
                           ci_fun=bootstrap_ecdf_bounds, presorted=False):
    """Computes contingency tables, using any intermediate results when available
    or else from scratch.

    :param labels: the labels, such that True=known positive, False=known negative, None=unlabeled
    :type labels: iterable
    :param decision_values: the decision values
    :type decision_values: iterable
    :param beta: fraction of positives in the unlabeled set
    :type beta: double
    :param cdf_bounds: precomputed bounds on rank CDF of known positives
    :type cdf_bounds: _lb_ub containing lists of (rank, TPR)-pairs
    :param ci_fun: function to compute CDF bounds, ignored if cdf_bounds are given
    :type ci_fun: callable
    :param presorted: are labels and decision values already sorted (by descending decision values)
    :type presorted: boolean

    """
    if presorted:
        sorted_dv = decision_values
        sorted_labels = labels
    else:
        sorted_labels, sorted_dv = zip(*sorted(zip(labels, decision_values),
                                                key=op.itemgetter(1),
                                                reverse=True))

    # compute confidence interval on rank CDF of known positives
    # if no bounds are given
    if not cdf_bounds:
        cdf_bounds = ci_fun(sorted_labels, sorted_dv, presorted=True)

    tables = compute_contingency_tables(labels=sorted_labels,
                                        decision_values=sorted_dv,
                                        reference_lb=cdf_bounds.lower,
                                        reference_ub=cdf_bounds.upper,
                                        beta=beta, presorted=True)
    return tables


def roc_bounds(labels, decision_values, beta=0.0,
               ci_fun=bootstrap_ecdf_bounds, cdf_bounds=None,
               tables=None, presorted=False):
    """Returns bounds on the ROC curve.

    :param labels: the labels, such that True=known positive, False=known negative, None=unlabeled
    :type labels: iterable
    :param decision_values: the decision values
    :type decision_values: iterable
    :param beta: fraction of positives in the unlabeled set
    :type beta: double
    :param cdf_bounds: precomputed bounds on rank CDF of known positives
    :type cdf_bounds: _lb_ub containing lists of (rank, TPR)-pairs
    :param ci_fun: function to compute CDF bounds, ignored if cdf_bounds are given
    :type ci_fun: callable
    :param presorted: are labels and decision values already sorted (by descending decision values)
    :type presorted: boolean

    """

    # compute contingency tables if none are given
    if not tables: tables = get_contingency_tables(labels, decision_values,
                                                   beta=beta, ci_fun=ci_fun,
                                                   cdf_bounds=cdf_bounds,
                                                   presorted=presorted)

    # LB on FPR corresponds to UB on PR curve and vice versa
    return _lb_ub(lower=sorted(map(lambda t: (FPR(t), TPR(t)), tables.lower)),
                  upper=sorted(map(lambda t: (FPR(t), TPR(t)), tables.upper)))

def pr_bounds(labels, decision_values, beta=0.0,
              ci_fun=bootstrap_ecdf_bounds, cdf_bounds=None,
              tables=None, presorted=False):
    """Returns bounds on the PR curve.

    :param labels: the labels, such that True=known positive, False=known negative, None=unlabeled
    :type labels: iterable
    :param decision_values: the decision values
    :type decision_values: iterable
    :param beta: fraction of positives in the unlabeled set
    :type beta: double
    :param cdf_bounds: precomputed bounds on rank CDF of known positives
    :type cdf_bounds: _lb_ub containing lists of (rank, TPR)-pairs
    :param ci_fun: function to compute CDF bounds, ignored if cdf_bounds are given
    :type ci_fun: callable
    :param presorted: are labels and decision values already sorted (by descending decision values)
    :type presorted: boolean

    """
    # compute contingency tables if none are given
    if not tables: tables = get_contingency_tables(labels, decision_values,
                                                   beta=beta, ci_fun=ci_fun,
                                                   cdf_bounds=cdf_bounds,
                                                   presorted=presorted)

    # LB on FPR corresponds to UB on PR curve and vice versa
    return _lb_ub(lower=sorted(map(lambda t: (TPR(t), precision(t)), tables.lower)),
                  upper=sorted(map(lambda t: (TPR(t), precision(t)), tables.upper)))

def auc(curve):
    """Computes the area under the specified curve.

    :param curve: a curve, specified as a list of (x, y) tuples
    :type curve: [(x, y), ...]

    """
    area = 0.0
    for i in range(len(curve) - 1):
        x1, y1 = curve[i]
        x2, y2 = curve[i + 1]
        if y1 is None:
            y1 = 0.0
        area += float(min(y1, y2)) * float(x2 - x1) + math.fabs(float(y2 - y1)) * float(x2 - x1) / 2

    return area
