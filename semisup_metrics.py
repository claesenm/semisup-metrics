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

    def __sub__(self, other):
        return ContingencyTable(TP = self.TP - other.TP,
                                FP = self.FP - other.FP,
                                TN = self.TN - other.TN,
                                FN = self.FN - other.FN)

    def __str__(self):
        return "TP=%d, FP=%d, TN=%d, FN=%d" % (self.TP, self.FP, self.TN, self.FN)


def sort_labels_by_dv(labels, decision_values):
    """Returns labels and decision values, sorted by descending decision values.

    :param labels: ground truth labels
    :type labels: iterable
    :param decision_values: decision values
    :type decision_values: iterable
    :returns: sorted list of labels and decision_values

    Note that: len(labels) == len(decision_values).

    """
    sorted_labels, sorted_dv = zip(*sorted(zip(labels, decision_values),
                                           key=op.itemgetter(1),
                                           reverse=True))
    return sorted_labels, sorted_dv


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
    if ct.TP + ct.FP == 0: return 0.0
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
        sort_labels, _ = sort_labels_by_dv(labels, decision_values)
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

    xs, ys = _remove_duplicates(xs, ys)
    nx = len(xs)

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
        sort_labels, sort_dv = sort_labels_by_dv(labels, decision_values)
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
        sorted_labels, _ = sort_labels_by_dv(labels, decision_values)

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
                           ci_fun=bootstrap_ecdf_bounds, presorted=False, switch_labels=False):
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
    :param switch_labels: should labels be switched to improve estimates? (default: False)
        True: switch labels,
        False: do not switch labels,
        None: switch if it improves results
    :type switch_labels: bool or None

    """

    if presorted:
        sorted_dv = decision_values
        sorted_labels = labels
    else:
        sorted_labels, sorted_dv = sort_labels_by_dv(labels, decision_values)

    switched_labels = False
    if switch_labels:
        sorted_labels = map(lambda x: (not x) if x is not None else None,
                            sorted_labels)
        beta = 1.0 - beta
        switched_labels = True

    elif switch_labels is None: # determine whether it is useful to switch labels
        num_known_pos = 0
        num_known_neg = 0
        for label in sorted_labels:
            if label is None: continue
            elif label: num_known_pos += 1
            else: num_known_neg += 1

        if num_known_neg > num_known_pos:
            sorted_labels = map(lambda x: (not x) if x is not None else None,
                                sorted_labels)
            beta = 1.0 - beta
            switched_labels = True

    # labels can only be switched when no references on rank CDF are given
    # otherwise the references might be wrong
    assert(not switch_labels or not cdf_bounds)

    # compute confidence interval on rank CDF of known positives
    # if no bounds are given
    if not cdf_bounds:
        cdf_bounds = ci_fun(sorted_labels, sorted_dv, presorted=True)

    if switched_labels:
        ref_lb = cdf_bounds.upper
        ref_ub = cdf_bounds.lower
    else:
        ref_lb = cdf_bounds.lower
        ref_ub = cdf_bounds.upper

    tables = compute_contingency_tables(labels=sorted_labels,
                                        decision_values=sorted_dv,
                                        reference_lb=ref_lb,
                                        reference_ub=ref_ub,
                                        beta=beta, presorted=True)

    # restructure contingency tables if we switched labels
    # restructuring implies converting negatives <> positives
    # and switching lower and upper bounds
    if switched_labels:
        reorder_ct = lambda ct: ContingencyTable(TP=ct.FP, FP=ct.TP, TN=ct.FN, FN=ct.TN)
        # TODO: is this correct or should lower/upper be switched?
        return _lb_ub(lower=list(map(reorder_ct, tables.lower)),
                      upper=list(map(reorder_ct, tables.upper)))

    return tables


def roc_bounds(labels, decision_values, beta=0.0, beta_ci=None,
               ci_fun=bootstrap_ecdf_bounds, cdf_bounds=None,
               tables=None, presorted=False, switch_labels=False):
    """Returns bounds on the ROC curve.

    :param labels: the labels, such that True=known positive, False=known negative, None=unlabeled
    :type labels: iterable
    :param decision_values: the decision values
    :type decision_values: iterable
    :param beta: fraction of positives in the unlabeled set
    :type beta: double
    :param beta_ci: bounds on confidence interval of beta
    :type beta_ci: (lb, ub)
    :param cdf_bounds: precomputed bounds on rank CDF of known positives
    :type cdf_bounds: _lb_ub containing lists of (rank, TPR)-pairs
    :param ci_fun: function to compute CDF bounds, ignored if cdf_bounds are given
    :type ci_fun: callable
    :param presorted: are labels and decision values already sorted (by descending decision values)
    :type presorted: boolean
    :param switch_labels: should labels be switched to improve estimates? (default: False)
        True: switch labels,
        False: do not switch labels,
        None: switch if it improves results
    :type switch_labels: bool or None

    """

    if beta_ci:
        result_lo = roc_bounds(labels, decision_values, beta=beta_ci[0],
                               ci_fun=ci_fun, cdf_bounds=cdf_bounds,
                               tables=tables, presorted=presorted,
                               switch_labels=switch_labels)
        result_up = roc_bounds(labels, decision_values, beta=beta_ci[1],
                               ci_fun=ci_fun, cdf_bounds=cdf_bounds,
                               tables=tables, presorted=presorted,
                               switch_labels=switch_labels)
        return _lb_ub(lower=result_lo.lower, upper=result_up.upper)
    # compute contingency tables if none are given
    if not tables: tables = get_contingency_tables(labels, decision_values,
                                                   beta=beta, ci_fun=ci_fun,
                                                   cdf_bounds=cdf_bounds,
                                                   presorted=presorted,
                                                   switch_labels=switch_labels)

    # LB on FPR corresponds to UB on PR curve and vice versa
    return _lb_ub(lower=sorted(map(lambda t: (FPR(t), TPR(t)), tables.lower)),
                  upper=sorted(map(lambda t: (FPR(t), TPR(t)), tables.upper)))

def pr_bounds(labels, decision_values, beta=0.0, beta_ci=None,
              ci_fun=bootstrap_ecdf_bounds, cdf_bounds=None,
              tables=None, presorted=False, switch_labels=False):
    """Returns bounds on the PR curve.

    :param labels: the labels, such that True=known positive, False=known negative, None=unlabeled
    :type labels: iterable
    :param decision_values: the decision values
    :type decision_values: iterable
    :param beta: fraction of positives in the unlabeled set
    :type beta: double
    :param beta_ci: bounds on confidence interval of beta
    :type beta_ci: (lb, ub)
    :param cdf_bounds: precomputed bounds on rank CDF of known positives
    :type cdf_bounds: _lb_ub containing lists of (rank, TPR)-pairs
    :param ci_fun: function to compute CDF bounds, ignored if cdf_bounds are given
    :type ci_fun: callable
    :param presorted: are labels and decision values already sorted (by descending decision values)
    :type presorted: boolean
    :param switch_labels: should labels be switched to improve estimates? (default: False)
        True: switch labels,
        False: do not switch labels,
        None: switch if it improves results
    :type switch_labels: bool or None

    """
    if beta_ci:
        result_lo = pr_bounds(labels, decision_values, beta=beta_ci[0],
                              ci_fun=ci_fun, cdf_bounds=cdf_bounds,
                              tables=tables, presorted=presorted,
                              switch_labels=switch_labels)
        result_up = pr_bounds(labels, decision_values, beta=beta_ci[1],
                              ci_fun=ci_fun, cdf_bounds=cdf_bounds,
                              tables=tables, presorted=presorted,
                              switch_labels=switch_labels)
        return _lb_ub(lower=result_lo.lower, upper=result_up.upper)

    # compute contingency tables if none are given
    if not tables: tables = get_contingency_tables(labels, decision_values,
                                                   beta=beta, ci_fun=ci_fun,
                                                   cdf_bounds=cdf_bounds,
                                                   presorted=presorted,
                                                   switch_labels=switch_labels)

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


def compute_labeled_cts(sorted_labels):
    """Computes contingency tables for the labeled instances at each rank."""

    # count the number of occurances of given target label
    count_target = lambda target: reduce(lambda accum, label: accum + (label == target),
                                         sorted_labels, 0)
    nknownpos = count_target(True)
    nknownneg = count_target(False)
    cts = [ContingencyTable(TP=0, FP=0, TN=nknownneg, FN=nknownpos)]

    for label in sorted_labels:
        if label == None: diffct = ContingencyTable(0, 0, 0, 0)
        elif label == True: diffct = ContingencyTable(TP=1, FN=-1)
        else: diffct = ContingencyTable(FP=1, TN=-1)
        cts.append(cts[-1] + diffct)

    return cts[1:]


def bisect(f, lo, up, maxeval=15):
    """Standard bisection to find x in [lo, up] such that f(x) <= 0.0."""
    neval = 0
    flo = f(lo)
    fup = f(up)
    if flo == 0.0: return lo
    if fup == 0.0: return up

    evals = [lo, up]

    # make sure we initialize correctly
    assert((flo < 0.0 and fup > 0.0) or (flo > 0.0 and fup < 0.0))
    if flo > 0.0:
        tmp = up
        up = lo
        lo = tmp
        tmp = fup
        fup = flo
        fup = tmp

    # recursion
    for neval in xrange(2, maxeval):
        new = float(lo + up) / 2
        evals.append(new)
        fnew = f(new)
        if fnew > 0: up = new
        elif fnew < 0: lo = new
        else: return new, evals

    return lo, evals


def estimate_beta(labels, decision_values,
                  ci_fun=bootstrap_ecdf_bounds, cdf_bounds=None,
                  tables=None, presorted=False, switch_labels=False,
                  beta_lo=0.01, beta_up=0.9,
                  dist_to_lo=0.01, dist_to_up=0.01,
                  estimate_lower=True, estimate_upper=True):

    # if we need not estimate anything, calling this function is a waste of time
    assert(estimate_lower or estimate_upper)

    # sort labels and decision values
    sorted_labels, sorted_dv = sort_labels_by_dv(labels, decision_values)
    labeled_cts = compute_labeled_cts(sorted_labels)

    #known_pos_ranks = [idx for idx, lab in enumerate(sorted_labels) if lab]
    #known_pos_ecdf = compute_ecdf_curve(known_pos_ranks)#, maxrank)
    #known_pos_ranks, known_pos_tprs = zip(*known_pos_ecdf)
    #times = lambda x: lambda y: x * y
    #cdf_bounds = _lb_ub(lower=zoh(known_pos_ranks, map(times(1.0 - dist_to_lo),
    #                                                   known_pos_tprs),
    #                              presorted=True),
    #                    upper=zoh(known_pos_ranks, map(times(1.0 + dist_to_up),
    #                                                   known_pos_tprs),
    #                              presorted=True))

    auc_from_cts = lambda cts: auc(sorted(map(lambda t: (TPR(t), precision(t)), cts)))

    # function that computes the difference in AUROC based only
    # on unlabeled instances and the AUROC based on all instances
    # if the former is largest, beta is too high and vice versa.
    # (this assumes a model better than random,
    # such that higher beta implies higher AUROC as per the manuscript)
    def beta_lb_fun(beta):
        # contingency tables based on all instances, labeled and unlabeled
        tables = get_contingency_tables(sorted_labels, sorted_dv,
                                        beta=beta, ci_fun=ci_fun,
                                        cdf_bounds=cdf_bounds,
                                        presorted=True,
                                        switch_labels=switch_labels)

        # sanity check
        assert(len(tables.upper) == len(labeled_cts))

        # AUC for all instances
        full_auc = auc_from_cts(tables.upper)

        # to compute AUC for only unlabeled instances
        # we subtract the contingency tables based on labeled instances
        unlabeled_cts = map(lambda x, y: x - y, tables.upper, labeled_cts)
        unlabeled_auc = auc_from_cts(unlabeled_cts)

        # dist_to_lo serves as a margin to account for noise
        # larger dist_to_lo leads to lower beta
        diff = unlabeled_auc - full_auc + dist_to_lo
#        diff = unlabeled_auc - full_auc
        print('%1.3f\t%1.3f' % (beta, diff))
        return diff


    # function that computes the difference in AUROC based only
    # on unlabeled instances and the AUROC based on all instances
    # if the former is largest, beta is too high and vice versa.
    # (this assumes a model better than random,
    # such that higher beta implies higher AUROC as per the manuscript)
    def beta_ub_fun(beta):
        # contingency tables based on all instances, labeled and unlabeled
        tables = get_contingency_tables(sorted_labels, sorted_dv,
                                        beta=beta, ci_fun=ci_fun,
                                        cdf_bounds=cdf_bounds,
                                        presorted=True,
                                        switch_labels=switch_labels)

        # sanity check
        assert(len(tables.lower) == len(labeled_cts))

        # AUC for all instances
        full_auc = auc_from_cts(tables.lower)

        # to compute AUC for only unlabeled instances
        # we subtract the contingency tables based on labeled instances
        unlabeled_cts = map(lambda x, y: x - y, tables.lower, labeled_cts)
        unlabeled_auc = auc_from_cts(unlabeled_cts)

        # dist_to_lo serves as a margin to account for noise
        # larger dist_to_lo leads to lower beta
        diff = full_auc - dist_to_up - unlabeled_auc
#        diff = full_auc - unlabeled_auc
        print('%1.3f\t%1.3f' % (beta, -diff))
        return diff

    # find zero of beta_lb_fun
    # we don't need many evals because 1% accuracy is plenty
    if estimate_lower:
        try:
            beta_lo, evals_lo = bisect(beta_lb_fun, beta_lo, beta_up, maxeval=15)
        except AssertionError:
            raise ValueError('Failed to compute lower bound on beta, consider increasing dist_to_lo.')
    else: beta_lo = None
    if estimate_upper:
        try:
            beta_up, evals_up = bisect(beta_ub_fun, beta_up, beta_lo, maxeval=15)
        except AssertionError:
            raise ValueError('Failed to compute upper bound on beta, consider increasing dist_to_up.')
    else: beta_up = None
    return _lb_ub(lower=beta_lo, upper=beta_up), _lb_ub(lower=evals_lo, upper=evals_up)



def beta_gap(labels, decision_values,
            ci_fun=bootstrap_ecdf_bounds, cdf_bounds=None,
            tables=None, presorted=False, switch_labels=False,
            dist_to_lo=0.01, dist_to_up=0.01):

    # sort labels and decision values
    sorted_labels, sorted_dv = sort_labels_by_dv(labels, decision_values)
    labeled_cts = compute_labeled_cts(sorted_labels)

    auc_from_cts = lambda cts: auc(sorted(map(lambda t: (TPR(t), precision(t)), cts)))

    # function that computes the difference in AUROC based only
    # on unlabeled instances and the AUROC based on all instances
    # if the former is largest, beta is too high and vice versa.
    # (this assumes a model better than random,
    # such that higher beta implies higher AUROC as per the manuscript)
    def beta_lb_fun(beta):
        # contingency tables based on all instances, labeled and unlabeled
        tables = get_contingency_tables(sorted_labels, sorted_dv,
                                        beta=beta, ci_fun=ci_fun,
                                        cdf_bounds=cdf_bounds,
                                        presorted=True,
                                        switch_labels=switch_labels)

        # sanity check
        assert(len(tables.upper) == len(labeled_cts))

        # AUC for all instances
        full_auc = auc_from_cts(tables.upper)

        # to compute AUC for only unlabeled instances
        # we subtract the contingency tables based on labeled instances
        unlabeled_cts = map(lambda x, y: x - y, tables.upper, labeled_cts)
        unlabeled_auc = auc_from_cts(unlabeled_cts)

        # dist_to_lo serves as a margin to account for noise
        # larger dist_to_lo leads to lower beta
        diff = unlabeled_auc - full_auc + dist_to_lo
        print('%1.3f\t%1.3f' % (beta, diff))
        return diff


    # function that computes the difference in AUROC based only
    # on unlabeled instances and the AUROC based on all instances
    # if the former is largest, beta is too high and vice versa.
    # (this assumes a model better than random,
    # such that higher beta implies higher AUROC as per the manuscript)
    def beta_ub_fun(beta):
        # contingency tables based on all instances, labeled and unlabeled
        tables = get_contingency_tables(sorted_labels, sorted_dv,
                                        beta=beta, ci_fun=ci_fun,
                                        cdf_bounds=cdf_bounds,
                                        presorted=True,
                                        switch_labels=switch_labels)

        # sanity check
        assert(len(tables.lower) == len(labeled_cts))

        # AUC for all instances
        full_auc = auc_from_cts(tables.lower)

        # to compute AUC for only unlabeled instances
        # we subtract the contingency tables based on labeled instances
        unlabeled_cts = map(lambda x, y: x - y, tables.lower, labeled_cts)
        unlabeled_auc = auc_from_cts(unlabeled_cts)

        # dist_to_lo serves as a margin to account for noise
        # larger dist_to_lo leads to lower beta
        diff = full_auc - dist_to_up - unlabeled_auc
        print('%1.3f\t%1.3f' % (beta, -diff))
        return diff

    betas = [0.01 * i for i in range(1, 90, 3)]
    los = map(beta_lb_fun, betas)
    ups = map(beta_ub_fun, betas)
    return betas, los, ups



#def estimate_beta(labels, probabilities):
#    """
#    Estimates the fraction of latent positives in the unlabeled set (beta)
#    based on the predictions of a trained classifier on a validation set.

#    :param labels: the labels (True means positive, False means negative, None means unlabeled)
#    :type labels: bool or None
#    :param probabilities: list of predicted probabilities
#    :type probabilities: iterable of floats in [0, 1]

#    :returns: estimate of beta (float)

#    This estimation approach is based on the first estimator for 'c' as given in:

#    Elkan, Charles, and Keith Noto. "Learning classifiers from only positive and unlabeled data."
#    Proceedings of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining.
#    ACM, 2008.

#    .. warning:

#        This function raises a ValueError if:
#            - no unlabeled instances are specified
#            - no known positives are specified
#            - all predicted probabilities for the known positives are exactly 0.0

#    """
#    nunlabeled = reduce(lambda accum, label: accum + (label == None), labels, 0)
#    if nunlabeled == 0: raise ValueError('Cannot compute beta without unlabeled instances.')

#    # estimate c = p(s == 1 | y == 1)
#    # where c is the probability of a positive instance to be labeled
#    pos_proba = map(op.itemgetter(1),
#                    filter(op.itemgetter(0),
#                           zip(labels, probabilities)))
#    nknownpos = len(pos_proba)
#    if nknownpos == 0: raise ValueError('Cannot compute beta without known positives.')

#    c = sum(pos_proba) / nknownpos
#    if c == 0.0: raise ValueError('Predicted probabilities for all known positives are 0.0.')

#    # beta = p(y == 1 | s == 0)
#    plabeledpos = float(nknownpos) / len(labels)
#    beta1 = (1.0 - c) / c * plabeledpos / (1.0 - plabeledpos)

#    # both appear equal in experiments
#    e = float(nknownpos) / c
#    nlatentpos = min(nunlabeled, e - nknownpos)
#    beta2 = float(nlatentpos) / nunlabeled

#    return beta2
#    return beta1, beta2, c


#import random
#npos = 1000
#nunl = 10000

#labels = [True] * npos + [None] * nunl
#probabilities = [0.1 + 0.9 * random.random() for _ in range(npos)] + [random.random() for _ in range(nunl)]
#beta1, beta2, c = estimate_beta(labels, probabilities)

#print('%1.4f\t%1.4f\f%1.4f' % (beta1, beta2, c))

