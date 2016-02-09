import random
import pickle

def generate_decvals(num, mean, sigma):
    return [random.normalvariate(mean, sigma) for _ in range(num)]

def load_dist_config(distid):
    """Creates a configuration with differently distributed
    decision values for positives and negatives, which will
    result in differently shaped performance curves.

    distid = 1: high initial precision (near-horizontal ROC curve)
    distid = 2: low initial precision (near-vertical ROC curve)
    distid = 3: standard performance, (circle-segment-like ROC curve)
    """
    if distid == 1:
        dist_config = {'mean_pos': 1.0, 'sigma_pos': 0.3,
                       'mean_neg': 0.0, 'sigma_neg': 1.0}
    elif distid == 2:
        dist_config = {'mean_pos': 2.0, 'sigma_pos': 2.0,
                       'mean_neg': 0.0, 'sigma_neg': 1.0}
    else:
        dist_config = {'mean_pos': 1.0, 'sigma_pos': 1.0,
                       'mean_neg': 0.0, 'sigma_neg': 1.0}
    return dist_config

def simulate_data(distid, num_pos, num_neg, known_pos_frac, known_neg_frac):
    dist_config = load_dist_config(distid)
    generate_pos_class_decvals = lambda num: generate_decvals(num, dist_config['mean_pos'], dist_config['sigma_pos'])
    generate_neg_class_decvals = lambda num: generate_decvals(num, dist_config['mean_neg'], dist_config['sigma_neg'])

    decision_values = generate_pos_class_decvals(num_pos) + generate_neg_class_decvals(num_neg)
    labels = [True] * num_pos + [False] * num_neg
    true_labels, labels, beta = induce_partial_labels(labels,
                                                      known_pos_frac,
                                                      known_neg_frac)
    return labels, true_labels, decision_values, beta


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

def load_dataset(choice, known_pos_frac, known_neg_frac):
    with open('data.pkl', 'rb') as f:
        data = pickle.load(f)
        labels = list(data[choice]['labels'])
        decision_values = list(data[choice]['decision_values'])

    true_labels, labels, beta = induce_partial_labels(labels,
                                                      known_pos_frac,
                                                      known_neg_frac)
    return labels, true_labels, decision_values, beta
