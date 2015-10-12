import semisup_metrics as ss
import resvm
import optunity
import optunity.metrics
import optunity.cross_validation as cv
import numpy as np
import sklearn.linear_model


def build_resvm_classifier(train_data, train_labels, validation_data, nmodels=200):
    """
    blah
    """
    @optunity.cross_validated(x=train_data, y=train_labels, num_folds=5,
                              strata=optunity.cross_validation.strata_by_labels(train_labels))
    def resvm_auroc(x_train, y_train, x_test, y_test, logC, logwpos, npos, nunl):
        model = resvm.RESVM(nmodels=nmodels, C=10 ** logC, wpos=10 ** logwpos,
                            nunl=nunl, npos=npos)
        model.fit(x_train, y_train)
        probas = model.predict_proba(x_test)
        auc = optunity.metrics.roc_auc(y_test, probas)
        print('C=%1.3f, wpos=%1.3f, npos=%d, nunl=%d\t-->\t%1.3f' % (10 ** logC, 10 ** logwpos, int(nunl), int(npos), auc))
        return auc

    print('maximizing')
    pars, _, _ = optunity.maximize(resvm_auroc, num_evals=150,
                                   logC=[0, 4], logwpos=[-1, 2],
                                   npos=[10, 200], nunl=[10, 500])
    pars['nmodels'] = nmodels
    pars['C'] = 10 ** pars['logC']
    pars['wpos'] = 10 ** pars['logwpos']
    del pars['logC']
    del pars['logwpos']
    model = resvm.RESVM(**pars)
    model.fit(train_data, train_labels)

    probabilities = model.predict_proba(validation_data)
    return probabilities, model

def estimate_beta_with_resvm(train_data, train_labels,
                             validation_data, validation_labels):
    val_probas, model = build_resvm_classifier(train_data, train_labels,
                                               validation_data)
    beta = ss.estimate_beta(validation_labels, val_probas)
    return beta, model

lr_pos_idx = lambda model: 0 if model.classes_[0] else 1


def build_tuned_lr(data, labels):

    @optunity.cross_validated(x=data, y=labels, num_folds=10,
                              strata=cv.strata_by_labels(labels))
    def lr_auroc(x_train, y_train, x_test, y_test, logC, logwpos):
        model = sklearn.linear_model.LogisticRegression(C=10 ** logC, class_weight={True: 10 ** logwpos})
        model.fit(x_train, y_train)
        probas = model.predict_proba(x_test)
        auc = optunity.metrics.roc_auc(y_test, probas[:, lr_pos_idx(model)])
        #print('C=%1.4f, wpos=%1.4f\t-->\t%1.4f' % (10 ** logC, 10 ** logwpos, auc))
        return auc

    pars, _, _ = optunity.maximize(lr_auroc, num_evals=100,
                                   logC=[0, 4], logwpos = [0, 3],
                                   pmap=optunity.pmap)
    model = sklearn.linear_model.LogisticRegression(C=10 ** pars['logC'],
                                                    class_weight={True: 10 ** pars['logwpos']})
    model.fit(data, labels)
    return model

def estimate_beta_with_lr(train_data, train_labels,
                          validation_data, validation_labels):
    model = build_tuned_lr(train_data, train_labels)
    val_probas = model.predict_proba(validation_data)
    beta = ss.estimate_beta(validation_labels, val_probas[:, lr_pos_idx(model)])
    return beta, model



nposknown = 500
nposlatent = 500
nneglatent = 5000

pos = np.random.rand(nposknown + nposlatent, 2) + np.array((0.5, 0.0))
neg = np.random.rand(nneglatent, 2)
knownpos = pos[:nposknown, :]
unlabeled = np.vstack((pos[nposknown:, :], neg))
data = np.vstack((knownpos, unlabeled))
labels = [True] * nposknown + [None] * (nposlatent + nneglatent)

nposknown_val = 500
nposlatent_val = 500
nneglatent_val = 5000

pos_val = np.random.rand(nposknown_val+ nposlatent_val, 2) + np.array((0.5, 0.0))
neg_val = np.random.rand(nneglatent_val, 2)
knownpos_val = pos[:nposknown_val, :]
unlabeled_val = np.vstack((pos_val[nposknown_val:, :], neg_val))
data_val = np.vstack((knownpos_val, unlabeled_val))
labels_val = [True] * nposknown_val + [None] * (nposlatent_val + nneglatent_val)

labels = np.array(labels)
labels_val = np.array(labels_val)

beta = ss.estimate_beta(


#beta, model = estimate_beta_with_resvm(data, labels, data_val, labels_val)
beta, model = estimate_beta_with_lr(data, labels, data_val, labels_val)
print('%1.4f\t%1.4f' % (float(nposlatent) / (nneglatent + nposlatent), beta))

nboot = 100
betas = [estimate_beta_with_lr(data, labels, data_val, labels_val)[0]
         for _ in xrange(nboot)]

betas = sorted(betas)
print('%1.3f\t%1.3f\t%1.3f' % (betas[1], betas[49], betas[98]))


#val_probas, model = build_resvm_classifier(data, labels, data_val)
