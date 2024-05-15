from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

from xgboost.sklearn import XGBClassifier
from lightgbm import LGBMClassifier

from Toolbox import load_data, per_processing, result_visualize_classification


def stacking(x=None, n_class=None, n_per_class=None, is_z=None):
    X_train, X_test, y_train, y_test = x

    # Define base learner
    rf_clf = RandomForestClassifier(max_depth=73, max_features=54, max_leaf_nodes=100, min_samples_leaf=1,
                                    min_samples_split=117, n_estimators=1020, verbose=False, n_jobs=-1,
                                    class_weight='balanced', oob_score=True)
    xgb_clf = XGBClassifier(colsample_bytree=0.95, learning_rate=0.01297, max_delta_step=1, max_depth=65,
                            min_child_weight=5, n_estimators=908, reg_alpha=0.69282, reg_lambda=0.92703,
                            subsample=0.26495)
    lgbm_clf = LGBMClassifier(colsample_bytree=0.94, learning_rate=0.1, max_depth=30, n_estimators=82, subsample=0.4,
                              num_leaves=60, min_child_samples=21)

    estimators = [("rf", rf_clf),
                  ("xgb", xgb_clf),
                  ("lgbm", lgbm_clf)]

    # Define meta learner
    # meta_clf = LogisticRegression(C=300, class_weight='balanced', max_iter=5000)
    meta_clf = LogisticRegression(max_iter=5000)

    # Define stacking learner
    stacking_clf = StackingClassifier(estimators=estimators, final_estimator=meta_clf,
                                      cv=5, n_jobs=-1)

    stacking_clf.fit(X_train, y_train)

    # Write output log
    if is_z:
        note = f'Estimated-Redshift Stacking {n_class}-class '
    else:
        note = f'Non-Redshift Stacking {n_class}-class '

    if n_per_class:
        note += f'Balanced({n_per_class})'

    result_visualize_classification(X_test, y_test, stacking_clf, label_encoder,
                                    save_dir='result/stacking', is_stacking=True,
                                    note=note)


if __name__ == '__main__':
    data = load_data('Datasets/SDSS_DR17_label_z_photo_new.csv')
    n_per_class = 10000

    for n_class in [3, 4, 7]:
        useless_features = ['objid', 'mjd', 'plate', 'fiberID', 'zWarning', 'class', 'subclass', 'redshift',
                            'dered_r', 'dered_g', 'dered_i', 'dered_u', 'dered_z',
                            'extinction_r', 'extinction_g', 'extinction_i', 'extinction_u', 'extinction_z']
        # useless_features = ['objid', 'mjd', 'plate', 'fiberID', 'zWarning', 'class', 'subclass', 'redshift']
        labels = ['label_3', 'label_4', 'label_7']
        labels.remove(f'label_{n_class}')
        useless_features.extend(labels)
        data = load_data('Datasets/SDSS_DR17_label_z_photo.csv')
        X_orig, y, label_encoder = per_processing(x=data, rm_features=useless_features, n_class=n_class,
                                                  n_per_class=n_per_class, is_z=True)
        for is_z in [True, False]:
            if is_z:
                X = X_orig
            else:
                X = X_orig.drop('z_photo', axis=1)
            x = train_test_split(X, y, train_size=0.8, random_state=28)

            stacking(x=x, n_class=n_class, is_z=is_z, n_per_class=n_per_class)
