import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from xgboost.sklearn import XGBClassifier
from lightgbm import LGBMClassifier

from Toolbox import load_data, per_processing, result_visualize_classification


def poc(x=None, clf=None, n_class=None, n_per_class=None, is_z=None):
    name, classifier = clf
    X_train, X_test, y_train, y_test = x
    classifier.fit(X_train, y_train)

    tmp = {
        'feature': X_train.columns,
        'importance': classifier.feature_importances_
    }

    if is_z:
        note = f'Estimated-Redshift {name} {n_class}-class '
    else:
        note = f'Non-Redshift {name} {n_class}-class '

    if n_per_class:
        note += f'Balanced({n_per_class})'

    result_visualize_classification(X_test, y_test, classifier, label_encoder,
                                    save_dir='result/POC',
                                    note=note)
    sorted_tmp = sorted(zip(tmp['feature'], tmp['importance']), key=lambda x: x[1], reverse=True)
    return sorted_tmp


if __name__ == '__main__':
    n_per_class = 10000
    feature_importances = []

    for n_class in [3, 4, 7]:
        useless_features = ['objid', 'mjd', 'plate', 'fiberID', 'zWarning', 'class', 'subclass', 'redshift',
                            'dered_r', 'dered_g', 'dered_i', 'dered_u', 'dered_z',
                            'extinction_r', 'extinction_g', 'extinction_i', 'extinction_u', 'extinction_z']
        # useless_features = ['objid', 'mjd', 'plate', 'fiberID', 'zWarning', 'class', 'subclass', 'redshift']
        labels = ['label_3', 'label_4', 'label_7']
        labels.remove(f'label_{n_class}')
        useless_features.extend(labels)
        data = load_data('Datasets/SDSS_DR17_label_z_photo_new.csv')
        X_orig, y, label_encoder = per_processing(x=data, rm_features=useless_features, n_class=n_class,
                                                  n_per_class=n_per_class, is_z=True)

        for is_z in [True, False]:
            if is_z:
                X = X_orig
            else:
                X = X_orig.drop('z_photo', axis=1)

            print(f"Is z_photo in X.columns: {'z_photo' in X.columns}")
            x = train_test_split(X, y, train_size=0.8, random_state=28)

            # Define base learner
            rf_clf = RandomForestClassifier(max_depth=73, max_features=54, max_leaf_nodes=100, min_samples_leaf=1,
                                            min_samples_split=117, n_estimators=1020, verbose=False, n_jobs=-1,
                                            class_weight='balanced', oob_score=True)
            xgb_clf = XGBClassifier(colsample_bytree=0.95, learning_rate=0.01297, max_delta_step=1, max_depth=65,
                                    min_child_weight=5, n_estimators=908, reg_alpha=0.69282, reg_lambda=0.92703,
                                    subsample=0.26495)
            lgbm_clf = LGBMClassifier(colsample_bytree=0.94, learning_rate=0.1, max_depth=30, n_estimators=82,
                                      subsample=0.4,
                                      num_leaves=60, min_child_samples=21)

            clfs = [['RF', rf_clf], ['XGB', xgb_clf], ['LGBM', lgbm_clf]]

            for clf in clfs:
                print(f"Running {clf[0]} when is_z={is_z} and n_class={n_class}:")
                feat_im = poc(x=x, clf=clf, is_z=is_z, n_class=n_class, n_per_class=n_per_class)
                feature_importances.append(feat_im)

    is_z = ['+', '-']
    with pd.ExcelWriter('result/POC/feature_importance.xlsx') as writer:
        for i in range(18):
            df = pd.DataFrame(feature_importances[i])
            df.to_excel(writer, sheet_name=f'sheet_{i}', index=False)
