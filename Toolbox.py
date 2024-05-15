import re
import os
import functools
import itertools
import textwrap
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

import joblib
import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


def load_data(path: str) -> pd.DataFrame:
    x = pd.read_csv(path)
    return x


def features_remove(features: list, rm: list) -> list:
    for i in rm:
        features.remove(i)
    return features


def dataset_rebalanced(x: pd.DataFrame, target_column: str, n: int):
    tmp = x.groupby(target_column, group_keys=False).apply(lambda p: p.sample(min(len(p), n)))
    return tmp


class PhotometricIterator:
    """
    Create an iterator for various photometric feature (modelMag, cmodelMag, psfMag, w*mpro).
    """

    def __init__(self, columns, patterns: list = None, is_w: bool = None):
        if patterns is None:
            patterns = ['modelMag_.$', 'cmodelMag_.$',
                        'psfMag_.$', 'w.$', 'dered_.$' , 'extinction_.$']
            # 'dered_.$' , 'extinction_.$' exist negative value, so temporarily deprecated.
        if not is_w or any(re.compile('w.$').match(i) for i in columns) is False:
            patterns.remove('w.$')
        self.patterns = patterns
        self.columns = columns
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.patterns):
            filtered_columns = [i for i in self.columns if re.match(self.patterns[self.index], i)]
            self.index += 1
            return filtered_columns
        else:
            raise StopIteration


def reconstruct_feature_space(x: pd.DataFrame, features: list, is_w: bool = None) -> pd.DataFrame:
    """
    Reconstruct the feature space of original data by adding color - color.
    :param features: The list of features
    :param x: The original data.
    :param is_w: Whether to use infrared data
    :return: The data after reconstructing the feature space.
    """
    # Add the differences between each pair of bands in various photometric feature as new feature
    for photometric_columns in PhotometricIterator(x.columns, is_w):
        pairs = itertools.combinations(photometric_columns, 2)
        for pair in pairs:
            tmp = x.loc[:, pair[0]] - x.loc[:, pair[1]]
            x.insert(1, f'{pair[0]}-{pair[1]}', tmp)
            features.append(f'{pair[0]}-{pair[1]}')
    return x


def preprocessing_numerical_pipeline():
    """
    The pre-processing of numerical data - standard scaler
    :return: The pipeline for numerical data pre-processing
    """
    scaler_numerical = StandardScaler()
    numerical_transformer = Pipeline(steps=[('scale', scaler_numerical)])
    return numerical_transformer


def preprocessing_categorical_pipeline():
    """
    The pre-processing of categorical data - missing value processing (mode) and One-hot encoding
    :return: The pipeline for categorical data pre-processing
    """
    impute_categorical = SimpleImputer(strategy="most_frequent")
    onehot_categorical = OneHotEncoder(handle_unknown='ignore')
    categorical_transformer = Pipeline(steps=[('impute', impute_categorical), ('onehot', onehot_categorical)])
    return categorical_transformer


def preprocessing_pipeline(x: pd.DataFrame):
    """
    Create a pre-processing pipeline that includes Standardization of numerical columns, imputation (mode) and One-Hot
    encoding for categorical columns.

    :param x: The target data
    :return:
    """
    # Numerical columns
    numerical_columns = x.select_dtypes(include=[float, int]).columns
    numerical_transformer = preprocessing_numerical_pipeline()

    # Categorical columns
    categorical_columns = x.select_dtypes(exclude=[float, int]).columns
    categorical_transformer = preprocessing_categorical_pipeline()

    # Pre-processing for all columns
    preprocessor_all_columns = ColumnTransformer(
        transformers=[('cat', categorical_transformer, categorical_columns),
                      ('num', numerical_transformer, numerical_columns)],
        remainder="passthrough")

    return preprocessor_all_columns


def per_processing(x: pd.DataFrame, rm_features: list, n_class: int, n_per_class: int = None, is_z: bool = None,
                   is_w: bool = None) -> tuple:
    features = x.columns.to_list()
    if not is_z:
        rm_features.append('z_photo')
    features_remove(features, rm_features)

    if not n_class or not (n_class in [3, 4, 7]):
        raise ValueError('n_class cannot be None, and must be one of [3, 4, 7]')

    label_column = f'label_{n_class}'
    if n_per_class:
        x = dataset_rebalanced(x, label_column, n_per_class)
    x = reconstruct_feature_space(x[features], features, is_w)  # Add new feature color - color
    features.remove(label_column)

    preprocessor = preprocessing_pipeline(x[features])

    transformed_x = preprocessor.fit_transform(x[features])
    transformed_x = pd.DataFrame(transformed_x, columns=features)

    label_encoder = LabelEncoder()
    label = label_encoder.fit_transform(x[f'label_{n_class}'])

    return transformed_x, label, label_encoder


def decorator_plot_save(func):
    """
    A decorator function for saving plot, which allows the figure to be saved to `save_dir` if provided or shown if not.

    save_dir: Optional. A string specifying the folder path where the plots will be saved.
                        If not provided, the plots will be shown instead of saved.
    :return: A decorator function that can be applied to other functions.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        folder_path = kwargs.get('save_dir')
        try:
            kwargs.pop('save_dir')
        finally:
            fig_name = func(*args, **kwargs)
        if folder_path:
            # Make the directory if 'folder_path' don't exist
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            plt.savefig(os.path.join(folder_path, fig_name),
                        bbox_inches='tight', dpi=800, pad_inches=0.0)
            plt.close()
        else:
            plt.show()

    return wrapper


def decorator_report_save(func):
    """
    Format and store the result report.
    :param func:
    :return:
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        current_time = datetime.now()
        kwargs['current_time'] = current_time

        report_str = '=' * 28 + "Time: {}\nNote:\n".format(current_time.strftime("%Y-%m-%d %H:%M:%S"))
        if kwargs.get('note'):
            for note_str in textwrap.wrap(kwargs['note'], len(report_str) - 4):
                report_str += '    ' + note_str + '\n'

        try:
            kwargs.pop('note')
        finally:
            report_content = func(*args, **kwargs)

        report_str += report_content
        print(report_str)

        if kwargs.get('save_dir'):
            report_file = os.path.join(kwargs['save_dir'], 'classification_report.txt')
            if os.path.exists(report_file):
                with open(report_file, 'a') as f:
                    f.write("\n\n" + report_str)
            else:
                with open(report_file, 'w') as f:
                    f.write(report_str)

    return wrapper


def dict_format_str(x: dict, width: int, indent_num: int = 0) -> str:
    """
    Format and convert dict data into a string.
    :param x: The target dict data
    :param width: Maximum line width
    :param indent_num: Line indentation width
    :return: Formatted string-formatted data
    """
    content = ''
    for key, value in x.items():
        if value is None:
            value = "N/A"
        content += "    " * indent_num + f"{key}: {value:>{width - len(key)}}\n"
    return content


@decorator_plot_save
def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, current_time: datetime, is_norm: bool = True) -> str:
    """
    Plot confusion matrix.
    :param is_norm: Control whether to normalize
    :param y_true: True data
    :param y_pred: Prediction data
    :param current_time: Current time used for the figure naming finally
    :return: The name of figure
    """
    if is_norm:
        skplt.metrics.plot_confusion_matrix(y_true, y_pred, text_fontsize="small", cmap='YlOrBr',
                                            figsize=(6, 4), normalize='true')
    else:
        skplt.metrics.plot_confusion_matrix(y_true, y_pred, text_fontsize="small", cmap='YlOrBr',
                                            figsize=(6, 4))
    return f'confusion_matrix_{current_time.strftime("%Y%m%d%H%M%S")}.png'


@decorator_plot_save
def roc(y_true: np.ndarray, y_probas: np.ndarray, current_time: datetime) -> str:
    """
    Plot ROC curves
    :param y_true: True data
    :param y_probas: Prediction probability
    :param current_time: Current time used for the figure naming finally
    :return: The name of figure
    """
    skplt.metrics.plot_roc(y_true, y_probas)
    return f'roc_curve_{current_time.strftime("%Y%m%d%H%M%S")}.png'


@decorator_plot_save
def precision_recall(y_true: np.ndarray, y_probas: np.ndarray, current_time: datetime) -> str:
    """
    Plot precision-recall curve
    :param y_true: True data
    :param y_probas: Prediction probability
    :param current_time: Current time used for the figure naming finally
    :return: The name of figure
    """
    skplt.metrics.plot_precision_recall(y_true, y_probas)
    return f'precision_recall_curve_{current_time.strftime("%Y%m%d%H%M%S")}.png'


@decorator_plot_save
def feature_importance(classifier, feature_names: list, current_time: datetime, rot=0) -> str:
    """
    Plot feature importance
    :param classifier: Trained classifier
    :param feature_names: The name of each feature
    :param current_time: Current time used for the figure naming finally
    :param rot: The Angle of rotation of the label
    :return: The name of figure
    """
    skplt.estimators.plot_feature_importances(
        classifier, figsize=(12, 12), feature_names=feature_names, x_tick_rotation=rot, text_fontsize='small')
    fig_name = f'feature_importance_{current_time.strftime("%Y%m%d%H%M%S")}.png'
    return fig_name


@decorator_report_save
def result_visualize_classification(x: pd.DataFrame, y_true: np.ndarray, classifier, encoder,
                                    save_dir: str = None, current_time: datetime = None,
                                    is_stacking: bool = None) -> str:
    """
    Classification result visualization, including confusion matrix, ROC curve, precision-recall curve,
    feature importance, and result reporting.
    :param is_stacking: Whether it is stacking
    :param x: Feature data
    :param y_true: Ture target data
    :param classifier: Train classifier
    :param encoder: Encoder used for data encoding in experiments
    :param save_dir: The root directory where the results are saved
    :param current_time: The time when the function is called
    :return: Content of the results report
    """
    # Prediction
    y_pred = classifier.predict(x)
    y_probas = classifier.predict_proba(x)

    # Inverse transformation encoding for true and prediction data
    y_pred = encoder.inverse_transform(y_pred)
    y_true = encoder.inverse_transform(y_true)

    # Plot evaluation metric
    save_dir_pack = os.path.join(save_dir, current_time.strftime("%Y%m%d%H%M%S"))
    if not save_dir_pack:
        os.mkdir(save_dir_pack)

    confusion_matrix(y_true, y_pred, current_time, save_dir=save_dir_pack)
    roc(y_true, y_probas, current_time, save_dir=save_dir_pack)
    precision_recall(y_true, y_probas, current_time, save_dir=save_dir_pack)
    if not is_stacking:
        feature_importance(classifier, x.columns, current_time, rot=60, save_dir=save_dir_pack)

    # Save model
    name_model = os.path.join(save_dir_pack, f'classifier_{current_time.strftime("%Y%m%d%H%M%S")}')
    try:
        classifier.save_model(name_model + '.json')
    except AttributeError:
        joblib.dump(classifier, name_model + '.pkl')

    # THe content of classification report about classifier parameters
    report_content = ("\nClassification Report:\n\n"
                      + classification_report(y_true, y_pred, digits=3, output_dict=False))
    # if is_stacking:
    #     return report_content
    # else:
    #     return ("\nClassifier Parameters:\n"
    #             + dict_format_str(classifier.get_params(), 44, 1)
    #             + report_content)

    return report_content