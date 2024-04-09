from sklearn.pipeline import Pipeline
from prediciton_model.config import config
from prediciton_model.processing import preprocessing as pp
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import numpy as np

classification_pipeline = Pipeline(
    [
        ('MeanImputation',pp.MeanImputer(variables = config.NUM_FEATURES))
        ,('ModeImputation',pp.ModeImputer(variables=config.CAT_FEATURES))
        ,('DomainProcessing',pp.DomainProcessing(variable_to_modify=config.FEATURE_TO_MODIFY, reference_variable=config.FEATURE_TO_ADD))
        ,('DropFeatures',pp.DropColumns(variables = config.DROP_FEATURES))
        ,('LabelEncoder',pp.CustomLabelEncoder(variables=config.FEATURES_TO_ENCODE))
        ,('LogTransform',pp.LogTransforms(variables=config.LOG_FEATURES))
        ,('MinMaxScale',MinMaxScaler())
        ,('LogisticClassifier',LogisticRegression(random_state=0))
    ]
)