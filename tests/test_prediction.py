import pytest
from prediciton_model.config import config
from prediciton_model.processing.data_handling import load_dataset
from prediciton_model.predict import generate_predictions

# output from predict script is not null
# output from predict script is sting
# output is Y for an example data

#Fixtures --> functions before test function --> ensure single_prediction

@pytest.fixture
def single_prediction():
    test_dataset = load_dataset(config.TEST_FILE)
    single_row = test_dataset[:1]
    result = generate_predictions(single_row)
    return result

def test_single_pred_not_none(single_prediction):
    assert single_prediction is not None

def test_single_pred_str_type(single_prediction):
    assert isinstance(single_prediction.get('Predictions')[0],str)

def test_single_pred_validate(single_prediction):
    assert single_prediction.get('Predictions')[0]=='Y'