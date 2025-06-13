import numpy as np

from src import DNNManager, create_default_chip
from examples.demo_models import create_vgg_cnn, create_resnet_cnn


def generate_weight_data(dnn_config):
    weight_data = {}
    idx = 0
    for layer in dnn_config.layers:
        if layer.weights_shape is not None:
            weight_data[f"layer_{idx}"] = np.random.randn(*layer.weights_shape)
            idx += 1
    return weight_data


def test_vgg_model_mapping():
    dnn_config = create_vgg_cnn()
    dnn_config.precision = 1
    chip = create_default_chip(crossbar_size=(128, 128), num_crossbars=128)
    manager = DNNManager(dnn_config, chip)

    weight_data = generate_weight_data(dnn_config)
    manager.map_dnn_to_hardware(weight_data)
    validation = manager.validate_hardware_capacity()
    assert validation["overall_sufficient"]


def test_resnet_model_mapping():
    dnn_config = create_resnet_cnn()
    dnn_config.precision = 1
    chip = create_default_chip(crossbar_size=(128, 128), num_crossbars=32)
    manager = DNNManager(dnn_config, chip)

    weight_data = generate_weight_data(dnn_config)
    manager.map_dnn_to_hardware(weight_data)
    validation = manager.validate_hardware_capacity()
    assert validation["overall_sufficient"]
