import numpy as np

from src import (
    create_default_chip,
    DNNManager,
    ExecutionEngine,
    ExecutionConfig,
    LayerConfig,
    LayerType,
    DNNConfig,
)


def build_tiny_cnn():
    layers = [
        LayerConfig(
            layer_type=LayerType.CONV2D,
            input_shape=(8, 8, 1),
            output_shape=(6, 6, 4),
            kernel_size=(3, 3),
            activation="relu",
            weights_shape=(3, 3, 1, 4),
        ),
        LayerConfig(
            layer_type=LayerType.POOLING,
            input_shape=(6, 6, 4),
            output_shape=(3, 3, 4),
            kernel_size=(2, 2),
        ),
        LayerConfig(
            layer_type=LayerType.DENSE,
            input_shape=(36,),
            output_shape=(3,),
            activation=None,
            weights_shape=(36, 3),
        ),
    ]

    return DNNConfig(
        model_name="TinyCNN",
        layers=layers,
        input_shape=(8, 8, 1),
        output_shape=(3,),
        precision=8,
    )


def test_cycle_accurate_mcu():
    chip = create_default_chip(crossbar_size=(16, 16), num_crossbars=16)
    dnn_config = build_tiny_cnn()
    dnn_manager = DNNManager(dnn_config, chip)
    engine = ExecutionEngine(
        chip,
        dnn_manager,
        ExecutionConfig(enable_cycle_accurate_simulation=True),
    )
    input_data = np.random.randn(*dnn_config.input_shape)
    result = engine.execute_inference(input_data)
    ipc = result["system_statistics"]["microcontroller_statistics"][
        "instructions_per_cycle"
    ]
    assert ipc > 0
