import numpy as np

from src.core.crossbar import CrossbarArray, CrossbarConfig
from src.core.peripherals import PeripheralManager


def test_mvm_updates_peripheral_counters():
    cfg = CrossbarConfig(rows=4, cols=4)
    xbar = CrossbarArray(cfg)
    pm = PeripheralManager(num_rows=cfg.rows, num_cols=cfg.cols)
    weights = np.eye(cfg.rows, cfg.cols)
    xbar.program_weights(weights)
    inp = np.arange(cfg.rows)
    xbar.matrix_vector_multiply(inp, peripheral_manager=pm)
    assert sum(d.conversion_count for d in pm.input_dacs) == cfg.rows
    assert sum(a.conversion_count for a in pm.output_adcs) == cfg.cols
