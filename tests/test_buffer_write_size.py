import numpy as np
from src.core.memory_system import BufferManager, MemoryConfig, MemoryType


def test_write_request_size_matches_data_length():
    cfg = {'buf': MemoryConfig(memory_type=MemoryType.SRAM, size_kb=4, banks=1, word_size_bits=32)}
    bm = BufferManager(cfg)
    region_id = bm.allocate_buffer('buf', size_words=16)
    assert region_id is not None
    data = [1, 2, 3, 4]
    req_id = bm.write_data('buf', region_id, 0, data)
    assert req_id is not None
    controller = bm.controllers['buf']
    assert controller.request_queue, "No request queued"
    queued_request = controller.request_queue[0][2]
    expected_bits = len(data) * cfg['buf'].word_size_bits
    assert queued_request.size_bits == expected_bits


def test_np_array_write_records_expected_size():
    cfg = {
        'buf': MemoryConfig(memory_type=MemoryType.SRAM, size_kb=4, banks=1, word_size_bits=16)
    }
    bm = BufferManager(cfg)
    region_id = bm.allocate_buffer('buf', size_words=32)
    assert region_id is not None
    data = np.arange(8, dtype=np.int16)
    req_id = bm.write_data('buf', region_id, 0, data)
    assert req_id is not None
    controller = bm.controllers['buf']
    queued_request = controller.request_queue[0][2]
    expected_bits = len(data) * cfg['buf'].word_size_bits
    assert queued_request.size_bits == expected_bits
