import numpy as np
from src.core.microcontroller import Microcontroller, MCUConfig, Instruction, InstructionType
from src.core.memory_system import BufferManager, MemoryConfig, MemoryType
from src.core.compute_units import ComputeUnitManager, ComputeUnitConfig, ActivationType, PoolingType


def create_mcu():
    mcu = Microcontroller(MCUConfig())
    buffer_cfg = {
        'buf': MemoryConfig(memory_type=MemoryType.SRAM, size_kb=4, read_latency=1, write_latency=1, banks=1)
    }
    bm = BufferManager(buffer_cfg)
    cu = ComputeUnitManager()
    cu.add_shift_add_unit(ComputeUnitConfig(unit_type='shift', latency_cycles=1))
    cu.add_activation_unit(ComputeUnitConfig(unit_type='act', latency_cycles=1), [ActivationType.RELU])
    cu.add_pooling_unit(ComputeUnitConfig(unit_type='pool', latency_cycles=1))
    mcu.set_hardware_resources(bm, cu)
    return mcu, bm, cu


def test_microcontroller_operations():
    mcu, bm, cu = create_mcu()
    insts = []
    insts.append(Instruction(0, InstructionType.LOAD_INPUT, {'source_buffer': 'buf', 'input_shape': (4,)}, []))
    insts.append(Instruction(1, InstructionType.SHIFT_ADD, {'bit_precision': 4}, [0]))
    insts.append(Instruction(2, InstructionType.ACTIVATION, {'activation_type': 'relu'}, [1]))
    insts.append(Instruction(3, InstructionType.POOLING, {'pooling_type': 'max', 'kernel_size': (2, 2)}, [2]))
    insts.append(Instruction(4, InstructionType.STORE_OUTPUT, {'target_buffer': 'buf', 'output_shape': (2,2)}, [3]))

    mcu.load_program(insts)
    result = mcu.run_until_completion(max_cycles=100)
    assert result['instructions_executed'] == 5
    stats = bm.controllers['buf'].get_statistics()
    assert stats['total_requests'] > 0
    assert cu.shift_add_units[0].operation_count == 1
    assert cu.activation_units[0].operation_count == 1
    assert cu.pooling_units[0].operation_count == 1
    assert mcu.energy_consumption > 0
