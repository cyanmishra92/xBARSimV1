import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import heapq
import logging
from collections import deque, defaultdict
from .memory_system import BufferManager, MemoryConfig, MemoryType
from .compute_units import ComputeUnitManager, ActivationType, PoolingType, NormalizationType

class InstructionType(Enum):
    NOP = "nop"
    LOAD_WEIGHTS = "load_weights"
    LOAD_INPUT = "load_input"
    XBAR_COMPUTE = "xbar_compute"
    SHIFT_ADD = "shift_add"
    ACTIVATION = "activation"
    POOLING = "pooling"
    NORMALIZATION = "normalization"
    STORE_OUTPUT = "store_output"
    SYNCHRONIZE = "synchronize"

class ExecutionState(Enum):
    IDLE = "idle"
    FETCHING = "fetching"
    DECODING = "decoding"
    EXECUTING = "executing"
    WAITING = "waiting"
    COMPLETED = "completed"

@dataclass
class Instruction:
    """Microcontroller instruction"""
    inst_id: int
    inst_type: InstructionType
    operands: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[int] = field(default_factory=list)  # Instruction IDs this depends on
    priority: int = 0
    estimated_cycles: int = 1
    issue_cycle: int = -1
    completion_cycle: int = -1
    state: ExecutionState = ExecutionState.IDLE

@dataclass
class MCUConfig:
    """Microcontroller configuration"""
    clock_frequency_mhz: float = 1000.0  # 1 GHz
    instruction_fetch_latency: int = 1
    instruction_decode_latency: int = 1
    pipeline_stages: int = 5
    max_outstanding_instructions: int = 16
    branch_prediction_accuracy: float = 0.95
    cache_size_kb: int = 32
    cache_associativity: int = 4

class TaskScheduler:
    """Task scheduler for the microcontroller"""
    def __init__(self):
        self.ready_queue = []  # Priority queue for ready instructions
        self.waiting_queue = {}  # inst_id -> instruction (waiting for dependencies)
        self.executing_queue = {}  # inst_id -> (instruction, completion_cycle)
        self.completed_instructions = {}  # inst_id -> instruction
        self.dependency_graph = defaultdict(set)  # inst_id -> set of dependent inst_ids
        self.tie_breaker = 0  # Counter to avoid comparing Instruction objects
        
    def add_instruction(self, instruction: Instruction):
        """Add instruction to scheduler"""
        if instruction.dependencies:
            # Has dependencies - add to waiting queue
            self.waiting_queue[instruction.inst_id] = instruction
            instruction.state = ExecutionState.WAITING
        else:
            # No dependencies - add to ready queue
            # Use tie_breaker to avoid comparing Instruction objects
            heapq.heappush(self.ready_queue, (-instruction.priority, self.tie_breaker, instruction.inst_id, instruction))
            self.tie_breaker += 1
            instruction.state = ExecutionState.IDLE
            
        # Update dependency graph
        for dep_id in instruction.dependencies:
            self.dependency_graph[dep_id].add(instruction.inst_id)
            
    def get_ready_instruction(self) -> Optional[Instruction]:
        """Get next ready instruction from queue"""
        if self.ready_queue:
            _, _, _, instruction = heapq.heappop(self.ready_queue)
            return instruction
        return None
        
    def mark_instruction_complete(self, inst_id: int, completion_cycle: int):
        """Mark instruction as complete and wake up dependent instructions"""
        if inst_id in self.executing_queue:
            instruction, _ = self.executing_queue[inst_id]
            instruction.completion_cycle = completion_cycle
            instruction.state = ExecutionState.COMPLETED
            self.completed_instructions[inst_id] = instruction
            del self.executing_queue[inst_id]
            
            # Check dependent instructions
            for dependent_id in self.dependency_graph[inst_id]:
                if dependent_id in self.waiting_queue:
                    dependent_inst = self.waiting_queue[dependent_id]
                    dependent_inst.dependencies.remove(inst_id)
                    
                    # If all dependencies satisfied, move to ready queue
                    if not dependent_inst.dependencies:
                        del self.waiting_queue[dependent_id]
                        heapq.heappush(self.ready_queue, 
                                     (-dependent_inst.priority, self.tie_breaker, dependent_id, dependent_inst))
                        self.tie_breaker += 1
                        dependent_inst.state = ExecutionState.IDLE
                        
    def mark_instruction_executing(self, instruction: Instruction, completion_cycle: int):
        """Mark instruction as executing"""
        instruction.state = ExecutionState.EXECUTING
        self.executing_queue[instruction.inst_id] = (instruction, completion_cycle)
        
    def get_statistics(self) -> Dict:
        """Get scheduler statistics"""
        return {
            'ready_queue_size': len(self.ready_queue),
            'waiting_queue_size': len(self.waiting_queue),
            'executing_queue_size': len(self.executing_queue),
            'completed_instructions': len(self.completed_instructions)
        }

class Pipeline:
    """Simplified instruction pipeline"""
    def __init__(self, config: MCUConfig):
        self.config = config
        # Simple 3-stage pipeline: FETCH, EXECUTE, WRITEBACK
        self.stages = [[] for _ in range(3)]
        self.stage_names = ["FETCH", "EXECUTE", "WRITEBACK"]
        self.current_cycle = 0
        self.stall_cycles = 0
        self.completed_instructions = 0
        
    def can_issue(self) -> bool:
        """Check if pipeline can issue new instruction"""
        return len(self.stages[0]) == 0  # Fetch stage must be empty
        
    def issue_instruction(self, instruction: Instruction) -> bool:
        """Issue instruction to pipeline"""
        if self.can_issue():
            instruction.issue_cycle = self.current_cycle
            instruction.state = ExecutionState.FETCHING
            self.stages[0].append(instruction)
            return True
        return False
        
    def tick(self) -> Dict[str, Any]:
        """Process one pipeline cycle - simplified"""
        self.current_cycle += 1
        events = {
            'cycle': self.current_cycle,
            'completed_instructions': [],
            'pipeline_stages': []
        }
        
        # Stage 2: WRITEBACK - Complete instructions
        if self.stages[2]:
            for instruction in self.stages[2]:
                instruction.completion_cycle = self.current_cycle
                instruction.state = ExecutionState.COMPLETED
                events['completed_instructions'].append(instruction.inst_id)
                self.completed_instructions += 1
            self.stages[2] = []
        
        # Stage 1: EXECUTE -> WRITEBACK
        if self.stages[1]:
            # Move from execute to writeback
            self.stages[2] = self.stages[1]
            self.stages[1] = []
            for instruction in self.stages[2]:
                instruction.state = ExecutionState.EXECUTING
        
        # Stage 0: FETCH -> EXECUTE  
        if self.stages[0]:
            # Move from fetch to execute
            self.stages[1] = self.stages[0]
            self.stages[0] = []
            for instruction in self.stages[1]:
                instruction.state = ExecutionState.EXECUTING
                
        return events
        
    def flush(self):
        """Flush pipeline"""
        for stage in self.stages:
            stage.clear()
        
    def get_statistics(self) -> Dict:
        """Get pipeline statistics"""
        total_instructions = sum(len(stage) for stage in self.stages)
        return {
            'current_cycle': self.current_cycle,
            'instructions_in_pipeline': total_instructions,
            'completed_instructions': self.completed_instructions,
            'stall_cycles': self.stall_cycles,
            'pipeline_utilization': total_instructions / len(self.stages)
        }

class Microcontroller:
    """Main microcontroller class"""
    def __init__(self, config: MCUConfig):
        self.config = config
        self.scheduler = TaskScheduler()
        self.pipeline = Pipeline(config)
        self.current_cycle = 0
        self.instruction_cache = {}  # Simple instruction cache
        
        # Hardware resources
        self.buffer_manager = None
        self.compute_manager = None
        
        # Statistics
        self.total_instructions_executed = 0
        self.total_cycles = 0
        self.energy_consumption = 0.0
        
        # Instruction sequence for current program
        self.program_counter = 0
        self.current_program = []
        
    def set_hardware_resources(self, buffer_manager: BufferManager, 
                              compute_manager: ComputeUnitManager):
        """Set references to hardware resources"""
        self.buffer_manager = buffer_manager
        self.compute_manager = compute_manager
        
    def reset(self):
        """Reset microcontroller state"""
        self.current_cycle = 0
        self.program_counter = 0
        self.total_cycles = 0
        
        # Reset scheduler
        self.scheduler = TaskScheduler()
        
        # Reset pipeline
        self.pipeline = Pipeline(self.config)
        
        # Keep cumulative statistics
        # self.total_instructions_executed and self.energy_consumption are preserved
        
    def load_program(self, instructions: List[Instruction]):
        """Load instruction sequence"""
        self.current_program = instructions
        self.program_counter = 0
        
        # Reset scheduler state for new program
        self.scheduler = TaskScheduler()
        
        # Add all instructions to scheduler
        for instruction in instructions:
            self.scheduler.add_instruction(instruction)
            
    def create_layer_program(self, layer_config: Dict, input_buffer: str, 
                           output_buffer: str, weight_data: Optional[np.ndarray] = None) -> List[Instruction]:
        """Create instruction sequence for executing a neural network layer"""
        instructions = []
        inst_id = 0
        
        layer_type = layer_config['type']
        
        if layer_type == 'conv2d':
            # Convolution layer instruction sequence
            
            # 1. Load weights into crossbars
            if weight_data is not None:
                load_weights_inst = Instruction(
                    inst_id=inst_id,
                    inst_type=InstructionType.LOAD_WEIGHTS,
                    operands={
                        'weight_data': weight_data,
                        'target_crossbars': layer_config.get('target_crossbars', [])
                    },
                    estimated_cycles=10
                )
                instructions.append(load_weights_inst)
                inst_id += 1
                
                # 2. Load input data
                load_input_inst = Instruction(
                    inst_id=inst_id,
                    inst_type=InstructionType.LOAD_INPUT,
                    operands={
                        'source_buffer': input_buffer,
                        'input_shape': layer_config['input_shape']
                    },
                    dependencies=[load_weights_inst.inst_id],
                    estimated_cycles=5
                )
                instructions.append(load_input_inst)
                inst_id += 1
                
                # 3. Perform crossbar computation
                xbar_compute_inst = Instruction(
                    inst_id=inst_id,
                    inst_type=InstructionType.XBAR_COMPUTE,
                    operands={
                        'operation': 'convolution',
                        'kernel_size': layer_config['kernel_size'],
                        'stride': layer_config.get('stride', (1, 1)),
                        'padding': layer_config.get('padding', 'valid')
                    },
                    dependencies=[load_input_inst.inst_id],
                    estimated_cycles=20
                )
                instructions.append(xbar_compute_inst)
                inst_id += 1
                
                # 4. Shift and add for multi-bit results
                shift_add_inst = Instruction(
                    inst_id=inst_id,
                    inst_type=InstructionType.SHIFT_ADD,
                    operands={
                        'bit_precision': layer_config.get('precision', 8)
                    },
                    dependencies=[xbar_compute_inst.inst_id],
                    estimated_cycles=5
                )
                instructions.append(shift_add_inst)
                inst_id += 1
                
                # 5. Apply activation function
                if 'activation' in layer_config:
                    activation_inst = Instruction(
                        inst_id=inst_id,
                        inst_type=InstructionType.ACTIVATION,
                        operands={
                            'activation_type': layer_config['activation']
                        },
                        dependencies=[shift_add_inst.inst_id],
                        estimated_cycles=3
                    )
                    instructions.append(activation_inst)
                    inst_id += 1
                    last_compute_inst = activation_inst
                else:
                    last_compute_inst = shift_add_inst
                    
                # 6. Store output
                store_output_inst = Instruction(
                    inst_id=inst_id,
                    inst_type=InstructionType.STORE_OUTPUT,
                    operands={
                        'target_buffer': output_buffer,
                        'output_shape': layer_config['output_shape']
                    },
                    dependencies=[last_compute_inst.inst_id],
                    estimated_cycles=3
                )
                instructions.append(store_output_inst)
                
        elif layer_type == 'dense':
            # Dense layer instruction sequence
            
            # 1. Load weights into crossbars
            if weight_data is not None:
                load_weights_inst = Instruction(
                    inst_id=inst_id,
                    inst_type=InstructionType.LOAD_WEIGHTS,
                    operands={
                        'weight_data': weight_data,
                        'target_crossbars': layer_config.get('target_crossbars', [])
                    },
                    estimated_cycles=8
                )
                instructions.append(load_weights_inst)
                inst_id += 1
                
                # 2. Load input data
                load_input_inst = Instruction(
                    inst_id=inst_id,
                    inst_type=InstructionType.LOAD_INPUT,
                    operands={
                        'source_buffer': input_buffer,
                        'input_shape': layer_config['input_shape']
                    },
                    dependencies=[load_weights_inst.inst_id],
                    estimated_cycles=4
                )
                instructions.append(load_input_inst)
                inst_id += 1
                
                # 3. Perform crossbar computation (matrix-vector multiply)
                xbar_compute_inst = Instruction(
                    inst_id=inst_id,
                    inst_type=InstructionType.XBAR_COMPUTE,
                    operands={
                        'operation': 'matrix_vector_multiply',
                        'input_size': layer_config['input_shape'][-1] if layer_config['input_shape'] else 1,
                        'output_size': layer_config['output_shape'][-1] if layer_config['output_shape'] else 1
                    },
                    dependencies=[load_input_inst.inst_id],
                    estimated_cycles=15
                )
                instructions.append(xbar_compute_inst)
                inst_id += 1
                
                # 4. Shift and add for multi-bit results
                shift_add_inst = Instruction(
                    inst_id=inst_id,
                    inst_type=InstructionType.SHIFT_ADD,
                    operands={
                        'bit_precision': layer_config.get('precision', 8)
                    },
                    dependencies=[xbar_compute_inst.inst_id],
                    estimated_cycles=4
                )
                instructions.append(shift_add_inst)
                inst_id += 1
                
                # 5. Apply activation function
                if 'activation' in layer_config and layer_config['activation']:
                    activation_inst = Instruction(
                        inst_id=inst_id,
                        inst_type=InstructionType.ACTIVATION,
                        operands={
                            'activation_type': layer_config['activation']
                        },
                        dependencies=[shift_add_inst.inst_id],
                        estimated_cycles=3
                    )
                    instructions.append(activation_inst)
                    inst_id += 1
                    last_compute_inst = activation_inst
                else:
                    last_compute_inst = shift_add_inst
                    
                # 6. Store output
                store_output_inst = Instruction(
                    inst_id=inst_id,
                    inst_type=InstructionType.STORE_OUTPUT,
                    operands={
                        'target_buffer': output_buffer,
                        'output_shape': layer_config['output_shape']
                    },
                    dependencies=[last_compute_inst.inst_id],
                    estimated_cycles=3
                )
                instructions.append(store_output_inst)
            
        elif layer_type == 'pooling':
            # Pooling layer
            load_input_inst = Instruction(
                inst_id=inst_id,
                inst_type=InstructionType.LOAD_INPUT,
                operands={
                    'source_buffer': input_buffer,
                    'input_shape': layer_config['input_shape']
                },
                estimated_cycles=3
            )
            instructions.append(load_input_inst)
            inst_id += 1
            
            pooling_inst = Instruction(
                inst_id=inst_id,
                inst_type=InstructionType.POOLING,
                operands={
                    'pooling_type': layer_config.get('pooling_type', 'max'),
                    'kernel_size': layer_config['kernel_size'],
                    'stride': layer_config.get('stride', layer_config['kernel_size'])
                },
                dependencies=[load_input_inst.inst_id],
                estimated_cycles=5
            )
            instructions.append(pooling_inst)
            inst_id += 1
            
            store_output_inst = Instruction(
                inst_id=inst_id,
                inst_type=InstructionType.STORE_OUTPUT,
                operands={
                    'target_buffer': output_buffer,
                    'output_shape': layer_config['output_shape']
                },
                dependencies=[pooling_inst.inst_id],
                estimated_cycles=2
            )
            instructions.append(store_output_inst)
            
        return instructions
        
    def execute_instruction(self, instruction: Instruction) -> int:
        """Execute a single instruction, returns completion cycle"""
        completion_cycle = self.current_cycle + instruction.estimated_cycles
        
        if instruction.inst_type == InstructionType.LOAD_WEIGHTS:
            # Simulate loading weights into crossbars
            self._execute_load_weights(instruction)
            
        elif instruction.inst_type == InstructionType.LOAD_INPUT:
            # Simulate loading input data
            self._execute_load_input(instruction)
            
        elif instruction.inst_type == InstructionType.XBAR_COMPUTE:
            # Simulate crossbar computation
            completion_cycle = self._execute_xbar_compute(instruction)
            
        elif instruction.inst_type == InstructionType.SHIFT_ADD:
            # Simulate shift-add operation
            self._execute_shift_add(instruction)
            
        elif instruction.inst_type == InstructionType.ACTIVATION:
            # Simulate activation function
            self._execute_activation(instruction)
            
        elif instruction.inst_type == InstructionType.POOLING:
            # Simulate pooling operation
            self._execute_pooling(instruction)
            
        elif instruction.inst_type == InstructionType.STORE_OUTPUT:
            # Simulate storing output
            self._execute_store_output(instruction)
            
        # Update energy consumption
        self.energy_consumption += 0.1 * instruction.estimated_cycles  # 0.1 pJ per cycle
        
        return completion_cycle
        
    def _execute_load_weights(self, instruction: Instruction):
        """Execute weight loading instruction"""
        # Simulate time to program ReRAM crossbars
        weight_data = instruction.operands['weight_data']
        target_crossbars = instruction.operands['target_crossbars']
        
        # In real implementation, this would program actual crossbars
        logging.info(f"Loading weights of shape {weight_data.shape} into {len(target_crossbars)} crossbars")
        
    def _execute_load_input(self, instruction: Instruction):
        """Execute input loading instruction"""
        source_buffer = instruction.operands['source_buffer']
        input_shape = instruction.operands['input_shape']
        
        # Simulate reading from input buffer
        if self.buffer_manager:
            # This would trigger actual buffer reads
            pass
            
        logging.info(f"Loading input of shape {input_shape} from buffer {source_buffer}")
        
    def _execute_xbar_compute(self, instruction: Instruction) -> int:
        """Execute crossbar computation"""
        operation = instruction.operands['operation']
        
        if operation == 'convolution':
            kernel_size = instruction.operands['kernel_size']
            # Simulate convolution computation time based on kernel size
            computation_cycles = kernel_size[0] * kernel_size[1] * 2
        else:
            computation_cycles = instruction.estimated_cycles
            
        logging.info(f"Executing {operation} on crossbars for {computation_cycles} cycles")
        return self.current_cycle + computation_cycles
        
    def _execute_shift_add(self, instruction: Instruction):
        """Execute shift-add operation"""
        bit_precision = instruction.operands['bit_precision']
        
        if self.compute_manager and self.compute_manager.shift_add_units:
            # Use actual shift-add unit
            shift_add_unit = self.compute_manager.shift_add_units[0]
            # This would trigger actual computation
            pass
            
        logging.info(f"Executing shift-add for {bit_precision}-bit precision")
        
    def _execute_activation(self, instruction: Instruction):
        """Execute activation function"""
        activation_type = instruction.operands['activation_type']
        
        if self.compute_manager and self.compute_manager.activation_units:
            # Use actual activation unit
            activation_unit = self.compute_manager.activation_units[0]
            # This would trigger actual computation
            pass
            
        logging.info(f"Executing {activation_type} activation")
        
    def _execute_pooling(self, instruction: Instruction):
        """Execute pooling operation"""
        pooling_type = instruction.operands['pooling_type']
        kernel_size = instruction.operands['kernel_size']
        
        if self.compute_manager and self.compute_manager.pooling_units:
            # Use actual pooling unit
            pooling_unit = self.compute_manager.pooling_units[0]
            # This would trigger actual computation
            pass
            
        logging.info(f"Executing {pooling_type} pooling with kernel {kernel_size}")
        
    def _execute_store_output(self, instruction: Instruction):
        """Execute output storage"""
        target_buffer = instruction.operands['target_buffer']
        output_shape = instruction.operands['output_shape']
        
        if self.buffer_manager:
            # This would trigger actual buffer writes
            pass
            
        logging.info(f"Storing output of shape {output_shape} to buffer {target_buffer}")
        
    def tick(self) -> Dict[str, Any]:
        """Process one cycle of microcontroller operation"""
        self.current_cycle += 1
        self.total_cycles += 1
        
        events = {
            'cycle': self.current_cycle,
            'pipeline_events': {},
            'scheduler_events': {},
            'completed_instructions': []
        }
        
        # Tick pipeline
        pipeline_events = self.pipeline.tick()
        events['pipeline_events'] = pipeline_events
        
        # Process completed instructions from pipeline
        for inst_id in pipeline_events['completed_instructions']:
            self.scheduler.mark_instruction_complete(inst_id, self.current_cycle)
            self.total_instructions_executed += 1
            events['completed_instructions'].append(inst_id)
            
        # Try to issue new instruction to pipeline
        if self.pipeline.can_issue():
            ready_instruction = self.scheduler.get_ready_instruction()
            if ready_instruction:
                if self.pipeline.issue_instruction(ready_instruction):
                    # Execute the instruction (simulate execution unit)
                    completion_cycle = self.execute_instruction(ready_instruction)
                    self.scheduler.mark_instruction_executing(ready_instruction, completion_cycle)
                    
        events['scheduler_events'] = self.scheduler.get_statistics()
        
        return events
        
    def run_until_completion(self, max_cycles: int = 10000) -> Dict[str, Any]:
        """Run microcontroller until all instructions complete"""
        execution_log = []
        
        while (self.current_cycle < max_cycles and 
               (len(self.scheduler.ready_queue) > 0 or 
                len(self.scheduler.waiting_queue) > 0 or 
                len(self.scheduler.executing_queue) > 0)):
            
            cycle_events = self.tick()
            execution_log.append(cycle_events)
            
            # Optional: print progress
            if self.current_cycle % 100 == 0:
                print(f"Cycle {self.current_cycle}: "
                      f"Ready: {len(self.scheduler.ready_queue)}, "
                      f"Waiting: {len(self.scheduler.waiting_queue)}, "
                      f"Executing: {len(self.scheduler.executing_queue)}")
                      
        return {
            'total_cycles': self.current_cycle,
            'instructions_executed': self.total_instructions_executed,
            'energy_consumption': self.energy_consumption,
            'execution_log': execution_log,
            'final_stats': self.get_statistics()
        }
        
    def get_statistics(self) -> Dict:
        """Get comprehensive microcontroller statistics"""
        return {
            'total_cycles': self.total_cycles,
            'total_instructions_executed': self.total_instructions_executed,
            'instructions_per_cycle': self.total_instructions_executed / max(self.total_cycles, 1),
            'energy_consumption': self.energy_consumption,
            'average_power': self.energy_consumption / max(self.total_cycles, 1),
            'pipeline_stats': self.pipeline.get_statistics(),
            'scheduler_stats': self.scheduler.get_statistics()
        }