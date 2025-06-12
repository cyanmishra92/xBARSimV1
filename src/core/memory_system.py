import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
from collections import deque, defaultdict
import heapq

class MemoryType(Enum):
    SRAM = "sram"
    EDRAM = "edram"
    DRAM = "dram"
    RERAM = "reram"
    CACHE = "cache"

class AccessType(Enum):
    READ = "read"
    WRITE = "write"

@dataclass
class MemoryConfig:
    """Memory configuration parameters"""
    memory_type: MemoryType
    size_kb: int
    word_size_bits: int = 32
    ports: int = 1  # Number of read/write ports
    banks: int = 1  # Number of memory banks
    
    # Timing parameters (in cycles)
    read_latency: int = 1
    write_latency: int = 1
    bank_conflict_penalty: int = 2
    refresh_period: int = 1000  # For DRAM/eDRAM
    
    # Energy parameters (in pJ)
    read_energy_per_bit: float = 1.0
    write_energy_per_bit: float = 2.0
    static_power_mw: float = 0.1
    
    # Bandwidth (bits per cycle)
    bandwidth_bits_per_cycle: int = 256

@dataclass
class MemoryRequest:
    """Memory access request"""
    request_id: int
    address: int
    size_bits: int
    access_type: AccessType
    data: Optional[Any] = None
    priority: int = 0
    timestamp: int = 0
    requester_id: str = ""

class MemoryBank:
    """Single memory bank implementation"""
    def __init__(self, bank_id: int, config: MemoryConfig):
        self.bank_id = bank_id
        self.config = config
        self.memory = {}  # Address -> Data mapping
        self.busy_until = 0  # Cycle when bank becomes free
        self.access_count = 0
        self.total_energy = 0.0
        self.last_refresh = 0
        
    def is_available(self, current_cycle: int) -> bool:
        """Check if bank is available for access"""
        return current_cycle >= self.busy_until
        
    def access(self, request: MemoryRequest, current_cycle: int) -> int:
        """Perform memory access, returns completion cycle"""
        if not self.is_available(current_cycle):
            # Bank conflict - add penalty
            start_cycle = self.busy_until + self.config.bank_conflict_penalty
        else:
            start_cycle = current_cycle
            
        # Determine latency based on access type
        if request.access_type == AccessType.READ:
            latency = self.config.read_latency
            energy_per_bit = self.config.read_energy_per_bit
            # Perform read
            request.data = self.memory.get(request.address, 0)
        else:  # WRITE
            latency = self.config.write_latency
            energy_per_bit = self.config.write_energy_per_bit
            # Perform write
            self.memory[request.address] = request.data
            
        completion_cycle = start_cycle + latency
        self.busy_until = completion_cycle
        
        # Update statistics
        self.access_count += 1
        self.total_energy += request.size_bits * energy_per_bit
        
        return completion_cycle
        
    def needs_refresh(self, current_cycle: int) -> bool:
        """Check if memory needs refresh (for DRAM/eDRAM)"""
        if self.config.memory_type in [MemoryType.DRAM, MemoryType.EDRAM]:
            return (current_cycle - self.last_refresh) >= self.config.refresh_period
        return False
        
    def refresh(self, current_cycle: int) -> int:
        """Perform memory refresh, returns completion cycle"""
        if self.config.memory_type in [MemoryType.DRAM, MemoryType.EDRAM]:
            self.last_refresh = current_cycle
            self.busy_until = max(self.busy_until, current_cycle + 5)  # Refresh takes 5 cycles
            return self.busy_until
        return current_cycle

class MemoryController:
    """Memory controller with request scheduling"""
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.banks = [MemoryBank(i, config) for i in range(config.banks)]
        self.request_queue = []  # Priority queue for requests
        self.pending_requests = {}  # request_id -> (request, completion_cycle)
        self.completed_requests = {}  # request_id -> request
        self.current_cycle = 0
        self.next_request_id = 0
        
        # Statistics
        self.total_requests = 0
        self.total_latency = 0
        self.bank_conflicts = 0
        self.refresh_count = 0
        
    def get_bank_id(self, address: int) -> int:
        """Determine which bank an address maps to"""
        return address % self.config.banks
        
    def schedule_request(self, address: int, size_bits: int, access_type: AccessType,
                        data: Optional[Any] = None, priority: int = 0, 
                        requester_id: str = "") -> int:
        """Schedule a memory request, returns request ID"""
        request = MemoryRequest(
            request_id=self.next_request_id,
            address=address,
            size_bits=size_bits,
            access_type=access_type,
            data=data,
            priority=priority,
            timestamp=self.current_cycle,
            requester_id=requester_id
        )
        
        # Add to priority queue (higher priority = lower number for heapq)
        heapq.heappush(self.request_queue, (-priority, self.current_cycle, request))
        self.next_request_id += 1
        self.total_requests += 1
        
        return request.request_id
        
    def tick(self) -> Dict[str, Any]:
        """Process one cycle of memory controller operation"""
        self.current_cycle += 1
        cycle_events = {
            'cycle': self.current_cycle,
            'completed_requests': [],
            'started_requests': [],
            'refreshes': []
        }
        
        # Check for completed requests
        completed_ids = []
        for req_id, (request, completion_cycle) in self.pending_requests.items():
            if completion_cycle <= self.current_cycle:
                self.completed_requests[req_id] = request
                cycle_events['completed_requests'].append(req_id)
                completed_ids.append(req_id)
                
                # Update latency statistics
                latency = self.current_cycle - request.timestamp
                self.total_latency += latency
                
        # Remove completed requests
        for req_id in completed_ids:
            del self.pending_requests[req_id]
            
        # Check for refresh needs
        for bank in self.banks:
            if bank.needs_refresh(self.current_cycle):
                completion_cycle = bank.refresh(self.current_cycle)
                cycle_events['refreshes'].append({
                    'bank_id': bank.bank_id,
                    'completion_cycle': completion_cycle
                })
                self.refresh_count += 1
                
        # Process new requests from queue
        while self.request_queue and len(self.pending_requests) < self.config.ports:
            # Get highest priority request
            _, _, request = heapq.heappop(self.request_queue)
            
            # Determine target bank
            bank_id = self.get_bank_id(request.address)
            bank = self.banks[bank_id]
            
            # Check if bank is available
            if bank.is_available(self.current_cycle):
                # Process the request
                completion_cycle = bank.access(request, self.current_cycle)
                self.pending_requests[request.request_id] = (request, completion_cycle)
                cycle_events['started_requests'].append({
                    'request_id': request.request_id,
                    'bank_id': bank_id,
                    'completion_cycle': completion_cycle
                })
            else:
                # Bank conflict - put request back in queue
                heapq.heappush(self.request_queue, (-request.priority, request.timestamp, request))
                self.bank_conflicts += 1
                break
                
        return cycle_events
        
    def is_request_complete(self, request_id: int) -> bool:
        """Check if a request is complete"""
        return request_id in self.completed_requests
        
    def get_request_data(self, request_id: int) -> Any:
        """Get data from completed request"""
        if request_id in self.completed_requests:
            return self.completed_requests[request_id].data
        return None
        
    def get_statistics(self) -> Dict:
        """Get memory controller statistics"""
        avg_latency = self.total_latency / max(self.total_requests, 1)
        
        return {
            'total_requests': self.total_requests,
            'completed_requests': len(self.completed_requests),
            'pending_requests': len(self.pending_requests),
            'average_latency': avg_latency,
            'bank_conflicts': self.bank_conflicts,
            'refresh_count': self.refresh_count,
            'conflict_rate': self.bank_conflicts / max(self.total_requests, 1),
            'bank_statistics': [
                {
                    'bank_id': bank.bank_id,
                    'access_count': bank.access_count,
                    'total_energy': bank.total_energy,
                    'utilization': bank.access_count / max(self.current_cycle, 1)
                }
                for bank in self.banks
            ]
        }

class BufferManager:
    """Manages data buffers and their allocation"""
    def __init__(self, buffer_configs: Dict[str, MemoryConfig]):
        self.buffers = {}
        self.controllers = {}
        
        for name, config in buffer_configs.items():
            self.controllers[name] = MemoryController(config)
            self.buffers[name] = {
                'config': config,
                'allocated_regions': {},  # region_id -> (start_addr, size, owner)
                'free_regions': [(0, config.size_kb * 1024 * 8 // config.word_size_bits)]  # (start, size) in words
            }
            
    def allocate_buffer(self, buffer_name: str, size_words: int, 
                       owner_id: str = "") -> Optional[int]:
        """Allocate buffer space, returns region_id or None if failed"""
        if buffer_name not in self.buffers:
            return None
            
        buffer = self.buffers[buffer_name]
        
        # Find suitable free region (first fit algorithm)
        for i, (start, size) in enumerate(buffer['free_regions']):
            if size >= size_words:
                # Allocate from this region
                region_id = len(buffer['allocated_regions'])
                buffer['allocated_regions'][region_id] = (start, size_words, owner_id)
                
                # Update free regions
                if size == size_words:
                    # Exact fit - remove the region
                    buffer['free_regions'].pop(i)
                else:
                    # Partial fit - update the region
                    buffer['free_regions'][i] = (start + size_words, size - size_words)
                    
                return region_id
                
        return None  # No suitable region found
        
    def deallocate_buffer(self, buffer_name: str, region_id: int) -> bool:
        """Deallocate buffer region"""
        if buffer_name not in self.buffers:
            return False
            
        buffer = self.buffers[buffer_name]
        if region_id not in buffer['allocated_regions']:
            return False
            
        start, size, _ = buffer['allocated_regions'][region_id]
        del buffer['allocated_regions'][region_id]
        
        # Add back to free regions (with merging)
        buffer['free_regions'].append((start, size))
        buffer['free_regions'].sort()
        
        # Merge adjacent free regions
        merged_regions = []
        for start, size in buffer['free_regions']:
            if merged_regions and merged_regions[-1][0] + merged_regions[-1][1] == start:
                # Merge with previous region
                prev_start, prev_size = merged_regions.pop()
                merged_regions.append((prev_start, prev_size + size))
            else:
                merged_regions.append((start, size))
                
        buffer['free_regions'] = merged_regions
        return True
        
    def write_data(self, buffer_name: str, region_id: int, offset: int, 
                   data: Any, requester_id: str = "") -> Optional[int]:
        """Write data to buffer, returns request_id"""
        if buffer_name not in self.buffers:
            return None
            
        buffer = self.buffers[buffer_name]
        if region_id not in buffer['allocated_regions']:
            return None
            
        start_addr, region_size, owner = buffer['allocated_regions'][region_id]
        if offset >= region_size:
            return None
            
        address = start_addr + offset
        data_size_bits = buffer['config'].word_size_bits  # Assuming one word
        
        return self.controllers[buffer_name].schedule_request(
            address, data_size_bits, AccessType.WRITE, data, 
            priority=1, requester_id=requester_id
        )
        
    def read_data(self, buffer_name: str, region_id: int, offset: int,
                  num_words: int = 1, requester_id: str = "") -> Optional[int]:
        """Read data from buffer, returns request_id"""
        if buffer_name not in self.buffers:
            return None
            
        buffer = self.buffers[buffer_name]
        if region_id not in buffer['allocated_regions']:
            return None
            
        start_addr, region_size, owner = buffer['allocated_regions'][region_id]
        # Assuming offset is in words. If region_size is also in words, this check is correct.
        if offset >= region_size:
            logging.warning(f"Read offset {offset} out of bounds for region {region_id} (size {region_size} words) in {buffer_name}")
            return None
            
        address = start_addr + offset # Assuming start_addr is a word address
        data_size_bits = num_words * buffer['config'].word_size_bits

        # Ensure read does not go beyond allocated region.
        # This check might need refinement based on how region_size and offset are defined (words vs bytes).
        # If offset and region_size are in words, then offset + num_words should not exceed region_size.
        if (offset + num_words) > region_size:
            logging.warning(f"Read of {num_words} words from offset {offset} exceeds region {region_id} (size {region_size} words) in {buffer_name}. Clamping read size.")
            # Adjust num_words to read only up to the end of the region
            # This simplistic clamping might not be what is always desired.
            # A more robust solution might involve how data_size was calculated by the caller.
            clamped_num_words = region_size - offset
            if clamped_num_words <= 0: # Should not happen if offset < region_size initially
                return None
            data_size_bits = clamped_num_words * buffer['config'].word_size_bits


        return self.controllers[buffer_name].schedule_request(
            address, data_size_bits, AccessType.READ, None,
            priority=1, requester_id=requester_id
        )
        
    def tick_all(self) -> Dict[str, Any]:
        """Tick all memory controllers"""
        all_events = {}
        for name, controller in self.controllers.items():
            all_events[name] = controller.tick()
        return all_events
        
    def get_buffer_utilization(self, buffer_name: str) -> float:
        """Get buffer utilization percentage"""
        if buffer_name not in self.buffers:
            return 0.0
            
        buffer = self.buffers[buffer_name]
        total_size = buffer['config'].size_kb * 1024 * 8 // buffer['config'].word_size_bits
        
        allocated_size = sum(size for _, size, _ in buffer['allocated_regions'].values())
        return allocated_size / total_size if total_size > 0 else 0.0
        
    def get_all_statistics(self) -> Dict:
        """Get statistics for all buffers"""
        stats = {}
        for name, controller in self.controllers.items():
            stats[name] = {
                'memory_stats': controller.get_statistics(),
                'utilization': self.get_buffer_utilization(name),
                'allocated_regions': len(self.buffers[name]['allocated_regions']),
                'free_regions': len(self.buffers[name]['free_regions'])
            }
        return stats

class PartialSumBuffer:
    """Specialized buffer for managing partial sums in neural network computations"""
    def __init__(self, size_entries: int, entry_bits: int = 32):
        self.size_entries = size_entries
        self.entry_bits = entry_bits
        self.buffer = np.zeros(size_entries, dtype=np.float32)
        self.valid = np.zeros(size_entries, dtype=bool)
        self.access_count = 0
        self.overflow_count = 0
        
    def accumulate(self, index: int, value: float) -> bool:
        """Accumulate value into partial sum buffer"""
        if index >= self.size_entries:
            self.overflow_count += 1
            return False
            
        self.buffer[index] += value
        self.valid[index] = True
        self.access_count += 1
        return True
        
    def read_and_clear(self, index: int) -> float:
        """Read partial sum and clear the entry"""
        if index >= self.size_entries:
            return 0.0
            
        value = self.buffer[index]
        self.buffer[index] = 0.0
        self.valid[index] = False
        self.access_count += 1
        return value
        
    def read_range(self, start_idx: int, end_idx: int) -> np.ndarray:
        """Read range of partial sums"""
        end_idx = min(end_idx, self.size_entries)
        start_idx = max(0, start_idx)
        return self.buffer[start_idx:end_idx].copy()
        
    def clear_all(self):
        """Clear all partial sum entries"""
        self.buffer.fill(0.0)
        self.valid.fill(False)
        
    def get_statistics(self) -> Dict:
        """Get partial sum buffer statistics"""
        return {
            'size_entries': self.size_entries,
            'access_count': self.access_count,
            'overflow_count': self.overflow_count,
            'utilization': np.sum(self.valid) / self.size_entries,
            'active_entries': np.sum(self.valid)
        }