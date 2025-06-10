import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import heapq
import logging
from collections import deque, defaultdict

class InterconnectTopology(Enum):
    BUS = "bus"
    MESH = "mesh"
    TORUS = "torus"
    CROSSBAR = "crossbar"
    TREE = "tree"
    RING = "ring"

class MessageType(Enum):
    DATA = "data"
    CONTROL = "control"
    SYNCHRONIZATION = "sync"
    MEMORY_REQUEST = "memory_request"
    MEMORY_RESPONSE = "memory_response"

@dataclass
class Message:
    """Message for interconnect communication"""
    msg_id: int
    msg_type: MessageType
    source: Tuple[int, ...]  # Source coordinates (supertile, tile, etc.)
    destination: Tuple[int, ...]  # Destination coordinates
    payload_bits: int
    priority: int = 0
    timestamp: int = 0
    hop_count: int = 0
    latency: int = 0
    data: Optional[Any] = None

@dataclass
class InterconnectConfig:
    """Interconnect configuration"""
    topology: InterconnectTopology
    data_width_bits: int = 256
    clock_frequency_mhz: float = 1000.0
    router_latency_cycles: int = 1
    link_latency_cycles: int = 1
    buffer_size_flits: int = 16
    virtual_channels: int = 2
    flow_control: str = "credit_based"  # "credit_based", "on_off", "handshake"
    
    # Energy parameters (pJ per bit per mm)
    link_energy_per_bit_per_mm: float = 0.1
    router_energy_per_bit: float = 0.05
    buffer_energy_per_bit: float = 0.01

class Router:
    """Network router implementation"""
    def __init__(self, router_id: Tuple[int, ...], config: InterconnectConfig, 
                 neighbors: List[Tuple[int, ...]]):
        self.router_id = router_id
        self.config = config
        self.neighbors = neighbors
        self.input_buffers = {neighbor: deque(maxlen=config.buffer_size_flits) 
                             for neighbor in neighbors + [router_id]}  # Include local port
        self.output_buffers = {neighbor: deque(maxlen=config.buffer_size_flits) 
                              for neighbor in neighbors + [router_id]}
        self.routing_table = {}
        self.current_cycle = 0
        
        # Statistics
        self.messages_routed = 0
        self.total_latency = 0
        self.buffer_overflows = 0
        self.energy_consumption = 0.0
        
        # Build routing table
        self._build_routing_table()
        
    def _build_routing_table(self):
        """Build routing table based on topology"""
        # This is a simplified routing table
        # In practice, this would be more sophisticated
        if self.config.topology == InterconnectTopology.MESH:
            self._build_mesh_routing_table()
        elif self.config.topology == InterconnectTopology.BUS:
            self._build_bus_routing_table()
        # Add other topologies as needed
        
    def _build_mesh_routing_table(self):
        """Build routing table for mesh topology using dimension-order routing"""
        # For 2D mesh, route in X dimension first, then Y dimension
        pass  # Simplified for now
        
    def _build_bus_routing_table(self):
        """Build routing table for bus topology"""
        # All messages go to the bus
        for neighbor in self.neighbors:
            self.routing_table[neighbor] = neighbor
            
    def route_message(self, message: Message) -> Optional[Tuple[int, ...]]:
        """Determine next hop for message"""
        destination = message.destination
        
        if destination == self.router_id:
            # Message has reached destination
            return self.router_id
            
        if self.config.topology == InterconnectTopology.MESH:
            # Dimension-order routing for mesh
            if len(self.router_id) == 2 and len(destination) == 2:
                curr_x, curr_y = self.router_id
                dest_x, dest_y = destination
                
                if curr_x != dest_x:
                    # Route in X dimension
                    if curr_x < dest_x:
                        return (curr_x + 1, curr_y)
                    else:
                        return (curr_x - 1, curr_y)
                elif curr_y != dest_y:
                    # Route in Y dimension
                    if curr_y < dest_y:
                        return (curr_x, curr_y + 1)
                    else:
                        return (curr_x, curr_y - 1)
                        
        elif self.config.topology == InterconnectTopology.BUS:
            # For bus, all non-local traffic goes to bus controller
            if self.router_id != (0, 0):  # Assuming (0,0) is bus controller
                return (0, 0)
            else:
                # Bus controller routes to destination
                return destination
                
        # Default: route to first available neighbor
        if self.neighbors:
            return self.neighbors[0]
            
        return None
        
    def accept_message(self, message: Message, from_port: Tuple[int, ...]) -> bool:
        """Accept message from input port"""
        input_buffer = self.input_buffers[from_port]
        
        if len(input_buffer) < self.config.buffer_size_flits:
            message.hop_count += 1
            input_buffer.append(message)
            return True
        else:
            self.buffer_overflows += 1
            return False
            
    def tick(self) -> Dict[str, Any]:
        """Process one cycle of router operation"""
        self.current_cycle += 1
        events = {
            'cycle': self.current_cycle,
            'router_id': self.router_id,
            'messages_processed': 0,
            'buffer_utilization': {}
        }
        
        # Process messages in input buffers
        for from_port, input_buffer in self.input_buffers.items():
            if input_buffer:
                message = input_buffer.popleft()
                next_hop = self.route_message(message)
                
                if next_hop == self.router_id:
                    # Message reached destination - deliver locally
                    message.latency = self.current_cycle - message.timestamp
                    self.total_latency += message.latency
                    events['messages_processed'] += 1
                elif next_hop and next_hop in self.output_buffers:
                    # Forward message
                    output_buffer = self.output_buffers[next_hop]
                    if len(output_buffer) < self.config.buffer_size_flits:
                        message.hop_count += 1
                        output_buffer.append(message)
                        
                        # Update energy consumption
                        self.energy_consumption += (message.payload_bits * 
                                                   self.config.router_energy_per_bit)
                    else:
                        # Output buffer full - drop message or stall
                        input_buffer.appendleft(message)  # Put back in input buffer
                        
                self.messages_routed += 1
                
        # Calculate buffer utilization
        for port, buffer in self.input_buffers.items():
            utilization = len(buffer) / self.config.buffer_size_flits
            events['buffer_utilization'][f'input_{port}'] = utilization
            
        return events
        
    def get_statistics(self) -> Dict:
        """Get router statistics"""
        avg_latency = self.total_latency / max(self.messages_routed, 1)
        return {
            'router_id': self.router_id,
            'messages_routed': self.messages_routed,
            'average_latency': avg_latency,
            'buffer_overflows': self.buffer_overflows,
            'energy_consumption': self.energy_consumption,
            'current_cycle': self.current_cycle
        }

class InterconnectNetwork:
    """Complete interconnect network"""
    def __init__(self, config: InterconnectConfig, topology_dims: Tuple[int, ...]):
        self.config = config
        self.topology_dims = topology_dims
        self.routers = {}
        self.links = {}
        self.current_cycle = 0
        
        # Message tracking
        self.message_queue = []  # Messages to be injected
        self.in_flight_messages = {}  # msg_id -> message
        self.completed_messages = {}  # msg_id -> message
        self.next_message_id = 0
        
        # Build network topology
        self._build_topology()
        
        # Statistics
        self.total_messages_sent = 0
        self.total_messages_received = 0
        self.total_network_latency = 0
        self.network_energy = 0.0
        
    def _build_topology(self):
        """Build network topology"""
        if self.config.topology == InterconnectTopology.MESH:
            self._build_mesh_topology()
        elif self.config.topology == InterconnectTopology.BUS:
            self._build_bus_topology()
        # Add other topologies
        
    def _build_mesh_topology(self):
        """Build 2D mesh topology"""
        if len(self.topology_dims) != 2:
            raise ValueError("Mesh topology requires 2D dimensions")
            
        rows, cols = self.topology_dims
        
        # Create routers
        for i in range(rows):
            for j in range(cols):
                router_id = (i, j)
                neighbors = []
                
                # Add neighbors (up, down, left, right)
                if i > 0:
                    neighbors.append((i-1, j))  # Up
                if i < rows - 1:
                    neighbors.append((i+1, j))  # Down
                if j > 0:
                    neighbors.append((i, j-1))  # Left
                if j < cols - 1:
                    neighbors.append((i, j+1))  # Right
                    
                self.routers[router_id] = Router(router_id, self.config, neighbors)
                
        # Create links
        for router_id, router in self.routers.items():
            for neighbor_id in router.neighbors:
                link_id = (router_id, neighbor_id)
                if link_id not in self.links:
                    self.links[link_id] = {
                        'latency': self.config.link_latency_cycles,
                        'bandwidth': self.config.data_width_bits,
                        'utilization': 0.0,
                        'messages_in_transit': deque()
                    }
                    
    def _build_bus_topology(self):
        """Build bus topology"""
        num_nodes = self.topology_dims[0]
        
        # Create bus controller at (0, 0)
        bus_controller_id = (0, 0)
        all_nodes = [(0, i) for i in range(num_nodes)]
        self.routers[bus_controller_id] = Router(bus_controller_id, self.config, all_nodes[1:])
        
        # Create other nodes
        for i in range(1, num_nodes):
            node_id = (0, i)
            self.routers[node_id] = Router(node_id, self.config, [bus_controller_id])
            
        # Create bus links
        for i in range(1, num_nodes):
            link_id = (bus_controller_id, (0, i))
            self.links[link_id] = {
                'latency': self.config.link_latency_cycles,
                'bandwidth': self.config.data_width_bits,
                'utilization': 0.0,
                'messages_in_transit': deque()
            }
            
    def send_message(self, source: Tuple[int, ...], destination: Tuple[int, ...],
                    msg_type: MessageType, payload_bits: int, 
                    data: Optional[Any] = None, priority: int = 0) -> int:
        """Send message through network"""
        message = Message(
            msg_id=self.next_message_id,
            msg_type=msg_type,
            source=source,
            destination=destination,
            payload_bits=payload_bits,
            priority=priority,
            timestamp=self.current_cycle,
            data=data
        )
        
        self.message_queue.append(message)
        self.next_message_id += 1
        self.total_messages_sent += 1
        
        return message.msg_id
        
    def inject_messages(self):
        """Inject messages from queue into network"""
        for message in self.message_queue[:]:
            source_router = self.routers.get(message.source)
            if source_router:
                # Try to inject message
                if source_router.accept_message(message, message.source):
                    self.in_flight_messages[message.msg_id] = message
                    self.message_queue.remove(message)
                    
    def tick(self) -> Dict[str, Any]:
        """Process one cycle of network operation"""
        self.current_cycle += 1
        events = {
            'cycle': self.current_cycle,
            'router_events': [],
            'link_events': [],
            'completed_messages': []
        }
        
        # Inject new messages
        self.inject_messages()
        
        # Tick all routers
        for router_id, router in self.routers.items():
            router_events = router.tick()
            events['router_events'].append(router_events)
            
            # Check for completed messages
            if router_events['messages_processed'] > 0:
                # Find completed messages
                for msg_id, message in list(self.in_flight_messages.items()):
                    if message.destination == router_id:
                        self.completed_messages[msg_id] = message
                        del self.in_flight_messages[msg_id]
                        events['completed_messages'].append(msg_id)
                        self.total_messages_received += 1
                        self.total_network_latency += message.latency
                        
        # Process links (simulate link delays)
        for link_id, link in self.links.items():
            # Move messages through link pipeline
            if link['messages_in_transit']:
                # Check if any messages complete transit
                while (link['messages_in_transit'] and 
                       link['messages_in_transit'][0]['completion_cycle'] <= self.current_cycle):
                    msg_info = link['messages_in_transit'].popleft()
                    message = msg_info['message']
                    dest_router = self.routers[link_id[1]]
                    
                    # Try to deliver to destination router
                    if dest_router.accept_message(message, link_id[0]):
                        # Successfully delivered
                        pass
                    else:
                        # Destination buffer full - put back
                        link['messages_in_transit'].appendleft(msg_info)
                        break
                        
            # Calculate link utilization
            link['utilization'] = len(link['messages_in_transit']) / 10  # Assume max 10 messages in link
            
        return events
        
    def is_message_complete(self, msg_id: int) -> bool:
        """Check if message transmission is complete"""
        return msg_id in self.completed_messages
        
    def get_message_latency(self, msg_id: int) -> Optional[int]:
        """Get latency for completed message"""
        if msg_id in self.completed_messages:
            return self.completed_messages[msg_id].latency
        return None
        
    def calculate_distance(self, source: Tuple[int, ...], dest: Tuple[int, ...]) -> int:
        """Calculate distance between two nodes"""
        if self.config.topology == InterconnectTopology.MESH:
            # Manhattan distance for mesh
            return sum(abs(s - d) for s, d in zip(source, dest))
        elif self.config.topology == InterconnectTopology.BUS:
            # Bus distance is always 2 hops (except for direct to bus controller)
            if dest == (0, 0) or source == (0, 0):
                return 1
            else:
                return 2
        return 1
        
    def get_network_statistics(self) -> Dict:
        """Get comprehensive network statistics"""
        avg_network_latency = (self.total_network_latency / 
                              max(self.total_messages_received, 1))
        
        router_stats = []
        for router_id, router in self.routers.items():
            router_stats.append(router.get_statistics())
            
        link_utilizations = []
        for link_id, link in self.links.items():
            link_utilizations.append(link['utilization'])
            
        return {
            'total_messages_sent': self.total_messages_sent,
            'total_messages_received': self.total_messages_received,
            'messages_in_flight': len(self.in_flight_messages),
            'average_network_latency': avg_network_latency,
            'network_energy': self.network_energy,
            'average_link_utilization': np.mean(link_utilizations) if link_utilizations else 0,
            'router_statistics': router_stats,
            'current_cycle': self.current_cycle
        }

class TimingModel:
    """Cycle-accurate timing model for the entire system"""
    def __init__(self):
        self.global_cycle = 0
        self.component_cycles = {}  # component_id -> local_cycle
        self.event_queue = []  # Priority queue for scheduled events
        self.next_event_id = 0
        
        # Timing constraints
        self.clock_domains = {}  # component_id -> clock_frequency
        self.synchronization_points = []  # Global sync points
        
    def register_component(self, component_id: str, clock_frequency_mhz: float):
        """Register component with its clock frequency"""
        self.component_cycles[component_id] = 0
        self.clock_domains[component_id] = clock_frequency_mhz
        
    def schedule_event(self, component_id: str, delay_cycles: int, 
                      event_type: str, event_data: Any = None) -> int:
        """Schedule event for future execution"""
        execution_cycle = self.component_cycles.get(component_id, 0) + delay_cycles
        
        event = {
            'event_id': self.next_event_id,
            'component_id': component_id,
            'execution_cycle': execution_cycle,
            'event_type': event_type,
            'event_data': event_data
        }
        
        heapq.heappush(self.event_queue, (execution_cycle, self.next_event_id, event))
        self.next_event_id += 1
        
        return event['event_id']
        
    def advance_component_clock(self, component_id: str, cycles: int = 1):
        """Advance component clock"""
        if component_id in self.component_cycles:
            self.component_cycles[component_id] += cycles
            
    def advance_global_clock(self, cycles: int = 1):
        """Advance global clock"""
        self.global_cycle += cycles
        
        # Process events that should execute at this cycle
        ready_events = []
        while (self.event_queue and 
               self.event_queue[0][0] <= self.global_cycle):
            _, _, event = heapq.heappop(self.event_queue)
            ready_events.append(event)
            
        return ready_events
        
    def synchronize_all_components(self):
        """Synchronize all components to global clock"""
        max_cycle = max(self.component_cycles.values()) if self.component_cycles else 0
        self.global_cycle = max(self.global_cycle, max_cycle)
        
        for component_id in self.component_cycles:
            self.component_cycles[component_id] = self.global_cycle
            
    def get_timing_statistics(self) -> Dict:
        """Get timing model statistics"""
        return {
            'global_cycle': self.global_cycle,
            'component_cycles': self.component_cycles.copy(),
            'pending_events': len(self.event_queue),
            'clock_domains': self.clock_domains.copy()
        }