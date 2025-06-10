"""
Text-based visualization for terminal display (WSL-friendly)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any

def print_architecture_diagram(chip):
    """Print a text-based architecture diagram"""
    print("\n" + "═" * 80)
    print("📊 CHIP ARCHITECTURE DIAGRAM")
    print("═" * 80)
    
    config = chip.get_chip_configuration()
    hierarchy = config['hierarchy']
    
    print("🔹 Chip Overview:")
    print(f"   ├── SuperTiles: {hierarchy['supertiles']}")
    print(f"   ├── Total Tiles: {hierarchy['supertiles'] * hierarchy['tiles_per_supertile']}")
    print(f"   ├── Total Crossbars: {hierarchy['total_crossbars']}")
    print(f"   └── Total ReRAM Cells: {config['compute_capacity']['total_ReRAM_cells']:,}")
    
    print("\n🔹 Hierarchy Structure:")
    for st_idx in range(hierarchy['supertiles']):
        print(f"   SuperTile_{st_idx}")
        print(f"   ├── Shared Buffer: {config['memory_hierarchy']['shared_buffers']}")
        
        for tile_idx in range(hierarchy['tiles_per_supertile']):
            print(f"   ├── Tile_{tile_idx}")
            print(f"   │   ├── Local Buffer: {config['memory_hierarchy']['local_buffers']}")
            
            for xbar_idx in range(hierarchy['crossbars_per_tile']):
                symbol = "└──" if xbar_idx == hierarchy['crossbars_per_tile'] - 1 else "├──"
                print(f"   │   {symbol} Crossbar_{xbar_idx}: {config['compute_capacity']['crossbar_size']}")
                
            if tile_idx < hierarchy['tiles_per_supertile'] - 1:
                print("   │")
        
        if st_idx < hierarchy['supertiles'] - 1:
            print("   │")
    
    print(f"\n🔹 Global Memory: {config['memory_hierarchy']['global_buffer']}")
    print("═" * 80)

def print_crossbar_heatmap(crossbar, title="Crossbar Weight Map"):
    """Print a simplified heatmap of crossbar weights"""
    print(f"\n📈 {title}")
    print("─" * 60)
    
    # Get conductance matrix (simplified view)
    conductance_matrix = crossbar.get_conductance_matrix()
    
    # Sample a smaller section for display (e.g., 8x8 from corner)
    sample_size = min(8, conductance_matrix.shape[0], conductance_matrix.shape[1])
    sample = conductance_matrix[:sample_size, :sample_size]
    
    # Normalize to 0-9 range for character display
    if sample.max() > sample.min():
        normalized = ((sample - sample.min()) / (sample.max() - sample.min()) * 9).astype(int)
    else:
        normalized = np.zeros_like(sample, dtype=int)
    
    print("    " + "".join([f"{i:2d}" for i in range(sample_size)]))
    print("  ┌" + "──" * sample_size + "┐")
    
    for i in range(sample_size):
        row_str = f"{i:2d}│"
        for j in range(sample_size):
            # Use different characters for different weight levels
            chars = " .·▫▪▬█"
            if normalized[i, j] < len(chars):
                char = chars[normalized[i, j]]
            else:
                char = "█"
            row_str += f"{char} "
        row_str += "│"
        print(row_str)
    
    print("  └" + "──" * sample_size + "┘")
    print(f"Range: {sample.min():.2e} to {sample.max():.2e} S")
    print("Legend: ' '=Low, '·'=Med, '█'=High conductance")

def print_dataflow_diagram(layer_mappings, current_layer=None):
    """Print dataflow through the system"""
    print("\n🌊 DATAFLOW DIAGRAM")
    print("─" * 60)
    
    if not layer_mappings:
        print("No layer mappings available")
        return
    
    for i, (layer_name, mapping) in enumerate(layer_mappings.items()):
        arrow = "➤" if i == current_layer else "→"
        status = "🔄" if i == current_layer else "✓" if i < (current_layer or -1) else "⏸"
        
        print(f"{status} Layer {i}: {mapping['layer_type']}")
        
        allocation = mapping['crossbar_allocation']
        crossbars_used = allocation['total_crossbars_used']
        pattern = f"{allocation['crossbars_per_col']}×{allocation['crossbars_per_row']}"
        
        print(f"   {arrow} Crossbars: {crossbars_used} ({pattern} layout)")
        print(f"   {arrow} Dataflow: {mapping['dataflow_pattern']}")
        
        if i < len(layer_mappings) - 1:
            print("   ↓")

def print_performance_summary(statistics):
    """Print performance metrics summary"""
    print("\n📊 PERFORMANCE SUMMARY")
    print("═" * 60)
    
    if 'chip_statistics' in statistics:
        chip_stats = statistics['chip_statistics']
        if 'performance' in chip_stats:
            perf = chip_stats['performance']
            print(f"🔹 Chip Performance:")
            print(f"   ├── Total Operations: {perf.get('total_operations', 'N/A'):,}")
            print(f"   ├── Crossbar Ops: {perf.get('total_crossbar_operations', 'N/A'):,}")
            print(f"   └── Tile Ops: {perf.get('total_tile_operations', 'N/A'):,}")
    
    if 'memory_statistics' in statistics:
        memory_stats = statistics['memory_statistics']
        print(f"\n🔹 Memory Performance:")
        for buffer_name, stats in memory_stats.items():
            memory_info = stats.get('memory_stats', {})
            print(f"   ├── {buffer_name}:")
            print(f"   │   ├── Requests: {memory_info.get('total_requests', 'N/A'):,}")
            print(f"   │   ├── Avg Latency: {memory_info.get('average_latency', 'N/A'):.1f} cycles")
            print(f"   │   └── Utilization: {stats.get('utilization', 0):.1%}")
    
    if 'microcontroller_statistics' in statistics:
        mcu_stats = statistics['microcontroller_statistics']
        print(f"\n🔹 Microcontroller Performance:")
        print(f"   ├── Instructions: {mcu_stats.get('total_instructions_executed', 'N/A'):,}")
        print(f"   ├── Total Cycles: {mcu_stats.get('total_cycles', 'N/A'):,}")
        print(f"   ├── IPC: {mcu_stats.get('instructions_per_cycle', 0):.3f}")
        print(f"   └── Energy: {mcu_stats.get('energy_consumption', 0):.2e} J")

def print_layer_execution_log(layer_log):
    """Print layer execution timeline"""
    print("\n⏱️  LAYER EXECUTION TIMELINE")
    print("─" * 60)
    
    if not layer_log:
        print("No execution log available")
        return
    
    total_cycles = sum(layer['execution_cycles'] for layer in layer_log)
    
    for layer in layer_log:
        layer_idx = layer['layer_index']
        layer_type = layer['layer_type']
        cycles = layer['execution_cycles']
        percentage = (cycles / total_cycles * 100) if total_cycles > 0 else 0
        
        # Create a simple progress bar
        bar_length = 30
        filled = int(bar_length * percentage / 100)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        print(f"Layer {layer_idx} ({layer_type}):")
        print(f"  [{bar}] {cycles:,} cycles ({percentage:.1f}%)")
        print(f"  Input: {layer['input_shape']} → Output: {layer['output_shape']}")
    
    print(f"\nTotal Execution: {total_cycles:,} cycles")

def print_utilization_chart(statistics):
    """Print hardware utilization chart"""
    print("\n📈 HARDWARE UTILIZATION")
    print("─" * 60)
    
    components = []
    
    # Extract utilization data
    if 'memory_statistics' in statistics:
        for name, stats in statistics['memory_statistics'].items():
            util = stats.get('utilization', 0) * 100
            components.append((name.replace('_', ' ').title(), util))
    
    if 'compute_statistics' in statistics:
        compute_stats = statistics['compute_statistics']
        if 'shift_add_units' in compute_stats:
            # Simplified compute utilization
            components.append(("Compute Units", 75.0))  # Placeholder
    
    # Print utilization bars
    for name, utilization in components:
        bar_length = 40
        filled = int(bar_length * utilization / 100)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        print(f"{name:15s} [{bar}] {utilization:5.1f}%")

def print_energy_breakdown(statistics):
    """Print energy consumption breakdown"""
    print("\n⚡ ENERGY BREAKDOWN")
    print("─" * 60)
    
    energy_components = []
    total_energy = 0
    
    # Extract energy data from various components
    if 'memory_statistics' in statistics:
        for name, stats in statistics['memory_statistics'].items():
            memory_stats = stats.get('memory_stats', {})
            # Calculate energy for each bank
            bank_stats = memory_stats.get('bank_statistics', [])
            memory_energy = sum(bank.get('total_energy', 0) for bank in bank_stats)
            if memory_energy > 0:
                energy_components.append((name.replace('_', ' ').title(), memory_energy))
                total_energy += memory_energy
    
    if 'microcontroller_statistics' in statistics:
        mcu_energy = statistics['microcontroller_statistics'].get('energy_consumption', 0)
        if mcu_energy > 0:
            energy_components.append(("Microcontroller", mcu_energy))
            total_energy += mcu_energy
    
    if energy_components:
        for name, energy in energy_components:
            percentage = (energy / total_energy * 100) if total_energy > 0 else 0
            bar_length = 30
            filled = int(bar_length * percentage / 100)
            bar = "█" * filled + "░" * (bar_length - filled)
            
            print(f"{name:15s} [{bar}] {energy:.2e} J ({percentage:.1f}%)")
        
        print(f"\nTotal Energy: {total_energy:.2e} J")
    else:
        print("No energy data available")

def print_detailed_hardware_analysis(chip, statistics):
    """Print detailed hardware component analysis"""
    print("\n🔧 DETAILED HARDWARE ANALYSIS")
    print("═" * 80)
    
    # Crossbar Analysis
    print("🔹 Crossbar Array Analysis:")
    total_crossbars = 0
    total_operations = 0
    
    for st_idx, supertile in enumerate(chip.supertiles):
        for t_idx, tile in enumerate(supertile.tiles):
            for xb_idx, crossbar in enumerate(tile.crossbars):
                total_crossbars += 1
                xb_stats = crossbar.get_statistics()
                ops = xb_stats.get('total_operations', 0)
                total_operations += ops
                
                if ops > 0:  # Only show active crossbars
                    print(f"   ├── ST{st_idx}_T{t_idx}_XB{xb_idx}: {ops:,} operations")
    
    print(f"   └── Total: {total_crossbars} crossbars, {total_operations:,} operations")
    
    # Peripheral Circuit Analysis
    print(f"\n🔹 Peripheral Circuit Analysis:")
    total_adcs = 0
    total_dacs = 0
    total_adc_ops = 0
    total_dac_ops = 0
    
    for st_idx, supertile in enumerate(chip.supertiles):
        for t_idx, tile in enumerate(supertile.tiles):
            if hasattr(tile, 'peripheral_manager'):
                pm = tile.peripheral_manager
                pm_stats = pm.get_statistics()
                
                # ADC statistics
                adc_stats = pm_stats.get('individual_stats', {}).get('adcs', [])
                tile_adc_ops = sum(stat.get('conversion_count', 0) for stat in adc_stats)
                total_adc_ops += tile_adc_ops
                total_adcs += len(adc_stats)
                
                # DAC statistics  
                dac_stats = pm_stats.get('individual_stats', {}).get('dacs', [])
                tile_dac_ops = sum(stat.get('conversion_count', 0) for stat in dac_stats)
                total_dac_ops += tile_dac_ops
                total_dacs += len(dac_stats)
                
                if tile_adc_ops > 0 or tile_dac_ops > 0:
                    print(f"   ├── ST{st_idx}_T{t_idx}: ADCs: {tile_adc_ops:,} ops, DACs: {tile_dac_ops:,} ops")
    
    print(f"   ├── Total ADCs: {total_adcs} units, {total_adc_ops:,} conversions")
    print(f"   └── Total DACs: {total_dacs} units, {total_dac_ops:,} conversions")
    
    # Memory Operation Analysis
    print(f"\n🔹 Memory Operation Analysis:")
    if 'memory_statistics' in statistics:
        memory_stats = statistics['memory_statistics']
        total_mem_ops = 0
        total_mem_energy = 0
        
        for buffer_name, stats in memory_stats.items():
            memory_info = stats.get('memory_stats', {})
            bank_stats = memory_info.get('bank_statistics', [])
            
            buffer_ops = sum(bank.get('access_count', 0) for bank in bank_stats)
            buffer_energy = sum(bank.get('total_energy', 0) for bank in bank_stats)
            
            total_mem_ops += buffer_ops
            total_mem_energy += buffer_energy
            
            if buffer_ops > 0:
                avg_latency = memory_info.get('average_latency', 0)
                conflict_rate = memory_info.get('conflict_rate', 0) * 100
                print(f"   ├── {buffer_name.replace('_', ' ').title()}:")
                print(f"   │   ├── Operations: {buffer_ops:,}")
                print(f"   │   ├── Energy: {buffer_energy:.2e} J")
                print(f"   │   ├── Avg Latency: {avg_latency:.1f} cycles")
                print(f"   │   └── Conflict Rate: {conflict_rate:.1f}%")
        
        print(f"   └── Total Memory Ops: {total_mem_ops:,}, Energy: {total_mem_energy:.2e} J")

def print_bottleneck_analysis(statistics, layer_log):
    """Analyze and print system bottlenecks"""
    print("\n🚨 BOTTLENECK ANALYSIS")
    print("═" * 80)
    
    # Time breakdown analysis
    if layer_log:
        print("🔹 Execution Time Breakdown:")
        total_cycles = sum(layer['execution_cycles'] for layer in layer_log)
        
        # Identify the slowest layer
        slowest_layer = max(layer_log, key=lambda x: x['execution_cycles'])
        print(f"   ├── Slowest Layer: Layer {slowest_layer['layer_index']} ({slowest_layer['layer_type']})")
        print(f"   │   └── {slowest_layer['execution_cycles']:,} cycles ({slowest_layer['execution_cycles']/total_cycles*100:.1f}%)")
        
        # Show layer time distribution
        for layer in sorted(layer_log, key=lambda x: x['execution_cycles'], reverse=True):
            percentage = (layer['execution_cycles'] / total_cycles * 100) if total_cycles > 0 else 0
            if percentage > 5:  # Only show layers taking >5% of time
                print(f"   ├── Layer {layer['layer_index']}: {percentage:.1f}% ({layer['execution_cycles']:,} cycles)")
    
    # Memory bottleneck analysis
    if 'memory_statistics' in statistics:
        print(f"\n🔹 Memory Bottlenecks:")
        memory_stats = statistics['memory_statistics']
        
        highest_latency = 0
        highest_conflicts = 0
        bottleneck_buffer = None
        conflict_buffer = None
        
        for buffer_name, stats in memory_stats.items():
            memory_info = stats.get('memory_stats', {})
            avg_latency = memory_info.get('average_latency', 0)
            conflict_rate = memory_info.get('conflict_rate', 0)
            
            if avg_latency > highest_latency:
                highest_latency = avg_latency
                bottleneck_buffer = buffer_name
                
            if conflict_rate > highest_conflicts:
                highest_conflicts = conflict_rate
                conflict_buffer = buffer_name
        
        if bottleneck_buffer:
            print(f"   ├── Highest Latency: {bottleneck_buffer.replace('_', ' ').title()} ({highest_latency:.1f} cycles)")
            
        if conflict_buffer and highest_conflicts > 0.01:
            print(f"   ├── Most Conflicts: {conflict_buffer.replace('_', ' ').title()} ({highest_conflicts*100:.1f}%)")
            
        # Utilization analysis
        low_util_buffers = []
        for buffer_name, stats in memory_stats.items():
            utilization = stats.get('utilization', 0)
            if utilization < 0.1:  # Less than 10% utilization
                low_util_buffers.append((buffer_name, utilization))
        
        if low_util_buffers:
            print(f"   └── Underutilized Buffers:")
            for buffer_name, util in low_util_buffers:
                print(f"       └── {buffer_name.replace('_', ' ').title()}: {util:.1%}")
    
    # Resource utilization recommendations
    print(f"\n🔹 Optimization Recommendations:")
    
    # Check crossbar utilization
    if 'chip_statistics' in statistics:
        chip_stats = statistics['chip_statistics']
        if 'performance' in chip_stats:
            crossbar_ops = chip_stats['performance'].get('total_crossbar_operations', 0)
            if crossbar_ops == 0:
                print(f"   ⚠️  No crossbar operations detected - check weight mapping")
    
    # Check microcontroller utilization
    if 'microcontroller_statistics' in statistics:
        mcu_stats = statistics['microcontroller_statistics']
        ipc = mcu_stats.get('instructions_per_cycle', 0)
        if ipc < 0.1:
            print(f"   ⚠️  Low IPC ({ipc:.3f}) - microcontroller underutilized")
    
    print(f"   ✓ Analysis complete - check metrics above for optimization opportunities")

def create_complete_text_report(chip, dnn_manager, execution_result=None):
    """Create a complete text-based report"""
    print("\n" + "═" * 80)
    print("🎯 RERAM CROSSBAR SIMULATOR - COMPLETE REPORT")
    print("═" * 80)
    
    # Architecture overview
    print_architecture_diagram(chip)
    
    # DNN mapping
    if dnn_manager.layer_mappings:
        print_dataflow_diagram(dnn_manager.layer_mappings)
    
    # Sample crossbar visualization
    if chip.supertiles and chip.supertiles[0].tiles and chip.supertiles[0].tiles[0].crossbars:
        sample_crossbar = chip.supertiles[0].tiles[0].crossbars[0]
        print_crossbar_heatmap(sample_crossbar, "Sample Crossbar (Tile 0, Crossbar 0)")
    
    # Performance results
    if execution_result and execution_result.get('success'):
        print_layer_execution_log(execution_result['layer_execution_log'])
        print_performance_summary(execution_result['system_statistics'])
        print_utilization_chart(execution_result['system_statistics'])
        print_energy_breakdown(execution_result['system_statistics'])
        
        # Detailed hardware analysis
        print_detailed_hardware_analysis(chip, execution_result['system_statistics'])
        
        # Bottleneck analysis
        print_bottleneck_analysis(execution_result['system_statistics'], 
                                 execution_result['layer_execution_log'])
        
        # Inference results
        inference_result = execution_result['inference_result']
        print(f"\n🎯 INFERENCE RESULT")
        print("─" * 60)
        print(f"Predicted Class: {inference_result['predicted_class']}")
        print(f"Confidence: {inference_result['confidence']:.3f}")
        print(f"Total Cycles: {execution_result['total_execution_cycles']:,}")
        print(f"Execution Time: {execution_result['total_execution_time_seconds']:.6f}s")
    
    print("\n" + "═" * 80)
    print("🏁 REPORT COMPLETE")
    print("═" * 80)

def print_simple_progress_bar(current, total, prefix="Progress", bar_length=50):
    """Print a simple progress bar"""
    percentage = (current / total) * 100 if total > 0 else 0
    filled = int(bar_length * current / total) if total > 0 else 0
    bar = "█" * filled + "░" * (bar_length - filled)
    print(f"\r{prefix}: [{bar}] {percentage:.1f}% ({current}/{total})", end="", flush=True)
    if current == total:
        print()  # New line when complete