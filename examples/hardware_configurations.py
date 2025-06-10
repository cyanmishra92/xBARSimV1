#!/usr/bin/env python3
"""
Hardware configuration examples for different scenarios
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.hierarchy import ReRAMChip, ChipConfig, SuperTileConfig, TileConfig
from core.crossbar import CrossbarConfig
from core.peripherals import ADCConfig, DACConfig, ADCType, DACType
from visualization.text_viz import print_architecture_diagram

def create_edge_device_config():
    """Ultra-low power edge device configuration"""
    print("🔋 Edge Device Configuration (Ultra-Low Power)")
    print("="*60)
    
    crossbar_config = CrossbarConfig(
        rows=32, cols=32,              # Small crossbars
        r_on=2e3, r_off=5e5,          # Low-power resistance states
        device_variability=0.05,       # Good process quality
        endurance=1e5                 # Moderate endurance
    )
    
    tile_config = TileConfig(
        crossbars_per_tile=2,          # Minimal crossbars
        crossbar_config=crossbar_config,
        local_buffer_size=16,          # 16 KB SRAM
        adc_sharing=True,              # Share ADCs to save power
        adcs_per_tile=8                # Reduced ADCs
    )
    
    supertile_config = SuperTileConfig(
        tiles_per_supertile=2,         # Small supertiles
        tile_config=tile_config,
        shared_buffer_size=64          # 64 KB eDRAM
    )
    
    chip_config = ChipConfig(
        supertiles_per_chip=1,         # Single supertile
        supertile_config=supertile_config,
        global_buffer_size=256         # 256 KB global
    )
    
    chip = ReRAMChip(chip_config)
    config = chip.get_chip_configuration()
    
    print(f"✓ Total Crossbars: {config['compute_capacity']['total_crossbars']}")
    print(f"✓ Weight Capacity: {config['compute_capacity']['total_weight_capacity']:,}")
    print(f"✓ Power Profile: Ultra-low power for mobile/IoT")
    print(f"✓ Use Case: Simple CNNs, keyword spotting, sensor fusion")
    
    return chip

def create_server_config():
    """High-performance server configuration"""
    print("\n🖥️  Server Configuration (High Performance)")
    print("="*60)
    
    crossbar_config = CrossbarConfig(
        rows=256, cols=256,            # Large crossbars
        r_on=500, r_off=2e6,          # Fast switching
        device_variability=0.02,       # Excellent process
        endurance=1e7                 # High endurance
    )
    
    tile_config = TileConfig(
        crossbars_per_tile=8,          # Many crossbars per tile
        crossbar_config=crossbar_config,
        local_buffer_size=512,         # 512 KB SRAM
        adc_sharing=False,             # Dedicated ADCs
        adcs_per_tile=256              # Full ADC complement
    )
    
    supertile_config = SuperTileConfig(
        tiles_per_supertile=8,         # Large supertiles
        tile_config=tile_config,
        shared_buffer_size=8192        # 8 MB eDRAM
    )
    
    chip_config = ChipConfig(
        supertiles_per_chip=4,         # Multiple supertiles
        supertile_config=supertile_config,
        global_buffer_size=65536       # 64 MB global
    )
    
    chip = ReRAMChip(chip_config)
    config = chip.get_chip_configuration()
    
    print(f"✓ Total Crossbars: {config['compute_capacity']['total_crossbars']}")
    print(f"✓ Weight Capacity: {config['compute_capacity']['total_weight_capacity']:,}")
    print(f"✓ Power Profile: High performance, datacenter power")
    print(f"✓ Use Case: Large CNNs, transformers, training acceleration")
    
    return chip

def create_automotive_config():
    """Automotive-grade reliable configuration"""
    print("\n🚗 Automotive Configuration (High Reliability)")
    print("="*60)
    
    crossbar_config = CrossbarConfig(
        rows=128, cols=128,            # Standard size
        r_on=1e3, r_off=1e6,          # Reliable states
        device_variability=0.03,       # Tight control
        endurance=5e6,                # High endurance
        retention_time=1e8             # Long retention (>3 years)
    )
    
    tile_config = TileConfig(
        crossbars_per_tile=4,          # Balanced design
        crossbar_config=crossbar_config,
        local_buffer_size=128,         # 128 KB SRAM
        adc_sharing=True,              # Moderate sharing
        adcs_per_tile=32               # Sufficient ADCs
    )
    
    supertile_config = SuperTileConfig(
        tiles_per_supertile=4,
        tile_config=tile_config,
        shared_buffer_size=2048        # 2 MB eDRAM
    )
    
    chip_config = ChipConfig(
        supertiles_per_chip=2,         # Dual supertiles
        supertile_config=supertile_config,
        global_buffer_size=8192        # 8 MB global
    )
    
    chip = ReRAMChip(chip_config)
    config = chip.get_chip_configuration()
    
    print(f"✓ Total Crossbars: {config['compute_capacity']['total_crossbars']}")
    print(f"✓ Weight Capacity: {config['compute_capacity']['total_weight_capacity']:,}")
    print(f"✓ Power Profile: Automotive-grade reliability")
    print(f"✓ Use Case: ADAS, autonomous driving, object detection")
    
    return chip

def create_research_config():
    """Research prototype with advanced features"""
    print("\n🔬 Research Configuration (Advanced Features)")
    print("="*60)
    
    crossbar_config = CrossbarConfig(
        rows=64, cols=64,              # Moderate size for experimentation
        r_on=800, r_off=1.5e6,        # Optimized states
        device_variability=0.15,       # Realistic variation
        endurance=2e6                 # Research-grade
    )
    
    tile_config = TileConfig(
        crossbars_per_tile=6,          # Experimental layout
        crossbar_config=crossbar_config,
        local_buffer_size=256,         # Large local buffers
        adc_sharing=True,
        adcs_per_tile=48               # Flexible ADC allocation
    )
    
    supertile_config = SuperTileConfig(
        tiles_per_supertile=3,         # Non-power-of-2 for research
        tile_config=tile_config,
        shared_buffer_size=4096        # 4 MB eDRAM
    )
    
    chip_config = ChipConfig(
        supertiles_per_chip=3,         # Experimental configuration
        supertile_config=supertile_config,
        global_buffer_size=16384       # 16 MB global
    )
    
    chip = ReRAMChip(chip_config)
    config = chip.get_chip_configuration()
    
    print(f"✓ Total Crossbars: {config['compute_capacity']['total_crossbars']}")
    print(f"✓ Weight Capacity: {config['compute_capacity']['total_weight_capacity']:,}")
    print(f"✓ Power Profile: Flexible for experiments")
    print(f"✓ Use Case: Algorithm research, novel architectures")
    
    return chip

def create_heterogeneous_config():
    """Heterogeneous configuration with mixed crossbar sizes"""
    print("\n🔀 Heterogeneous Configuration (Mixed Crossbar Sizes)")
    print("="*60)
    
    # This would require extending the current architecture
    # For now, we'll simulate with different tiles having different configurations
    
    # Small crossbar config for edge layers
    small_crossbar_config = CrossbarConfig(
        rows=64, cols=64,
        r_on=1.5e3, r_off=8e5,
        device_variability=0.08
    )
    
    # Large crossbar config for dense layers  
    large_crossbar_config = CrossbarConfig(
        rows=128, cols=128,
        r_on=1e3, r_off=1e6,
        device_variability=0.05
    )
    
    # Use the large config as default for this demo
    tile_config = TileConfig(
        crossbars_per_tile=4,
        crossbar_config=large_crossbar_config,
        local_buffer_size=128
    )
    
    supertile_config = SuperTileConfig(
        tiles_per_supertile=4,
        tile_config=tile_config,
        shared_buffer_size=1024
    )
    
    chip_config = ChipConfig(
        supertiles_per_chip=2,
        supertile_config=supertile_config,
        global_buffer_size=4096
    )
    
    chip = ReRAMChip(chip_config)
    config = chip.get_chip_configuration()
    
    print(f"✓ Total Crossbars: {config['compute_capacity']['total_crossbars']}")
    print(f"✓ Weight Capacity: {config['compute_capacity']['total_weight_capacity']:,}")
    print(f"✓ Power Profile: Optimized per layer type")
    print(f"✓ Use Case: Mixed workloads, adaptive computing")
    print("📝 Note: Future version will support true heterogeneous crossbars")
    
    return chip

def compare_configurations():
    """Compare different hardware configurations"""
    print("\n📊 CONFIGURATION COMPARISON")
    print("="*80)
    
    configs = [
        ("Edge Device", create_edge_device_config),
        ("Server", create_server_config),
        ("Automotive", create_automotive_config),
        ("Research", create_research_config),
        ("Heterogeneous", create_heterogeneous_config)
    ]
    
    print(f"{'Configuration':<15} {'Crossbars':<10} {'Capacity':<12} {'Memory':<10} {'Use Case':<20}")
    print("-" * 80)
    
    for name, config_func in configs:
        chip = config_func()
        config = chip.get_chip_configuration()
        
        crossbars = config['compute_capacity']['total_crossbars']
        capacity = f"{config['compute_capacity']['total_weight_capacity']//1000}K"
        memory = config['memory_hierarchy']['global_buffer'].split()[0] + "KB"
        
        if name == "Edge Device":
            use_case = "IoT, Mobile"
        elif name == "Server":
            use_case = "Datacenter, Training"
        elif name == "Automotive":
            use_case = "ADAS, Autonomous"
        elif name == "Research":
            use_case = "Experiments"
        else:
            use_case = "Mixed Workloads"
            
        print(f"{name:<15} {crossbars:<10} {capacity:<12} {memory:<10} {use_case:<20}")

def detailed_architecture_analysis():
    """Detailed analysis of a specific configuration"""
    print("\n🔍 DETAILED ARCHITECTURE ANALYSIS")
    print("="*80)
    
    print("Analyzing Server Configuration in detail...")
    chip = create_server_config()
    
    # Show detailed architecture
    print_architecture_diagram(chip)
    
    # Analysis
    config = chip.get_chip_configuration()
    
    print("\n📈 Performance Analysis:")
    total_crossbars = config['compute_capacity']['total_crossbars']
    crossbar_size = 256 * 256  # From server config
    total_ops_per_cycle = total_crossbars * crossbar_size
    
    print(f"✓ Theoretical Peak: {total_ops_per_cycle:,} ops/cycle")
    print(f"✓ At 1 GHz: {total_ops_per_cycle * 1e9 / 1e12:.1f} TOPS")
    
    print("\n💾 Memory Analysis:")
    hierarchy = config['memory_hierarchy']
    print(f"✓ L1 (Local): {hierarchy['local_buffers']} × {config['hierarchy']['tiles_per_supertile']} tiles × {config['hierarchy']['supertiles']} supertiles")
    print(f"✓ L2 (Shared): {hierarchy['shared_buffers']} × {config['hierarchy']['supertiles']} supertiles")
    print(f"✓ L3 (Global): {hierarchy['global_buffer']}")
    
    total_buffer_kb = (
        int(hierarchy['local_buffers'].split()[0]) * config['hierarchy']['tiles_per_supertile'] * config['hierarchy']['supertiles'] +
        int(hierarchy['shared_buffers'].split()[0]) * config['hierarchy']['supertiles'] +
        int(hierarchy['global_buffer'].split()[0])
    )
    print(f"✓ Total Buffer Memory: {total_buffer_kb:,} KB ({total_buffer_kb/1024:.1f} MB)")

if __name__ == "__main__":
    print("🏗️  ReRAM Crossbar Hardware Configuration Examples")
    print("="*80)
    
    # Create and analyze different configurations
    compare_configurations()
    
    # Detailed analysis
    detailed_architecture_analysis()
    
    print("\n✅ Hardware configuration examples completed!")
    print("💡 Use these configurations as templates for your own designs")