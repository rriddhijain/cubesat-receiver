class Config:
    """Configuration management for all phases."""
    
    def __init__(self):
        self.dataset_base_path = "cubesat_dataset"
        
        # Phase-specific configurations
        self.phase_configs = {
            1: {
                'dataset_path': f"{self.dataset_base_path}/phase1_timing",
                'samples_per_symbol': 4,
                'performance_threshold': 1e-2,  # BER ≤ 1×10⁻² @ 10 dB
                'impairments': ['timing_offset'],
                'modulation': 'BPSK'
            },
            2: {
                'dataset_path': f"{self.dataset_base_path}/phase2_snr",
                'samples_per_symbol': 4,
                'performance_threshold': 2.0,  # Within ±2 dB of theory
                'impairments': ['timing_offset', 'snr_scaling'],
                'modulation': 'BPSK'
            },
            3: {
                'dataset_path': f"{self.dataset_base_path}/phase3_coding",
                'samples_per_symbol': 4,
                'performance_threshold': {'RS': 1e-3, 'Conv': 1e-4},
                'impairments': ['timing_offset', 'snr_scaling', 'error_correction'],
                'modulation': 'BPSK',
                'coding': ['RS_15_11', 'Convolutional']
            },
            4: {
                'dataset_path': f"{self.dataset_base_path}/phase4_doppler",
                'samples_per_symbol': 4,
                'performance_threshold': 1e-3,  # BER ≤ 1×10⁻³ @ 15 dB
                'impairments': ['timing_offset', 'snr_scaling', 'error_correction', 'doppler'],
                'modulation': 'BPSK',
                'coding': ['RS_15_11', 'Convolutional']
            }
        }
    
    def get_phase_config(self, phase_num):
        """Get configuration for specific phase."""
        return self.phase_configs.get(phase_num, {})