import unittest
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from receiver.demodulator import BPSKDemodulator
from receiver.utils import generate_test_signal
from evaluation.metrics import calculate_ber, theoretical_ber_bpsk

class TestBasicFunctionality(unittest.TestCase):
    
    def setUp(self):
        self.demodulator = BPSKDemodulator(samples_per_symbol=4, phase=1)
    
    def test_demodulator_initialization(self):
        """Test demodulator initializes correctly."""
        self.assertEqual(self.demodulator.samples_per_symbol, 4)
        self.assertEqual(self.demodulator.phase, 1)
        self.assertIsNotNone(self.demodulator.matched_filter)
    
    def test_ber_calculation(self):
        """Test BER calculation."""
        tx_bits = np.array([0, 1, 0, 1, 0])
        rx_bits = np.array([0, 0, 0, 1, 1])  # 2 errors out of 5
        
        ber = calculate_ber(tx_bits, rx_bits)
        self.assertAlmostEqual(ber, 0.4)
    
    def test_test_signal_generation(self):
        """Test test signal generation."""
        test_data = generate_test_signal(num_bits=100, samples_per_symbol=4, snr_db=10)
        
        self.assertIn('rx_samples', test_data)
        self.assertIn('ground_truth_bits', test_data)
        self.assertEqual(len(test_data['ground_truth_bits']), 100)
        self.assertEqual(len(test_data['rx_samples']), 400)  # 100 bits * 4 samples/symbol
    
    def test_theoretical_ber(self):
        """Test theoretical BER calculation."""
        ber = theoretical_ber_bpsk(10.0)  # 10 dB SNR
        
        # Should be around 1e-6 for 10 dB
        self.assertLess(ber, 1e-5)
        self.assertGreater(ber, 1e-7)

if __name__ == '__main__':
    unittest.main()