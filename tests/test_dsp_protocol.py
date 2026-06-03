import unittest

import numpy as np

from backend.dsp import compute_waterfall, demodulate_wbfm
from backend.protocol import HEADER_SIZE, make_waterfall_frame, parse_waterfall_header
from backend.state import RadioConfig


class DspProtocolTests(unittest.TestCase):
    def test_waterfall_frame_has_aligned_payload(self):
        config = RadioConfig(center_freq=100_800_000, sample_rate=2_400_000, gain="auto")
        bins = np.linspace(-120, -20, 1024, dtype=np.float32)

        frame = make_waterfall_frame(42, config, bins)
        header = parse_waterfall_header(frame)
        payload = np.frombuffer(frame, dtype="<f4", offset=HEADER_SIZE)

        self.assertEqual(HEADER_SIZE, 20)
        self.assertEqual(HEADER_SIZE % 4, 0)
        self.assertEqual(header.seq, 42)
        self.assertEqual(header.fft_size, 1024)
        self.assertTrue(np.allclose(payload, bins))

    def test_compute_waterfall_returns_finite_bins(self):
        n = np.arange(8192)
        iq = np.exp(2j * np.pi * 200_000 * n / 2_400_000).astype(np.complex64)

        bins = compute_waterfall(iq, 4096)

        self.assertEqual(bins.dtype, np.float32)
        self.assertEqual(bins.shape, (4096,))
        self.assertTrue(np.isfinite(bins).all())

    def test_wbfm_demodulator_returns_pcm(self):
        sample_rate = 240_000
        n = np.arange(sample_rate // 10)
        tone = np.sin(2 * np.pi * 1_000 * n / sample_rate)
        phase = np.cumsum(2 * np.pi * 25_000 * tone / sample_rate)
        iq = np.exp(1j * phase).astype(np.complex64)

        pcm = demodulate_wbfm(iq, sample_rate=sample_rate, audio_rate=48_000)

        self.assertEqual(pcm.dtype, np.float32)
        self.assertGreater(pcm.size, 0)
        self.assertLessEqual(float(np.max(np.abs(pcm))), 1.0001)


if __name__ == "__main__":
    unittest.main()

