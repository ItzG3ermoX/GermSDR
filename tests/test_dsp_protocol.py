import unittest

import numpy as np

from backend.config import Settings
from backend.dsp import (
    DemodState,
    _decimate_complex_stateful,
    _zoom_ratio,
    compute_waterfall,
    demodulate,
    demodulate_wbfm,
)
from backend.protocol import HEADER_SIZE, make_waterfall_frame, parse_waterfall_header
from backend.state import RadioConfig, RadioState


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

    def test_zoom_ratio_tracks_requested_zoom(self):
        self.assertEqual(_zoom_ratio(1.0), (1, 1))
        for zoom in (1.01, 1.25, 3.7, 16.0, 64.0):
            up, down = _zoom_ratio(zoom)
            self.assertAlmostEqual(down / up, zoom, delta=0.01)

    def test_channelizer_decimation_is_continuous_across_blocks(self):
        rng = np.random.default_rng(42)
        iq = (rng.standard_normal(131_072) + 1j * rng.standard_normal(131_072)).astype(
            np.complex64
        )

        whole = _decimate_complex_stateful(iq, 50, state=DemodState())
        chunked_state = DemodState()
        chunked = np.concatenate(
            (
                _decimate_complex_stateful(iq[:65_536], 50, state=chunked_state),
                _decimate_complex_stateful(iq[65_536:], 50, state=chunked_state),
            )
        )

        self.assertEqual(chunked.shape, whole.shape)
        self.assertTrue(np.allclose(chunked, whole, rtol=1e-5, atol=1e-5))

    def test_invalid_settings_and_tuning_frequency_are_rejected(self):
        with self.assertRaises(ValueError):
            Settings(ring_slots=0)
        with self.assertRaises(ValueError):
            RadioState(
                RadioConfig(center_freq=1, sample_rate=2_400_000, gain="auto")
            )

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


    def test_freq_offset_recovers_offset_fm_signal(self):
        # An FM signal sitting at +200 kHz from band centre should demodulate
        # to roughly the same audio whether we capture it centred (offset 0)
        # or off-centre and shift it down with freq_offset_hz.
        sample_rate = 240_000
        n = np.arange(sample_rate // 5)
        tone = np.sin(2 * np.pi * 1_000 * n / sample_rate)
        phase = np.cumsum(2 * np.pi * 25_000 * tone / sample_rate)
        baseband = np.exp(1j * phase).astype(np.complex64)

        offset_hz = 60_000
        shift = np.exp(2j * np.pi * offset_hz * n / sample_rate).astype(np.complex64)
        offset_signal = (baseband * shift).astype(np.complex64)

        centred = demodulate(baseband, "wbfm", sample_rate=sample_rate, audio_rate=48_000)
        shifted = demodulate(
            offset_signal, "wbfm", sample_rate=sample_rate, audio_rate=48_000,
            freq_offset_hz=offset_hz,
        )

        self.assertEqual(shifted.dtype, np.float32)
        self.assertGreater(shifted.size, 0)
        self.assertLessEqual(float(np.max(np.abs(shifted))), 1.0001)
        # Both paths should carry comparable audio energy (the offset one isn't
        # silence / noise). Compare RMS within a generous factor.
        rms_c = float(np.sqrt(np.mean(centred[200:] ** 2))) + 1e-9
        rms_s = float(np.sqrt(np.mean(shifted[200:] ** 2))) + 1e-9
        self.assertGreater(rms_s, 0.2 * rms_c)

    def test_tune_offset_stays_in_band(self):
        state = RadioState(
            RadioConfig(center_freq=100_000_000, sample_rate=2_400_000, gain="auto")
        )
        # In-band offset is kept; the band centre (config) does not move.
        state.set_tune_offset(300_000)
        self.assertEqual(state.tune_offset, 300_000)
        self.assertEqual(state.tuned_freq, 100_300_000)
        self.assertEqual(state.snapshot().center_freq, 100_000_000)

        # Out-of-range offset is clamped to +/- sample_rate/2.
        state.set_tune_offset(9_000_000)
        self.assertEqual(state.tune_offset, 1_200_000)

        # A hardware retune resets the offset to 0.
        state.update(center_freq=90_000_000)
        self.assertEqual(state.tune_offset, 0.0)
        self.assertEqual(state.tuned_freq, 90_000_000)

    def test_pan_is_independent_of_offset(self):
        # The zoom window (pan) must NOT recentre on the tuned offset while the
        # tuned point is still inside the visible window -- clicking inside the
        # view only moves the listening point, not the view.
        state = RadioState(
            RadioConfig(center_freq=100_000_000, sample_rate=2_400_000, gain="auto")
        )
        state.set_view(zoom=4.0)  # half-span = 0.125 -> window [0.375, 0.625]
        _zoom, pan = state.view
        self.assertEqual(_zoom, 4.0)
        self.assertAlmostEqual(pan, 0.5)

        # Small offset: tuned point (pan 0.55) is inside the window -> pan unchanged.
        state.set_tune_offset(120_000)  # 0.05 of the band -> tuned_pan 0.55
        _zoom, pan = state.view
        self.assertAlmostEqual(pan, 0.5, places=3)

    def test_pan_hz_within_band_when_zoomed(self):
        # Drag-to-pan sends the desired view centre as an absolute frequency.
        # While zoomed in there is headroom to pan WITHIN the captured band, so
        # the backend just moves the view fraction (no hardware retune).
        state = RadioState(
            RadioConfig(center_freq=100_000_000, sample_rate=2_400_000, gain="auto")
        )
        # zoom=4 keeps the 2.4 MHz hardware rate (digital zoom only) so the band
        # is still 2.4 MHz wide; half-span 0.125 -> pan headroom [0.125, 0.875].
        state.set_view(zoom=4.0)
        self.assertEqual(state.snapshot().sample_rate, 2_400_000)
        # Centre of band -> pan 0.5, band unchanged.
        state.set_view(pan_hz=100_000_000)
        self.assertAlmostEqual(state.view[1], 0.5, places=6)
        self.assertEqual(state.snapshot().center_freq, 100_000_000)
        # +600 kHz from a 2.4 MHz band centre -> +0.25 -> pan 0.75, still in band.
        state.set_view(pan_hz=100_600_000)
        self.assertAlmostEqual(state.view[1], 0.75, places=6)
        self.assertEqual(state.snapshot().center_freq, 100_000_000)

    def test_pan_hz_at_1x_scrolls_the_band(self):
        # At 1x the whole band is already on screen, so there is no room to pan
        # within it -- a drag must RETUNE the hardware centre to scroll the band.
        state = RadioState(
            RadioConfig(center_freq=100_000_000, sample_rate=2_400_000, gain="auto")
        )
        state.set_view(pan_hz=100_600_000)
        # Band centre followed the drag; view re-centres on the new band.
        self.assertEqual(state.snapshot().center_freq, 100_600_000)
        self.assertAlmostEqual(state.view[1], 0.5, places=6)
        self.assertEqual(state.tune_offset, 0.0)

    def test_pan_hz_past_edge_when_zoomed_clamps_without_retune(self):
        # Regression for the "+/-25 MHz jump": while zoomed in, a drag that runs
        # past the band edge must CLAMP the view to the edge, never retune the
        # hardware centre. Previously a frac just outside the pannable region fell
        # through to the retune branch and teleported the centre.
        state = RadioState(
            RadioConfig(center_freq=100_000_000, sample_rate=2_400_000, gain="auto")
        )
        state.set_view(zoom=4.0)  # 2.4 MHz band, half-span 0.125
        self.assertEqual(state.snapshot().sample_rate, 2_400_000)
        # Way past the right edge: pan_hz well beyond the band. Must clamp the
        # view fraction to (1 - half_span) and leave the hardware centre put.
        state.set_view(pan_hz=200_000_000)
        self.assertEqual(state.snapshot().center_freq, 100_000_000)
        self.assertAlmostEqual(state.view[1], 1.0 - 0.125, places=6)
        # ...and past the left edge clamps to half_span, still no retune.
        state.set_view(pan_hz=10_000_000)
        self.assertEqual(state.snapshot().center_freq, 100_000_000)
        self.assertAlmostEqual(state.view[1], 0.125, places=6)

    def test_pan_hz_at_1x_step_is_capped_to_one_band(self):
        # At 1x a retune follows the drag, but a single pan can move the centre
        # at most one captured-band width -- a stray huge pan_hz cannot teleport.
        state = RadioState(
            RadioConfig(center_freq=100_000_000, sample_rate=2_400_000, gain="auto")
        )
        state.set_view(pan_hz=500_000_000)  # absurd jump request
        moved = abs(state.snapshot().center_freq - 100_000_000)
        self.assertLessEqual(moved, 2_400_000)

    def test_pan_hz_clamps_to_tuning_range(self):
        # Start near the low end of the tuner's range so a single band-width step
        # toward an out-of-range pan_hz lands on the clamp, never below it.
        state = RadioState(
            RadioConfig(
                center_freq=RadioState.TUNE_MIN_HZ + 1_000_000,
                sample_rate=2_400_000,
                gain="auto",
            )
        )
        # Drag toward DC (below the tuner's range): the retune clamps to the
        # tuning floor and never crashes or goes negative.
        state.set_view(pan_hz=0)
        self.assertGreaterEqual(state.snapshot().center_freq, RadioState.TUNE_MIN_HZ)
        self.assertLessEqual(state.snapshot().center_freq, RadioState.TUNE_MAX_HZ)

    def test_pan_edge_follows_when_tuned_leaves_window(self):
        # When the tuned point leaves the visible window, the window follows so
        # the signal stays reachable (KiwiSDR-style edge-follow).
        state = RadioState(
            RadioConfig(center_freq=100_000_000, sample_rate=2_400_000, gain="auto")
        )
        state.set_view(zoom=4.0)  # window [0.375, 0.625]
        state.set_tune_offset(600_000)  # tuned_pan 0.75, outside the window
        _zoom, pan = state.view
        # Window slid right to keep 0.75 just inside (0.75 - 0.9*0.125).
        self.assertAlmostEqual(pan, 0.6375, places=3)

    def test_hw_rate_ladder_steps_down_with_zoom(self):
        # Deeper perceived zoom must select a narrower native hardware rate, and
        # the residual digital zoom must stay within the modest cap (so the DSP
        # never decimates a huge band -> no lag).
        from backend.state import (
            hw_rate_for_zoom, digital_zoom_for, HW_RATE_LADDER, DIGITAL_ZOOM_MAX,
        )
        rates = [hw_rate_for_zoom(z) for z in (1, 4, 8, 16, 32, 64)]
        # Monotonically non-increasing as zoom grows.
        self.assertEqual(rates, sorted(rates, reverse=True))
        self.assertEqual(hw_rate_for_zoom(1), HW_RATE_LADDER[0])
        self.assertEqual(hw_rate_for_zoom(64), HW_RATE_LADDER[-1])
        for z in (1, 2, 4, 5, 8, 16, 32, 64):
            r = hw_rate_for_zoom(z)
            self.assertGreaterEqual(digital_zoom_for(z, r), 1.0)
            # 64x on the narrowest rate is the only case allowed past the cap.
            if z < 48:
                self.assertLessEqual(digital_zoom_for(z, r), DIGITAL_ZOOM_MAX + 1e-6)

    def test_zoom_in_retunes_hardware_and_keeps_span_continuous(self):
        # Zooming in past a threshold retunes the tuner to a narrower rate centred
        # on the listened-to signal, resets the offset, and the visible span
        # (BASE_RATE/zoom) stays continuous (no jump) across the switch.
        from backend.state import BASE_RATE
        state = RadioState(
            RadioConfig(center_freq=100_000_000, sample_rate=2_400_000, gain="auto")
        )
        # Listen slightly off-centre, then zoom deep.
        state.set_tune_offset(80_000)  # tuned = 100.08 MHz
        tuned_before = state.tuned_freq
        state.set_view(zoom=16.0)
        cfg = state.snapshot()
        # Hardware dropped to a narrow native rate, centred on the tuned signal.
        self.assertLess(cfg.sample_rate, 2_400_000)
        self.assertEqual(cfg.center_freq, round(tuned_before))
        self.assertEqual(state.tune_offset, 0.0)
        # Visible span matches the perceived zoom relative to the base rate.
        digital_zoom, _pan = state.view
        visible_span = cfg.sample_rate / digital_zoom
        self.assertAlmostEqual(visible_span, BASE_RATE / 16.0, delta=1.0)

    def test_zoom_back_out_restores_base_rate(self):
        state = RadioState(
            RadioConfig(center_freq=100_000_000, sample_rate=2_400_000, gain="auto")
        )
        state.set_view(zoom=32.0)
        self.assertLess(state.snapshot().sample_rate, 2_400_000)
        state.set_view(zoom=1.0)
        self.assertEqual(state.snapshot().sample_rate, 2_400_000)
        digital_zoom, _pan = state.view
        self.assertAlmostEqual(digital_zoom, 1.0)

    def test_hi_res_zoom_returns_sharp_finite_bins(self):
        # A tone offset within the band should zoom to a finite, peaked spectrum.
        from backend.dsp import compute_waterfall_zoom
        sample_rate = 2_400_000
        n = np.arange(65536)
        iq = np.exp(2j * np.pi * 200_000 * n / sample_rate).astype(np.complex64)
        bins, center, rate = compute_waterfall_zoom(
            iq, 4096, sample_rate, 100_000_000, zoom=8.0, pan=0.5 + 200_000 / sample_rate
        )
        self.assertEqual(bins.shape, (4096,))
        self.assertTrue(np.isfinite(bins).all())
        self.assertLess(rate, sample_rate)  # narrower span = sharper

    def test_zoom_accumulates_to_a_large_fft(self):
        # Feeding several blocks at high zoom should let the accumulator build a
        # large FFT (finer resolution) and keep returning finite display bins.
        from backend.dsp import DemodState, compute_waterfall_zoom, _zoom_target_nfft
        ds = DemodState()

        # The target FFT grows with zoom (finer Hz/bin the deeper you zoom) and
        # never drops below the display width (so it's at least as sharp as 1x).
        self.assertGreaterEqual(_zoom_target_nfft(2.0, 4096), 4096)
        self.assertGreater(
            _zoom_target_nfft(32.0, 4096),
            _zoom_target_nfft(4.0, 4096),
        )
        self.assertGreaterEqual(_zoom_target_nfft(64.0, 4096), 65536)

        sample_rate = 2_400_000
        out = None
        for b in range(8):
            n = np.arange(b * 65536, (b + 1) * 65536)
            iq = np.exp(2j * np.pi * 150_000 * n / sample_rate).astype(np.complex64)
            out, _c, _r = compute_waterfall_zoom(
                iq, 4096, sample_rate, 100_000_000,
                zoom=16.0, pan=0.5 + 150_000 / sample_rate,
                state=ds,
            )
        self.assertEqual(out.shape, (4096,))
        self.assertTrue(np.isfinite(out).all())


    def test_wbfm_isolates_offset_channel_in_wide_band(self):
        # Regression for "audio is weird until you zoom in" + "clicking a few kHz
        # over still demods the old station". Two FM stations sit in a wide 2.4
        # MHz capture: a 1 kHz-tone station at -400 kHz and a 600 Hz-tone station
        # at +400 kHz. Tuning (freq_offset) to the +400 kHz station must recover
        # ITS audio, not a mush of both -- which only works if the demod
        # channelizes to a narrow IF before the discriminator.
        from backend.dsp import DemodState, _reset_audio_dsp_state
        ds = DemodState()
        sample_rate = 2_400_000
        n = np.arange(sample_rate // 4)

        def fm_station(audio_hz, dev, carrier):
            phase = np.cumsum(2 * np.pi * dev * np.sin(2 * np.pi * audio_hz * n / sample_rate) / sample_rate)
            base = np.exp(1j * phase)
            shift = np.exp(2j * np.pi * carrier * n / sample_rate)
            return (base * shift).astype(np.complex64)

        wanted = fm_station(600.0, 75_000, +400_000)
        other = fm_station(1_000.0, 75_000, -400_000)
        iq = (wanted + other).astype(np.complex64)

        _reset_audio_dsp_state(ds)
        pcm = demodulate(
            iq, "wbfm", sample_rate=sample_rate, audio_rate=48_000,
            freq_offset_hz=+400_000, state=ds,
        )
        self.assertGreater(pcm.size, 0)
        self.assertLessEqual(float(np.max(np.abs(pcm))), 1.0001)

        # The recovered audio should be dominated by the wanted station's 600 Hz
        # tone, NOT the other station's 1 kHz tone. Compare spectral energy in a
        # band around each tone.
        tail = pcm[480:]  # skip filter warm-up
        spec = np.abs(np.fft.rfft(tail * np.hanning(tail.size)))
        freqs = np.fft.rfftfreq(tail.size, 1.0 / 48_000)

        def band_energy(f0):
            m = (freqs > f0 - 80) & (freqs < f0 + 80)
            return float(np.sum(spec[m] ** 2))

        e_wanted = band_energy(600.0)
        e_other = band_energy(1_000.0)
        # Wanted tone clearly dominates the rejected station's tone.
        self.assertGreater(e_wanted, 5.0 * e_other)


if __name__ == "__main__":
    unittest.main()
