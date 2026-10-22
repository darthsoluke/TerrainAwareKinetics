import numpy as np
import pytest

from scripts.analyze_mb_control_oscillation import signal_metrics


@pytest.mark.parametrize('dt', [.02, .001])
def test_spectral_window_resolves_gait_and_absolute_high_frequency_energy(dt):
    t = np.arange(round(4/dt))*dt
    values = (np.sin(2*np.pi*2*t)+.5*np.sin(2*np.pi*14*t))[:, None]
    report = signal_metrics(values, dt)
    assert report['spectral_resolution_hz'] == pytest.approx(.25)
    assert report['high_frequency_power'] == pytest.approx(.125, abs=.003)
    assert report['power_fraction_above_8hz'] == pytest.approx(.2, abs=.005)


def test_standing_still_cannot_fake_a_meaningful_high_frequency_fraction():
    report = signal_metrics(np.ones((200, 2)), .02)
    assert report['power_fraction_above_8hz'] == 0
