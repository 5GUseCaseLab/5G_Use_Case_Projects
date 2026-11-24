import json
import pytest
import security
from controller import CoolingController
import config


def test_command_counter_strictly_increases_and_persists(tmp_path):
    path = str(tmp_path / "n.json")
    c = security.CommandCounter(path)
    a, b = c.next(), c.next()
    assert b == a + 1
    assert security.CommandCounter(path).next() == b + 1     # survives restart


def test_command_counter_corrupt_state_fails_loudly(tmp_path):
    p = tmp_path / "n.json"
    p.write_text("{garbage")
    with pytest.raises(RuntimeError):
        security.CommandCounter(str(p))


def test_command_is_signed_and_verifiable():
    env = security.make_command("uno_q_edge_01", 17, 150)
    p = security.verify_envelope(json.dumps(env))
    assert (p["nonce"], p["pwm"], p["command"]) == (17, 150, "set_cooling")


def test_controller_hysteresis():
    c = CoolingController()
    assert c.update(5.0) == (0, None)                 # inside band, off
    pwm, _ = c.update(6.6)
    assert pwm >= config.PWM_MIN_ON                   # turned on
    assert c.update(5.5)[0] > 0                       # between OFF and ON: stays on (hysteresis)
    assert c.update(4.4)[0] == 0                      # below COOL_OFF: off


def test_controller_alarms():
    c = CoolingController()
    pwm, alarm = c.update(9.0)
    assert alarm == "HIGH_TEMP" and pwm > 0
    assert c.update(1.5) == (0, "LOW_TEMP")


def test_pwm_clamped():
    c = CoolingController()
    assert c.update(30.0)[0] == 255
