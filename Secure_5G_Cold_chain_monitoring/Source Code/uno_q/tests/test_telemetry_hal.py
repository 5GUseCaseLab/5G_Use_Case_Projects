import json
import config
config.SENSOR_SIMULATION = True  # tests default to the simulator
import hardware
import security
import telemetry
import pytest


def test_telemetry_seq_persists_and_signs(tmp_path):
    path = str(tmp_path / "seq.json")
    b = telemetry.TelemetryBuilder(path)
    e1, e2 = b.build(4.256), b.build(5.0)
    assert e2["payload"]["seq"] == e1["payload"]["seq"] + 1
    assert e1["payload"]["temperature"] == 4.26
    assert e1["sig"] == security.compute_sig(e1["payload"])
    assert telemetry.TelemetryBuilder(path).build(1.0)["payload"]["seq"] == e2["payload"]["seq"] + 1


def test_hal_validates_pwm():
    for bad in (-1, 256, 1.0, "9", True):
        with pytest.raises(ValueError):
            hardware.set_cooling_power(bad)


def test_sim_heats_then_cools():
    sim = hardware.ThermalSim()
    sim.pwm = 0
    t0 = sim.temp
    for _ in range(5):
        sim.step()
    assert sim.temp > t0
    sim.pwm = 255
    hot = sim.temp
    for _ in range(5):
        sim.step()
    assert sim.temp < hot


def test_real_sensor_failures_raise(monkeypatch):
    import sys, types
    monkeypatch.setattr(config, "SENSOR_SIMULATION", False)

    def fake_bridge(value=None, exc=None):
        mod = types.ModuleType("arduino.app_utils")
        class B:
            @staticmethod
            def call(name):
                if exc:
                    raise exc
                return value
        mod.Bridge = B
        monkeypatch.setitem(sys.modules, "arduino", types.ModuleType("arduino"))
        monkeypatch.setitem(sys.modules, "arduino.app_utils", mod)

    fake_bridge(21.6)
    assert hardware.read_temperature() == 21.6
    for bad in (-127.0, 85.0, 500.0, "abc", None):
        fake_bridge(bad)
        with pytest.raises(hardware.SensorError):
            hardware.read_temperature()
    fake_bridge(exc=TimeoutError("no answer"))
    with pytest.raises(hardware.SensorError):
        hardware.read_temperature()


def test_real_sensor_without_bridge_raises(monkeypatch):
    import sys
    monkeypatch.setattr(config, "SENSOR_SIMULATION", False)
    monkeypatch.setitem(sys.modules, "arduino", None)         # makes the import fail
    with pytest.raises(hardware.SensorError, match="Bridge unavailable"):
        hardware.read_temperature()


def _install_bridge(monkeypatch, handler):
    import sys, types
    mod = types.ModuleType("arduino.app_utils")
    calls = []
    class B:
        @staticmethod
        def call(name, *args):
            calls.append((name, args))
            return handler(name, *args)
    mod.Bridge = B
    monkeypatch.setitem(sys.modules, "arduino", types.ModuleType("arduino"))
    monkeypatch.setitem(sys.modules, "arduino.app_utils", mod)
    return calls


def test_real_cooling_calls_bridge(monkeypatch):
    monkeypatch.setattr(config, "COOLING_SIMULATION", False)
    calls = _install_bridge(monkeypatch, lambda name, v: v)
    hardware.set_cooling_power(150)
    assert calls == [("set_cooling_pwm", (150,))]


def test_real_cooling_failure_forces_off_and_raises(monkeypatch):
    monkeypatch.setattr(config, "COOLING_SIMULATION", False)
    def boom(name, v):
        if v != 0:
            raise TimeoutError("no answer")
        return 0
    calls = _install_bridge(monkeypatch, boom)
    with pytest.raises(hardware.CoolingError):
        hardware.set_cooling_power(200)
    assert calls[-1] == ("set_cooling_pwm", (0,))          # tried to switch it off


def test_real_cooling_sketch_rejection_raises(monkeypatch):
    monkeypatch.setattr(config, "COOLING_SIMULATION", False)
    _install_bridge(monkeypatch, lambda name, v: -1)
    with pytest.raises(hardware.CoolingError, match="rejected"):
        hardware.set_cooling_power(100)


def test_invalid_pwm_never_reaches_bridge(monkeypatch):
    monkeypatch.setattr(config, "COOLING_SIMULATION", False)
    calls = _install_bridge(monkeypatch, lambda name, v: v)
    for bad in (-1, 256, 1.5, "9", True, None):
        with pytest.raises(ValueError):
            hardware.set_cooling_power(bad)
    assert calls == []
