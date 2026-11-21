"""Hardware abstraction layer (HAL).

The rest of the app only ever calls two functions:
  read_temperature()        -> float, deg C   (raises SensorError on any failure)
  set_cooling_power(pwm)    -> None, pwm 0..255

Sensor: DS18B20 on the MCU side. The sketch exposes `get_temperature` through the
Arduino Bridge and Python calls it. The Bridge only exists when the app is started
through Arduino App Lab (or `arduino-app-cli app start`), not from a plain SSH shell.

Cooling: the sketch exposes `set_cooling_pwm(int)`; Python calls it through the Bridge.
While config.COOLING_SIMULATION is True the output is only simulated.
"""
import random

import config


class SensorError(RuntimeError):
    """The temperature sensor failed. We never substitute an invented value."""


class CoolingError(RuntimeError):
    """The cooling output could not be set as requested."""


# =====================================================================
# Temperature simulator (used while config.SENSOR_SIMULATION is True)
# =====================================================================
class ThermalSim:
    def __init__(self):
        self.temp = config.SIM_START_C
        self.pwm = 0
        self.tick = 0

    def step(self) -> float:
        self.tick += 1
        self.temp += config.SIM_HEAT_RATE + random.uniform(-0.05, 0.05)
        self.temp -= (self.pwm / 255.0) * config.SIM_COOL_RATE
        if config.SIM_DOOR_OPEN_EVERY and self.tick % config.SIM_DOOR_OPEN_EVERY == 0:
            self.temp += config.SIM_DOOR_SPIKE_C
        self.temp = max(-10.0, min(30.0, self.temp))
        return round(self.temp, 2)


_sim = ThermalSim()


# =====================================================================
# Real sensor (DS18B20 through the Arduino Bridge)
# =====================================================================
def _read_real_temperature() -> float:
    try:
        from arduino.app_utils import Bridge        # only available inside App Lab's environment
    except ImportError as e:
        raise SensorError("Bridge unavailable: start the app through App Lab / arduino-app-cli") from e
    try:
        t = float(Bridge.call("get_temperature"))
    except Exception as e:
        raise SensorError(f"Bridge call failed: {e}") from e
    if t == -127.0:                                  # DallasTemperature 'disconnected' value
        raise SensorError("DS18B20 disconnected (-127): check wiring and the 4.7k pull-up")
    if t == 85.0:                                    # chip power-on default, not a measurement
        raise SensorError("DS18B20 returned 85.0 (power-on value); reading not valid")
    if not -55.0 <= t <= 125.0:
        raise SensorError(f"Reading {t} is outside the sensor's range")
    return round(t, 2)


# =====================================================================
# HAL: the ONLY functions the rest of the app uses
# =====================================================================
def read_temperature() -> float:
    if config.SENSOR_SIMULATION:
        return _sim.step()
    return _read_real_temperature()


def _apply_real_cooling(pwm: int) -> int:
    try:
        from arduino.app_utils import Bridge
    except ImportError as e:
        raise CoolingError("Bridge unavailable: start the app through App Lab / arduino-app-cli") from e
    try:
        applied = Bridge.call("set_cooling_pwm", pwm)
    except Exception as e:
        try:                                         # best effort: force the output off
            Bridge.call("set_cooling_pwm", 0)
        except Exception:
            pass
        raise CoolingError(f"Bridge call failed: {e}") from e
    if isinstance(applied, bool) or not isinstance(applied, int) or applied < 0:
        raise CoolingError(f"Sketch rejected PWM {pwm} (returned {applied!r})")
    return applied                                   # value after the sketch's own safety cap


def set_cooling_power(pwm_value) -> None:
    if isinstance(pwm_value, bool) or not isinstance(pwm_value, int) or not 0 <= pwm_value <= 255:
        raise ValueError("PWM must be an integer from 0 to 255")
    if config.COOLING_SIMULATION:
        _sim.pwm = pwm_value
        print(f"[SIMULATION] Cooling PWM = {pwm_value}/255")
        return
    applied = _apply_real_cooling(pwm_value)
    if applied != pwm_value:
        print(f"[COOLING] requested {pwm_value}, sketch applied {applied} (safety cap)")


if __name__ == "__main__":
    # Quick check: python hardware.py  (prints 3 readings)
    import time
    for _ in range(3):
        print("temperature:", read_temperature())
        time.sleep(1)
