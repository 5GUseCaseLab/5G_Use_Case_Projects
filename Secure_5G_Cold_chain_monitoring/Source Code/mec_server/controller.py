"""Cooling decision logic with hysteresis. No MQTT, no GPIO: pure logic."""
import config


class CoolingController:
    def __init__(self):
        self.cooling = False

    def update(self, temp: float):
        """Return (pwm, alarm). alarm is None, 'LOW_TEMP' or 'HIGH_TEMP'."""
        alarm = None
        if temp < config.SAFE_LOW_C:
            alarm = "LOW_TEMP"
            self.cooling = False                      # never cool an already-cold box
        elif temp > config.SAFE_HIGH_C:
            alarm = "HIGH_TEMP"
            self.cooling = True
        elif not self.cooling and temp >= config.COOL_ON_C:
            self.cooling = True
        elif self.cooling and temp <= config.COOL_OFF_C:
            self.cooling = False

        if not self.cooling:
            return 0, alarm
        pwm = config.PWM_MIN_ON + (temp - config.COOL_ON_C) * config.PWM_GAIN_PER_C
        pwm = int(max(config.PWM_MIN_ON, min(config.PWM_MAX, round(pwm))))
        return pwm, alarm
