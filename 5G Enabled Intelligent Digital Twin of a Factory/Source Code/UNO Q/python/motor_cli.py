from arduino.app_utils import Bridge

print("Motor Control CLI Ready.")
print("Commands: ON | OFF | PWM:150 | PWM:-200 | EXIT")

while True:
    try:
        cmd = input("> ").strip().upper()
        if not cmd:
            continue

        if cmd == "ON":
            Bridge.call("set_motor_on", 1)
            print("Motor ON")

        elif cmd in ("OFF", "STOP"):
            Bridge.call("set_motor_on", 0)
            print("Motor OFF")

        elif cmd.startswith("PWM:"):
            pwm_val = int(cmd[4:])
            pwm_val = max(-255, min(255, pwm_val))
            Bridge.call("set_motor_pwm", pwm_val & 0xFFFFFFFF)
            print(f"PWM set to {pwm_val}")

        elif cmd in ("EXIT", "QUIT"):
            Bridge.call("set_motor_on", 0)
            print("Motor stopped. Exiting.")
            break

        else:
            print("Use: ON, OFF, PWM:<value> (-255 to 255), or EXIT")

    except ValueError:
        print("Invalid number. Example: PWM:150 or PWM:-180")
    except TimeoutError as e:
        print(f"Bridge Timeout: {e}")
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")
        break