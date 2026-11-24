# Troubleshooting

[← Back to README](../README.md)

Problems we actually ran into while building this, and what fixed them.

## ESP32 won't join Wi-Fi (`Connecting.....` forever)

The ESP32 sketch prints the reason for each failed attempt:

| Printed reason | Fix |
|---|---|
| `hotspot NOT FOUND` (201) | the ESP32 only sees **2.4 GHz**; switch the hotspot band, check the name, move closer |
| `WRONG PASSWORD` (15, 204) | re-type `WIFI_PASS` |
| `AUTH FAILED` (202, 2) | wrong password, or the network is WPA3-only |
| `SECURITY MISMATCH` (210, 211) | set the hotspot to **WPA2** |

The sketch also lowers the transmit power (`WiFi.setTxPower(WIFI_POWER_8_5dBm)`), which fixes connection failures on many cheap ESP32 boards, and retries from scratch every 10 s.

## Slave always shows `link=down`

The ESP32 isn't receiving packets. Check the **UNO Q console** to see why:

| UNO Q console shows | Cause | Fix |
|---|---|---|
| `waiting for pot data from the MCU sketch...` | the sketch didn't compile or upload, so Python never gets pot values | check the App Lab sketch output; v2 `main.py` also works with the v1 sketch |
| `REPORT` blocks with `echoed 0` | packets leave but don't arrive | `ESP32_IP` is wrong (phone hotspots often change it after a reboot): press EN, read the new IP. Check both boards are on the same network and the hotspot doesn't isolate clients |
| `send failed: ...` | the UNO Q lost Wi-Fi | reconnect it to the same network |
| `Bridge ping turned off` | old sketch on the MCU | upload the v2 sketch; everything else still works |

## Upload fails: `Failed to connect to ESP32: No serial data received`

1. Make sure the port really is the ESP32: unplug it and see which COM port disappears (**Tools → Get Board Info** shows VID `0x10C4` / PID `0xEA60` for the CP2102 USB chip).
2. Close the Serial Monitor before uploading.
3. Put the board in download mode: hold **BOOT**, tap **EN**, release **BOOT**, then upload. Or hold **BOOT** while the IDE prints `Connecting....`.
4. **Tools → Upload Speed → 115200**.
5. Disconnect everything except USB; a wire on GPIO 0/2/12/15 can block download mode.
6. Try another data-capable USB cable and port.

If the Serial Monitor shows **nothing at all** after pressing EN, even in download mode, the ESP32 chip itself is not running. Measure 3V3 to GND: about 0 V means the regulator has failed; 3.3 V with no output means the chip is damaged. Replace the board, and fix the servo power wiring first (see [WIRING.md](WIRING.md)).

## Servos twitch, buzz or the ESP32 resets

- Servos on their own 5–6 V supply, never the ESP32's pins.
- Supply (–) joined to ESP32 GND.
- 470–1000 µF capacitor across the servo rails.
- Shoulder sags or buzzes at full extension: try 6 V, shorter links, or an MG996R.

## Arm behaves oddly

| Symptom | Fix |
|---|---|
| A joint moves the wrong way | swap the two angles on its `CAL` line |
| Slave offset from master | fit the servo horns with the link running and the master in a neutral pose |
| A joint hits the frame | tighten its `LIMITS` in **both** `main.py` and the ESP32 sketch |
| Slave too jittery | lower `ALPHA` (smoother, more lag) |
| Slave too sluggish | raise `ALPHA` or `MAX_STEP` |
| Pot value stuck at 0 or 4095, or jumps | check that pot's wiring; 3V3 not 5V |
