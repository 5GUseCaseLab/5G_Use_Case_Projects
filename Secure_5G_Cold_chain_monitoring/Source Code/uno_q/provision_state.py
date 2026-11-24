"""Trusted (manual) creation or reset of the persistent nonce file.

  python provision_state.py                 # create file with last_nonce=0 (refuses to overwrite)
  python provision_state.py --force         # overwrite
  python provision_state.py --force --nonce 5000
"""
import argparse
import os
import sys

import config
import security

ap = argparse.ArgumentParser()
ap.add_argument("--nonce", type=int, default=0)
ap.add_argument("--force", action="store_true")
a = ap.parse_args()

if os.path.exists(config.NONCE_FILE) and not a.force:
    sys.exit(f"{config.NONCE_FILE} already exists. Use --force to overwrite deliberately.")
security.provision_state(config.NONCE_FILE, config.DEVICE_ID, a.nonce)
print(f"Provisioned {config.NONCE_FILE} with last_nonce={a.nonce}")
