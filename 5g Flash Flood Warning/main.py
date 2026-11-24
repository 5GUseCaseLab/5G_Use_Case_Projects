import os

os.environ["URLLC_SERVER_HOST"] = "192.168.31.121"
os.environ["URLLC_UDP_PORT"] = "9000"

from board_sender import main

main()
