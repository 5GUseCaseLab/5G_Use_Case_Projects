"""Standalone UNO Q router client for the board's system Python."""

import socket
import threading

import msgpack


ROUTER_SOCKET = "/var/run/arduino-router.sock"


class RouterBridge:
    def __init__(self, path=ROUTER_SOCKET):
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.connect(path)
        self.send_lock = threading.Lock()
        self.pending_lock = threading.Lock()
        self.pending = {}
        self.handlers = {}
        self.next_id = 0
        self.closed = False
        self.reader = threading.Thread(target=self._receive, daemon=True)
        self.reader.start()

    def _send(self, message):
        data = msgpack.packb(message, use_bin_type=True)
        with self.send_lock:
            self.socket.sendall(data)

    def call(self, method, *args, timeout=10):
        if self.closed:
            raise ConnectionError("Arduino router connection closed")
        with self.pending_lock:
            self.next_id += 1
            msgid = self.next_id
            event = threading.Event()
            pending = {"event": event, "error": None, "result": None}
            self.pending[msgid] = pending
        try:
            self._send([0, msgid, method, list(args)])
            if not event.wait(timeout):
                raise TimeoutError(f"Bridge call timed out: {method}")
            if pending["error"] is not None:
                raise RuntimeError(f"Bridge call failed: {pending['error']}")
            return pending["result"]
        finally:
            with self.pending_lock:
                self.pending.pop(msgid, None)

    def provide(self, name, handler):
        if not callable(handler):
            raise ValueError("Bridge handler must be callable")
        self.handlers[name] = handler
        try:
            self.call("$/register", name, timeout=5)
        except Exception:
            self.handlers.pop(name, None)
            raise

    def _receive(self):
        unpacker = msgpack.Unpacker(raw=False)
        try:
            while True:
                data = self.socket.recv(4096)
                if not data:
                    break
                unpacker.feed(data)
                for message in unpacker:
                    self._dispatch(message)
        except Exception as exc:
            print("Arduino router receive failed:", exc)
        finally:
            self.closed = True
            with self.pending_lock:
                for pending in self.pending.values():
                    pending["error"] = "Arduino router disconnected"
                    pending["event"].set()

    def _dispatch(self, message):
        if not isinstance(message, list) or not message:
            return
        kind = message[0]
        if kind == 1 and len(message) == 4:
            _, msgid, error, result = message
            with self.pending_lock:
                pending = self.pending.get(msgid)
                if pending is not None:
                    pending["error"] = error
                    pending["result"] = result
                    pending["event"].set()
        elif kind == 2 and len(message) == 3:
            _, name, args = message
            handler = self.handlers.get(name)
            if handler is not None and isinstance(args, list):
                try:
                    handler(*args)
                except Exception as exc:
                    print("Arduino router notification failed:", exc)
        elif kind == 0 and len(message) == 4:
            _, msgid, name, args = message
            handler = self.handlers.get(name)
            if handler is None or not isinstance(args, list):
                response = [1, msgid, "Unknown method", None]
            else:
                try:
                    response = [1, msgid, None, handler(*args)]
                except Exception as exc:
                    response = [1, msgid, str(exc), None]
            self._send(response)

    def close(self):
        if not self.closed:
            self.closed = True
            try:
                self._send([0, self.next_id + 1, "$/reset", []])
            except OSError:
                pass
        self.socket.close()
