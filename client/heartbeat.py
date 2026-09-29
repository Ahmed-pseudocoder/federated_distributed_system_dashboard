from __future__ import annotations

import os
import threading

from core.messages import message


class HeartbeatEmitter(threading.Thread):
    def __init__(self, client_id, event_queue, progress, lock, interval, stop_event):
        super().__init__(name=f"heartbeat-{client_id}", daemon=True)
        self.client_id = client_id
        self.event_queue = event_queue
        self.progress = progress
        self.lock = lock
        self.interval = interval
        self.stop_event = stop_event

    def run(self) -> None:
        while not self.stop_event.is_set():
            with self.lock:
                snapshot = dict(self.progress)
            self.event_queue.put(
                message("heartbeat", client_id=self.client_id, pid=os.getpid(), **snapshot)
            )
            self.stop_event.wait(self.interval)

