#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import time
from queue import Queue
from threading import Thread
from typing import Any

from azureml.tensorboard import Tensorboard


class HotFixedTensorBoard(Tensorboard):
    """
    This is a TEMPORARY hotfixed version of the TensorBoard class with commented _get_url call
    as that is buggy in the AzureML library and will wait forever.

    REMOVE THIS FILE WHEN THIS IS FIXED IN THE LIBRARY
    """

    def _wait_for_url(self, timeout: int = 60) -> None:
        def enqueue_output(out: Any, q: Any) -> None:
            for line in iter(out.readline, b''):
                q.put(line)

        queue: Queue = Queue()
        thread = Thread(target=enqueue_output, args=(self._tb_proc.stderr, queue))
        thread.daemon = True
        thread.start()

        log = ""
        start_time = time.monotonic()
        while time.monotonic() - start_time < timeout:
            time_remaining = timeout - (time.monotonic() - start_time)
            try:
                line = queue.get(timeout=time_remaining)
                url = self._get_url(line)
                if url is not None:
                    return url
                else:
                    log += line
            except Exception:
                continue

        raise Exception("Tensorboard did not report a listening URL. Log from Tensorboard follows:\n{}".format(log))
