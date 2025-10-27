"""
DataLab remote control example
==============================

Example of remote control of DataLab current session,
from a Python script running outside DataLab (e.g. in Spyder)

This example demonstrates how to control DataLab remotely using the
SimpleRemoteProxy from Sigima. The script can be run from any Python
environment outside of DataLab to interact with an active DataLab session.

.. note::
   This example is not executed during documentation build as it requires
   an active DataLab session to connect to. It is included in the gallery
   for reference and can be run manually when DataLab is running.
"""

# %% Importing necessary modules

# NumPy for numerical array computations:
import numpy as np

# DataLab remote control client:
from sigima.client import SimpleRemoteProxy as RemoteProxy

# %% Connecting to DataLab current session

proxy = RemoteProxy(autoconnect=False)
proxy.connect(timeout=0.0, retries=1)  # No timeout for rapid failure if not running

# %% Executing commands in DataLab (...)

z = np.random.rand(20, 20)
proxy.add_image("toto", z)

# %% Executing commands in DataLab (...)

proxy.toggle_auto_refresh(False)  # Turning off auto-refresh
x = np.array([1.0, 2.0, 3.0])
y = np.array([4.0, 5.0, -1.0])
proxy.add_signal("toto", x, y)

# %% Executing commands in DataLab (...)

proxy.calc("derivative")
proxy.toggle_auto_refresh(True)  # Turning on auto-refresh

# %% Executing commands in DataLab (...)

proxy.set_current_panel("image")

# %% Executing a lot of commands without refreshing DataLab

z = np.random.rand(400, 400)
proxy.add_image("foobar", z)
with proxy.context_no_refresh():
    for _idx in range(100):
        proxy.calc("fft")
