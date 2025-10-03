# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Sigima Client
=============

Sigima Client provides a proxy to DataLab application through XML-RPC protocol.
This is the client subpackage within the Sigima scientific computing engine.
"""

from sigima.client.baseproxy import SimpleBaseProxy
from sigima.client.remote import SimpleRemoteProxy

__all__ = ["SimpleBaseProxy", "SimpleRemoteProxy"]

# TODO: Refactor the `sigima.client` with `datalab.baseproxy` and `datalab.remote`
