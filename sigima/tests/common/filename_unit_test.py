# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.
"""
Unit tests for the `format_object_names` function in `sigima.io.common.filename`.
"""
# Pylint: test stubs intentionally do not call base __init__
# pylint: disable=super-init-not-called

import pytest

from sigima.io.common.filename import format_object_names
from sigima.objects.image import ImageObj
from sigima.objects.signal import SignalObj


class DummySignal(SignalObj):
    """Lightweight SignalObj stub for tests."""

    def __init__(
        self,
        title="",
        xlabel="",
        xunit="",
        ylabel="",
        yunit="",
        metadata=None,
    ):
        # Minimal stub to satisfy attribute access in format_object_names
        self.title = title
        self.xlabel = xlabel
        self.xunit = xunit
        self.ylabel = ylabel
        self.yunit = yunit
        self.metadata = {} if metadata is None else metadata


class DummyImage(ImageObj):
    """Lightweight ImageObj stub for tests."""

    def __init__(self, title="", metadata=None):
        # Minimal stub to satisfy attribute access in format_object_names
        self.title = title
        self.metadata = {} if metadata is None else metadata


def test_basic_sanitize_and_name_only():
    """Sanitize titles and output {title}."""
    objs = [DummySignal("A/B"), DummyImage("C/D")]  # '/' must be sanitized on all OSes
    names = format_object_names(objs, fmt="{title}")
    assert names == ["A_B", "C_D"]


def test_indices_and_total_count():
    """Indexing and total count placeholders."""
    objs = [DummySignal("sig1"), DummySignal("sig2"), DummySignal("sig3")]
    names = format_object_names(objs, fmt="{title}_{index:02d}-of-{n}")
    assert names == ["sig1_01-of-3", "sig2_02-of-3", "sig3_03-of-3"]


def test_metadata_and_axes_placeholders():
    """Use metadata and axis placeholders."""
    sig = DummySignal(
        title="My/Signal",
        xlabel="Time",
        xunit="s",
        ylabel="Amp",
        yunit="V",
        metadata={"id": 42},
    )
    names = format_object_names([sig], fmt="{title}_{xlabel}[{xunit}]_{metadata[id]}")
    assert names == ["My_Signal_Time[s]_42"]


def test_type_placeholder_for_signal_and_image():
    """Use {type} for signal and image."""
    objs = [DummySignal("s"), DummyImage("i")]
    names = format_object_names(objs, fmt="{type}_{title}")
    assert names == ["signal_s", "image_i"]


def test_custom_replacement_character():
    """Use custom replacement character."""
    objs = [DummySignal("a/b"), DummyImage("c/d")]
    names = format_object_names(objs, fmt="{title}", replacement="-")
    assert names == ["a-b", "c-d"]


def test_unknown_placeholder_raises_keyerror():
    """Unknown placeholder should raise KeyError."""
    with pytest.raises(KeyError):
        format_object_names([DummySignal("x")], fmt="{unknown}")
