# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests for annotation API."""

import json

import pytest

from sigima.objects.image.creation import create_image
from sigima.objects.signal.creation import create_signal


def test_get_empty_annotations():
    """Test getting annotations from object without annotations."""
    obj = create_signal("Test")
    assert obj.get_annotations() == []
    assert not obj.has_annotations()


def test_get_empty_annotations_image():
    """Test getting annotations from image object without annotations."""
    obj = create_image("Test")
    assert obj.get_annotations() == []
    assert not obj.has_annotations()


def test_set_get_annotations():
    """Test setting and getting annotations."""
    obj = create_signal("Test")
    annotations = [{"type": "label", "text": "Test"}]
    obj.set_annotations(annotations)
    assert obj.get_annotations() == annotations
    assert obj.has_annotations()


def test_set_get_annotations_multiple():
    """Test setting and getting multiple annotations."""
    obj = create_signal("Test")
    annotations = [
        {"type": "label", "x": 10, "y": 20, "text": "Peak"},
        {"type": "rectangle", "x0": 0, "y0": 0, "x1": 100, "y1": 100},
    ]
    obj.set_annotations(annotations)
    result = obj.get_annotations()
    assert len(result) == 2
    assert result == annotations


def test_add_annotation():
    """Test adding single annotation."""
    obj = create_signal("Test")
    obj.add_annotation({"type": "circle", "r": 5})
    obj.add_annotation({"type": "rect", "w": 10})
    annotations = obj.get_annotations()
    assert len(annotations) == 2
    assert annotations[0]["type"] == "circle"
    assert annotations[1]["type"] == "rect"


def test_clear_annotations():
    """Test clearing annotations."""
    obj = create_signal("Test")
    obj.set_annotations([{"type": "test"}])
    assert obj.has_annotations()
    obj.clear_annotations()
    assert obj.get_annotations() == []
    assert not obj.has_annotations()


def test_invalid_json():
    """Test handling of invalid JSON."""
    obj = create_signal("Test")
    obj.annotations = "invalid{json"
    assert obj.get_annotations() == []


def test_non_json_serializable():
    """Test rejection of non-serializable data."""
    obj = create_signal("Test")
    with pytest.raises(ValueError):
        obj.set_annotations([{"func": lambda x: x}])


def test_invalid_type():
    """Test rejection of non-list annotations."""
    obj = create_signal("Test")
    with pytest.raises(TypeError):
        obj.set_annotations({"type": "dict"})  # type: ignore


def test_versioned_format():
    """Test that annotations are stored with version."""
    obj = create_signal("Test")
    obj.set_annotations([{"type": "test"}])

    # Check internal storage format
    data = json.loads(obj.annotations)
    assert "version" in data
    assert data["version"] == "1.0"
    assert "annotations" in data


def test_annotation_persistence():
    """Test that annotations persist through copy operations."""
    obj = create_signal("Test")
    obj.set_annotations([{"type": "label", "text": "Original"}])

    # Copy the object
    obj_copy = obj.copy()

    # Annotations should be copied
    assert obj_copy.get_annotations() == obj.get_annotations()


def test_complex_annotation_data():
    """Test complex nested annotation structures."""
    obj = create_image("Test")
    complex_annotation = {
        "type": "complex",
        "properties": {
            "color": "red",
            "style": {"width": 2, "dash": [5, 3]},
        },
        "geometry": [[0, 0], [10, 10], [20, 0]],
        "metadata": {"created": "2025-11-02", "author": "test"},
    }
    obj.add_annotation(complex_annotation)

    retrieved = obj.get_annotations()
    assert len(retrieved) == 1
    assert retrieved[0] == complex_annotation


def test_empty_annotations_field():
    """Test behavior with explicitly empty annotations field."""
    obj = create_signal("Test")
    obj.annotations = ""
    assert obj.get_annotations() == []
    assert not obj.has_annotations()


def test_malformed_json_structure():
    """Test handling of valid JSON but wrong structure."""
    obj = create_signal("Test")
    obj.annotations = json.dumps({"wrong": "structure"})
    assert obj.get_annotations() == []


if __name__ == "__main__":
    pytest.main([__file__])
