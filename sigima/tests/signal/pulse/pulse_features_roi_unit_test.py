# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Pulse features with ROIs unit test.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

from __future__ import annotations

from sigima.objects import SignalObj, TableResult, create_signal_roi
from sigima.proc.signal import PulseFeaturesParam, extract_pulse_features, extract_roi
from sigima.tests.data import create_paracetamol_signal


def generate_source_signal() -> SignalObj:
    """Generate source signal with ROIs."""
    sig = create_paracetamol_signal()
    roi = create_signal_roi([[12.0, 15.3], [33.1, 35.4], [37.1, 40.1]])
    sig.roi = roi
    return sig


def check_results_equal(src_res: TableResult, dst_res: list[TableResult]) -> None:
    """Check that two TableResults are equal."""
    src_df = src_res.to_dataframe()
    diff_txt = ""
    for roi_idx, dst in enumerate(dst_res):
        dst_df = dst.to_dataframe()
        if len(dst_df) != 1:
            diff_txt += f"ROI {roi_idx}: unexpected number of rows: {len(dst_df)}\n"
            continue
        for col in src_df.columns:
            val_src = src_df.iloc[roi_idx][col]
            # Skip roi_index column - it's metadata, not a pulse feature
            # (extracted signals have roi_index=-1 since they have no ROIs)
            if col == "roi_index":
                assert int(val_src) == roi_idx
                continue
            val_dst = dst_df.iloc[0][col]
            if val_src != val_dst:
                diff_txt += (
                    f"ROI {roi_idx}, column '{col}': "
                    f"source value={val_src} != extracted value={val_dst}\n"
                )
    if diff_txt:
        raise AssertionError(
            "Pulse features extracted from ROIs "
            "do not match original signal ROI features:\n" + diff_txt
        )


def __extract_pulse_features(obj: SignalObj) -> TableResult:
    """Extract pulse features."""
    param = PulseFeaturesParam()
    param.update_from_obj(obj)
    return extract_pulse_features(obj, param)


def test_pulse_features_roi():
    """Pulse features with ROIs application test."""
    # Test signal with multiple ROIs defined around peaks of the spectrum
    sig = generate_source_signal()
    pf_sig = __extract_pulse_features(sig)

    # Extract ROIs in separate signals and test each one
    pf_extracted_sigs = []
    for roi_idx, single_roi in enumerate(sig.roi):
        extracted_sig = extract_roi(sig, single_roi.to_param(sig, roi_idx))
        pf_extracted_sig = __extract_pulse_features(extracted_sig)
        pf_extracted_sigs.append(pf_extracted_sig)
    check_results_equal(pf_sig, pf_extracted_sigs)


if __name__ == "__main__":
    test_pulse_features_roi()
