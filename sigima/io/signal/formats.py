# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
I/O signal formats
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.io as sio
from guidata.io import HDF5Reader, HDF5Writer

from sigima.config import _
from sigima.io import ftlab
from sigima.io.base import FormatInfo
from sigima.io.common.converters import convert_array_to_valid_dtype
from sigima.io.signal import funcs
from sigima.io.signal.base import SignalFormatBase
from sigima.objects.signal import SignalObj
from sigima.objects.signal.constants import (
    DATETIME_X_FORMAT_KEY,
    DEFAULT_DATETIME_FORMAT,
)
from sigima.worker import CallbackWorkerProtocol


class HDF5SignalFormat(SignalFormatBase):
    """Object representing a HDF5 signal file type"""

    FORMAT_INFO = FormatInfo(
        name=_("HDF5 files"),
        extensions="*.h5sig",
        readable=True,
        writeable=True,
    )
    GROUP_NAME = "signal"

    def read(
        self, filename: str, worker: CallbackWorkerProtocol | None = None
    ) -> list[SignalObj]:
        """Read list of signal objects from file

        Args:
            filename: File name
            worker: Callback worker object

        Returns:
            List of signal objects
        """
        reader = HDF5Reader(filename)
        try:
            with reader.group(self.GROUP_NAME):
                obj = SignalObj()
                obj.deserialize(reader)
        except ValueError as exc:
            raise ValueError("No valid signal data found") from exc
        except Exception as exc:  # pylint: disable=broad-except
            raise RuntimeError(
                f"Unexpected error reading HDF5 signal from {filename}"
            ) from exc
        finally:
            reader.close()
        return [obj]

    def write(self, filename: str, obj: SignalObj) -> None:
        """Write data to file

        Args:
            filename: Name of file to write
            obj: Signal object to read data from
        """
        assert isinstance(obj, SignalObj), "Object is not a signal"
        writer = HDF5Writer(filename)
        with writer.group(self.GROUP_NAME):
            obj.serialize(writer)
        writer.close()


class CSVSignalFormat(SignalFormatBase):
    """Object representing a CSV signal file type"""

    FORMAT_INFO = FormatInfo(
        name=_("CSV files"),
        extensions="*.csv *.txt",
        readable=True,
        writeable=True,
    )

    def read(
        self, filename: str, worker: CallbackWorkerProtocol | None = None
    ) -> list[SignalObj]:
        """Read list of signal objects from file

        Args:
            filename: File name
            worker: Callback worker object

        Returns:
            List of signal objects
        """
        csv_data = funcs.read_csv(filename, worker)

        if csv_data.ylabels:
            # If y labels are present, we are sure that the data contains at least
            # two columns (x and y)
            objs = []
            for i, (ylabel, yunit) in enumerate(zip(csv_data.ylabels, csv_data.yunits)):
                obj = self.create_object(
                    filename, i if len(csv_data.ylabels) > 1 else None
                )
                obj.set_xydata(csv_data.xydata[:, 0], csv_data.xydata[:, i + 1])
                obj.xlabel = csv_data.xlabel or ""
                # Set xunit, defaulting to 's' if datetime signal and no unit specified
                if csv_data.datetime_metadata and not csv_data.xunit:
                    obj.xunit = "s"  # Default unit for datetime signals
                else:
                    obj.xunit = csv_data.xunit or ""
                obj.ylabel = ylabel or ""
                obj.yunit = yunit or ""
                if csv_data.header:
                    obj.metadata[self.HEADER_KEY] = csv_data.header
                # Add datetime metadata if detected
                if csv_data.datetime_metadata:
                    obj.metadata.update(csv_data.datetime_metadata)
                # Add column metadata (constant-value columns like serial numbers)
                if csv_data.column_metadata:
                    obj.metadata.update(csv_data.column_metadata)
                objs.append(obj)
            return objs
        return self.create_signals_from(csv_data.xydata, filename)

    def write(self, filename: str, obj: SignalObj) -> None:
        """Write data to file

        Args:
            filename: Name of file to write
            obj: Signal object to read data from
        """
        # If X is datetime, convert back to datetime strings for CSV
        if obj.is_x_datetime():
            datetime_values = obj.get_x_as_datetime()
            # Convert to strings with appropriate format
            datetime_format = obj.metadata.get(
                DATETIME_X_FORMAT_KEY, DEFAULT_DATETIME_FORMAT
            )
            x_data = pd.to_datetime(datetime_values).strftime(datetime_format).values
            # Create modified xydata with datetime strings in X column
            # We'll write manually with pandas to preserve datetime strings
            data_dict = {obj.xlabel or "Time": x_data, obj.ylabel or "Y": obj.y}
            df = pd.DataFrame(data_dict)
            df.to_csv(filename, index=False)
        else:
            funcs.write_csv(
                filename,
                obj.xydata,
                obj.xlabel,
                obj.xunit,
                [obj.ylabel],
                [obj.yunit],
                obj.metadata.get(self.HEADER_KEY, ""),
            )


class NumPySignalFormat(SignalFormatBase):
    """Object representing a NumPy signal file type"""

    FORMAT_INFO = FormatInfo(
        name=_("NumPy binary files"),
        extensions="*.npy",
        readable=True,
        writeable=True,
    )  # pylint: disable=duplicate-code

    def read_xydata(self, filename: str) -> np.ndarray:
        """Read data and metadata from file, write metadata to object, return xydata

        Args:
            filename: Name of file to read

        Returns:
            NumPy array xydata
        """
        return convert_array_to_valid_dtype(np.load(filename), SignalObj.VALID_DTYPES)

    def write(self, filename: str, obj: SignalObj) -> None:
        """Write data to file

        Args:
            filename: Name of file to write
            obj: Signal object to read data from
        """
        np.save(filename, obj.xydata.T)


class MatSignalFormat(SignalFormatBase):
    """Object representing a MAT-File .mat signal file type"""

    FORMAT_INFO = FormatInfo(
        name=_("MAT-Files"),
        extensions="*.mat",
        readable=True,
        writeable=True,
    )  # pylint: disable=duplicate-code

    def read(
        self, filename: str, worker: CallbackWorkerProtocol | None = None
    ) -> list[SignalObj]:
        """Read data and metadata from file, write metadata to object, return xydata

        Args:
            filename: Name of file to read

        Returns:
            NumPy array xydata
        """
        mat = sio.loadmat(filename)
        allsig: list[SignalObj] = []
        for dname, data in mat.items():
            if dname.startswith("__") or not isinstance(data, np.ndarray):
                continue
            for sig in self.create_signals_from(data.squeeze(), filename):
                if dname != "sig":
                    sig.title += f" ({dname})"
                allsig.append(sig)
        return allsig

    def write(self, filename: str, obj: SignalObj) -> None:
        """Write data to file

        Args:
            filename: Name of file to write
            obj: Signal object to read data from
        """
        # metadata cannot be saved as such as their type will be lost and
        # cause problems when reading the file back
        sio.savemat(filename, {"sig": obj.xydata.T})


class FTLabSignalFormat(SignalFormatBase):
    """FT-Lab signal file."""

    FORMAT_INFO = FormatInfo(
        name=_("FT-Lab"),
        extensions="*.sig",
        readable=True,
        writeable=False,
    )

    def read_xydata(self, filename: str) -> np.ndarray:
        """Read data and metadata from file, populate metadata, return xydata.

        Args:
            filename: Path to FT-Lab file.

        Returns:
            Signal data.
        """
        return ftlab.sigread_ftlabsig(filename)


class MCASignalFormat(SignalFormatBase):
    """Object representing a MCA signal file type"""

    FORMAT_INFO = FormatInfo(
        name=_("MCA files"),
        extensions="*.mca",
        readable=True,
        writeable=False,
    )

    def read(
        self, filename: str, worker: CallbackWorkerProtocol | None = None
    ) -> list[SignalObj]:
        """Read list of signal objects from file

        Args:
            filename: File name
            worker: Callback worker object

        Returns:
            List of signal objects
        """
        mca = funcs.MCAFile(filename)
        mca.read()
        obj = self.create_object(filename)
        obj.set_xydata(mca.x, mca.y)
        obj.xlabel = mca.xlabel or ""
        obj.metadata = mca.metadata
        return [obj]
