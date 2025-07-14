# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
I/O signal formats
"""

from __future__ import annotations

import numpy as np
import scipy.io as sio
from guidata.io import HDF5Reader, HDF5Writer

from sigima.config import _
from sigima.io import ftlab
from sigima.io.base import FormatInfo
from sigima.io.common.converters import convert_array_to_standard_type
from sigima.io.signal import funcs
from sigima.io.signal.base import SignalFormatBase
from sigima.objects.signal import SignalObj
from sigima.worker import CallbackWorkerProtocol


class HDF5SignalFormat(SignalFormatBase):
    """Object representing a HDF5 signal file type"""

    FORMAT_INFO = FormatInfo(
        name=_("HDF5 files"),
        extensions="*.hdf5 *.h5",
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
        with reader.group(self.GROUP_NAME):
            obj = SignalObj()
            obj.deserialize(reader)
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
        xydata, xlabel, xunit, ylabels, yunits, header = funcs.read_csv(
            filename, worker
        )
        if ylabels:
            # If y labels are present, we are sure that the data contains at least
            # two columns (x and y)
            objs = []
            for i, (ylabel, yunit) in enumerate(zip(ylabels, yunits)):
                obj = self.create_object(filename, i if len(ylabels) > 1 else None)
                obj.set_xydata(xydata[:, 0], xydata[:, i + 1])
                obj.xlabel = xlabel or ""
                obj.xunit = xunit or ""
                obj.ylabel = ylabel or ""
                obj.yunit = yunit or ""
                if header:
                    obj.set_metadata_option(self.HEADER_KEY, header)
                objs.append(obj)
            return objs
        return self.create_signals_from(xydata, filename)

    def write(self, filename: str, obj: SignalObj) -> None:
        """Write data to file

        Args:
            filename: Name of file to write
            obj: Signal object to read data from
        """
        funcs.write_csv(
            filename,
            obj.xydata,
            obj.xlabel,
            obj.xunit,
            [obj.ylabel],
            [obj.yunit],
            obj.get_metadata_option(self.HEADER_KEY, ""),
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
        return convert_array_to_standard_type(np.load(filename))

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
