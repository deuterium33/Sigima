# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Sigima I/O image formats
"""

from __future__ import annotations

import os.path as osp
import re

import imageio.v3 as iio
import numpy as np
import pandas as pd
import scipy.io as sio
import skimage.io
from guidata.io import HDF5Reader, HDF5Writer

from sigima.config import _, options
from sigima.io import ftlab
from sigima.io.base import FormatInfo
from sigima.io.common.converters import convert_array_to_valid_dtype
from sigima.io.enums import FileEncoding
from sigima.io.image import funcs
from sigima.io.image.base import (
    ImageFormatBase,
    MultipleImagesFormatBase,
    SingleImageFormatBase,
)
from sigima.objects.image import ImageObj, create_image
from sigima.worker import CallbackWorkerProtocol


class HDF5ImageFormat(ImageFormatBase):
    """Object representing HDF5 image file type"""

    FORMAT_INFO = FormatInfo(
        name="HDF5",
        extensions="*.h5ima",
        readable=True,
        writeable=True,
    )
    GROUP_NAME = "image"

    # pylint: disable=unused-argument
    def read(
        self, filename: str, worker: CallbackWorkerProtocol | None = None
    ) -> list[ImageObj]:
        """Read list of image objects from file

        Args:
            filename: File name
            worker: Callback worker object

        Returns:
            List of image objects
        """
        reader = HDF5Reader(filename)
        try:
            with reader.group(self.GROUP_NAME):
                obj = ImageObj()
                obj.deserialize(reader)
        except ValueError as exc:
            raise ValueError("No valid image data found") from exc
        except Exception as exc:  # pylint: disable=broad-except
            raise RuntimeError(
                f"Unexpected error reading HDF5 image from {filename}"
            ) from exc
        finally:
            reader.close()
        return [obj]

    def write(self, filename: str, obj: ImageObj) -> None:
        """Write data to file

        Args:
            filename: file name
            obj: native object (signal or image)

        Raises:
            NotImplementedError: if format is not supported
        """
        assert isinstance(obj, ImageObj), "Object is not an image"
        writer = HDF5Writer(filename)
        with writer.group(self.GROUP_NAME):
            obj.serialize(writer)
        writer.close()


class ClassicsImageFormat(SingleImageFormatBase):
    """Object representing classic image file types"""

    FORMAT_INFO = FormatInfo(
        name="BMP, JPEG, PNG, TIFF, JPEG2000",
        extensions="*.bmp *.jpg *.jpeg *.png *.tif *.tiff *.jp2",
        readable=True,
        writeable=True,
    )

    @staticmethod
    def read_data(filename: str) -> np.ndarray:
        """Read data and return it

        Args:
            filename: File name

        Returns:
            Image array data
        """
        return skimage.io.imread(filename, as_gray=True)

    @staticmethod
    def write_data(filename: str, data: np.ndarray) -> None:
        """Write data to file

        Args:
            filename: File name
            data: Image array data
        """
        ext = osp.splitext(filename)[1].lower()
        if ext in (".bmp", ".jpg", ".jpeg", ".png"):
            if data.dtype is not np.uint8:
                data = data.astype(np.uint8)
        if ext in (".jp2",):
            if data.dtype not in (np.uint8, np.uint16):
                data = data.astype(np.uint16)
        skimage.io.imsave(filename, data, check_contrast=False)


class NumPyImageFormat(SingleImageFormatBase):
    """Object representing NumPy image file type"""

    FORMAT_INFO = FormatInfo(
        name="NumPy",
        extensions="*.npy",
        readable=True,
        writeable=True,
    )  # pylint: disable=duplicate-code

    @staticmethod
    def read_data(filename: str) -> np.ndarray:
        """Read data and return it

        Args:
            filename: File name

        Returns:
            Image array data
        """
        return convert_array_to_valid_dtype(np.load(filename), ImageObj.VALID_DTYPES)

    @staticmethod
    def write_data(filename: str, data: np.ndarray) -> None:
        """Write data to file

        Args:
            filename: File name
            data: Image array data
        """
        np.save(filename, data)


class NotMatrisFileError(Exception):
    """Exception raised when a file is not a Matris file"""


class TextImageFormat(SingleImageFormatBase):
    """Object representing text image file type"""

    FORMAT_INFO = FormatInfo(
        name=_("Text files"),
        extensions="*.txt *.csv *.asc",
        readable=True,
        writeable=True,
    )

    def read(
        self, filename: str, worker: CallbackWorkerProtocol | None = None
    ) -> list[ImageObj]:
        """Read list of image objects from file

        Args:
            filename: File name
            worker: Callback worker object

        Returns:
            List of image objects
        """
        # Default implementation covers the case of a single image:
        try:
            matris_image = MatrisImageFormat()
            return matris_image.read(filename, worker)
        except NotMatrisFileError:
            obj = self.create_object(filename)
            obj.data = self.read_data(filename)
            unique_values = np.unique(obj.data)
            if len(unique_values) == 2:
                # Binary image: set LUT range to unique values
                obj.zscalemin, obj.zscalemax = unique_values.tolist()
            return [obj]

    @staticmethod
    def read_data(filename: str) -> np.ndarray:
        """Read data and return it

        Args:
            filename: File name

        Returns:
            Image array data
        """
        for encoding in FileEncoding:
            for decimal in (".", ","):
                for delimiter in (",", ";", r"\s+"):
                    try:
                        df = pd.read_csv(
                            filename,
                            decimal=decimal,
                            delimiter=delimiter,
                            encoding=encoding,
                            header=None,
                        )
                        # Handle the extra column created with trailing delimiters.
                        df = df.dropna(axis=1, how="all")
                        data = df.to_numpy()
                        return convert_array_to_valid_dtype(data, ImageObj.VALID_DTYPES)
                    except ValueError:
                        continue
        raise ValueError(f"Could not read image data from file {filename}.")

    @staticmethod
    def write_data(filename: str, data: np.ndarray) -> None:
        """Write data to file.

        Args:
            filename: File name.
            data: Image array data.
        """
        if np.issubdtype(data.dtype, np.integer):
            fmt = "%d"
        elif np.issubdtype(data.dtype, np.floating) or np.issubdtype(
            data.dtype, np.complexfloating
        ):
            fmt = "%.18e"
        else:
            raise NotImplementedError(
                f"Writing data of type {data.dtype} to text file is not supported."
            )
        ext = osp.splitext(filename)[1]
        if ext.lower() in (".txt", ".asc", ""):
            np.savetxt(filename, data, fmt=fmt)
        elif ext.lower() == ".csv":
            np.savetxt(filename, data, fmt=fmt, delimiter=",")
        else:
            raise ValueError(f"Unknown text file extension {ext}")


class MatrisImageFormat(ImageFormatBase):
    """Object representing image file type issue from matris software"""

    FORMAT_INFO = FormatInfo(
        name=_("Matris software files format"),
        extensions="*.matris",
        readable=True,
        writeable=False,
    )

    def read(self, filename, worker=None):
        """Read list of image objects from file
        Args:
            filename: File name
            worker: Callback worker object

        Returns:
            List of image objects
        """
        file_metadata = self.read_metadata(filename)

        # Validate metadata and raise on inconsistent or missing keys
        self.verify_metadata(filename, file_metadata)

        dict_keys = file_metadata.keys()
        allowed_column_header = {
            "X",
            "Y",
            "Z",
            "Zre",
            "Zim",
            "Z Error",
            "Zre Error",
            "Zim Error",
        }
        columns_header = [k for k in dict_keys if k in allowed_column_header]

        # Remove keys that are in columns_header and construct metadata dictionary
        metadata = {
            key: value[0]
            for key, value in file_metadata.items()
            if key not in columns_header
        }
        metadata["source"] = filename

        df = self.read_data(filename, columns_header)

        name = osp.basename(filename)

        try:
            data = df.pivot(index="Y", columns="X", values="Z").values
            data = convert_array_to_valid_dtype(data, ImageObj.VALID_DTYPES)

            (zlabel, zunit) = file_metadata.get("Z", file_metadata.get("Zre", ("", "")))
            (xlabel, xunit) = file_metadata.get("X", ("X", ""))
            (ylabel, yunit) = file_metadata.get("Y", ("Y", ""))

            if xlabel is None:
                xlabel = "X"
            if ylabel is None:
                ylabel = "Y"
            if zlabel is None:
                zlabel = "Z"

            xunit = "" if xunit is None else str(xunit)
            yunit = "" if yunit is None else str(yunit)
            zunit = "" if zunit is None else str(zunit)

            image = create_image(
                name,
                metadata=metadata,
                data=data,
                units=(xunit, yunit, zunit),
                labels=(xlabel, ylabel, zlabel),
            )
            images_list = [image]
        except ValueError as exc:
            raise ValueError(f"File {filename} wrong format.\n{exc}") from exc

        if "Z Error" in df.columns:
            image_error = create_image(
                name + " error",
                metadata={"source": filename},
                data=df.pivot(index="Y", columns="X", values="Z Error").values,
                units=(
                    file_metadata["X"][1],
                    file_metadata["Y"][1],
                    file_metadata.get(
                        "Z Error",
                        file_metadata.get(
                            "Zre Error",
                            file_metadata.get("Z", file_metadata.get("Zre", ("", ""))),
                        ),
                    )[1],
                ),
                labels=(
                    file_metadata["X"][0],
                    file_metadata["Y"][0],
                    file_metadata.get("Z", file_metadata.get("Zre", ("", "")))[0]
                    + " error",
                ),
            )
            images_list.append(image_error)

        return images_list

    @staticmethod
    def read_metadata(filename: str) -> dict[str, tuple | None]:
        """Read metadata from file

        Args:
            filename: File name

        Returns:
            Metadata dictionary structured as {key: (value, unit)}
            Available keys can be are:
            - nx (value is int)
            - ny (value is int)
            - X (value represents axis label)
            - Y (value represents axis label)
            - Z (value represents axis label)
            - Zre (value represents axis label)
            - Zim (value represents axis label)
            - Z Error (value is none)
            - Zre Error (value is none)
            - Zim Error (value is none)
        """
        metadata = {}
        generic_metadata_pattern = (
            r"^([\w+ ]*\w)\s*:\s*([\d]*)?([^\(\)]*)?\s*\(?([^\(\)]*)\)?"
        )
        date_time_pattern = r"# Created on (\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}\.\d+)"
        software_version_pattern = r"# Using matrislib ([\d\.a-zA-Z-]+)"
        author_pattern = r"# Created by (.*)"
        try:
            with open(filename, encoding="utf-8") as f:
                for line in f:
                    # strip initial whitespaces
                    line = line.strip()
                    if not line.startswith("#"):
                        break
                    # Remove leading '#' and strip whitespace
                    line = line[1:].strip()

                    if match := re.match(author_pattern, line):
                        metadata["author"] = (match.group(1).strip(), None)
                        continue

                    if match := re.search(date_time_pattern, line):
                        date_str = match.group(1)  # "2020-12-03"
                        time_str = match.group(2)  # "09:35:33.049769"
                        metadata["creation_date"] = (f"{date_str}", None)
                        metadata["creation_time"] = (time_str, None)
                        continue

                    if match := re.search(software_version_pattern, line):
                        metadata["software_version"] = (
                            f"matrislib {match.group(1)}",
                            None,
                        )
                        continue

                    # Ensure the line contains a colon for key-value separation
                    if ":" not in line:
                        if line.startswith(("Z Error", "Zre Error", "Zim Error")):
                            # Handle special cases for Z Error
                            line = line.replace("Error", "Error :", 1)
                    # Remove Real(...) or Imaginary(...) wrappers if present
                    line = re.sub(
                        r"(?:Real|Imaginary)\(([^\)]*)\)", r"\1", line, count=1
                    )
                    # Match key : value (possibly with units in parentheses)

                    if match := re.match(generic_metadata_pattern, line):
                        key = match.group(1).strip()

                        # Determine if value is int, str, or None
                        if match.group(2) != "":
                            value = int(match.group(2).strip())
                        elif match.group(3) != "":
                            value = match.group(3)
                        else:
                            value = None
                        unit = match.group(4).strip()
                        if unit == "":
                            unit = None
                        metadata[key] = (value, unit)

        except ValueError as exc:
            raise ValueError(f"Could not read metadata from file {filename}") from exc

        return metadata

    @staticmethod
    def verify_metadata(filename: str, metadata: dict[str, tuple | None]) -> None:
        """Verify metadata keys consistency.

        Perform a set of sanity checks on the parsed metadata and raise an
        appropriate exception on failure.

        Args:
            filename: Parsed filename used for error messages.
            metadata: Metadata dictionary parsed from file header.

        Raises:
            SyntaxError: On mutually exclusive or missing field combinations.
            ValueError: When required fields are missing.
        """
        err = ValueError
        if "software_version" not in metadata:
            err = NotMatrisFileError

        columns_header = [k for k in metadata.keys() if k not in ("nx", "ny")]

        if "Z Error" in columns_header and (
            "Zre Error" in columns_header or "Zim Error" in columns_header
        ):
            raise err(
                f"File {filename} contains both Z Error and Zre Error/Zim Error"
                " columns, which is not supported."
            )

        one_and_only_one_z = ("Z" in columns_header) ^ (
            "Zre" in columns_header or "Zim" in columns_header
        )

        if one_and_only_one_z is False:
            raise err(
                f"File {filename} should contain one and only one "
                "between both Z and Zre/Zim columns,"
                " which is not supported."
            )

        if "X" not in columns_header or "Y" not in columns_header:
            raise err(
                f"File {filename} wrong format: X, Y and Z columns should be present"
            )

        if ("Zre" in columns_header) ^ ("Zim" in columns_header):
            raise err(
                f"File {filename} contains only one between Zre/Zim columns,"
                " which is not supported."
            )
        if ("Zre Error" in columns_header) ^ ("Zim Error" in columns_header):
            raise err(
                f"File {filename} contains only one between 'Zre Error'/ 'Zim Error'"
                " columns, which is not supported."
            )

    @staticmethod
    def _try_df_reading(filename: str, columns_header: list[str]) -> pd.DataFrame:
        """Try to read the data file with various parsing options.
        Args:
            filename: File name
            columns_header: List of column headers to use when reading the data.
        Returns:
            DataFrame containing the image data.
        Raises:
            ValueError: If the file cannot be read with any of the tried options.
        Note:
            use this function to avoid nested loops with continue and break
            i.e. see PEP 3136 rejection note (https://peps.python.org/pep-3136/)
        """
        for encoding in FileEncoding:
            for decimal in (".", ","):
                for delimiter in (r"\s+", ",", ";"):
                    try:
                        df = pd.read_csv(
                            filename,
                            decimal=decimal,
                            comment="#",
                            delimiter=delimiter,
                            encoding=encoding,
                            names=columns_header,
                        )
                        # drop entirely empty columns
                        # introduced by trailing delimiters
                        df = df.dropna(axis=1, how="all")
                        return df
                    except ValueError:
                        continue
        raise ValueError(f"Could not read image data from file {filename}.")

    @staticmethod
    def read_data(filename: str, columns_header: list[str]) -> pd.DataFrame:
        """Read data and return it.

        Args:
            filename: File name

        Returns:
            Image array data
        """
        # Try several parsing variants (encoding, decimal and delimiter).
        df: pd.DataFrame | None = None

        df = MatrisImageFormat._try_df_reading(filename, columns_header)

        # if Z is present, the image is Real

        if "Zre" in df.columns:
            df["Z"] = df["Zre"] + 1j * df["Zim"]
            df = df.drop(columns=["Zre", "Zim"])
            if "Zre Error" in df.columns:
                df["Z Error"] = df["Zre Error"] + 1j * df["Zim Error"]
                df = df.drop(columns=["Zre Error", "Zim Error"])

        return df

    def write(self, filename: str, obj: ImageObj) -> None:
        """Write data to file not implemented"""
        raise NotImplementedError("Writing Matris files is not supported.")


class MatImageFormat(SingleImageFormatBase):
    """Object representing MAT-File image file type"""

    FORMAT_INFO = FormatInfo(
        name=_("MAT-Files"),
        extensions="*.mat",
        readable=True,
        writeable=True,
    )  # pylint: disable=duplicate-code

    def read(
        self, filename: str, worker: CallbackWorkerProtocol | None = None
    ) -> list[ImageObj]:
        """Read list of image objects from file

        Args:
            filename: File name
            worker: Callback worker object

        Returns:
            List of image objects
        """
        mat = sio.loadmat(filename)
        allimg: list[ImageObj] = []
        for dname, data in mat.items():
            if dname.startswith("__") or not isinstance(data, np.ndarray):
                continue
            if len(data.shape) != 2:
                continue
            obj = self.create_object(filename)
            obj.data = data
            if dname != "img":
                obj.title += f" ({dname})"
            allimg.append(obj)
        return allimg

    @staticmethod
    def read_data(filename: str) -> np.ndarray:
        """Read data and return it

        Args:
            filename: File name

        Returns:
            Image array data
        """
        # This method is not used, as read() is overridden

    @staticmethod
    def write_data(filename: str, data: np.ndarray) -> None:
        """Write data to file

        Args:
            filename: File name
            data: Image array data
        """
        sio.savemat(filename, {"img": data})


class DICOMImageFormat(SingleImageFormatBase):
    """Object representing DICOM image file type"""

    FORMAT_INFO = FormatInfo(
        name="DICOM",
        extensions="*.dcm *.dicom",
        readable=True,
        writeable=False,
        requires=["pydicom"],
    )

    @staticmethod
    def read_data(filename: str) -> np.ndarray:
        """Read data and return it

        Args:
            filename: File name

        Returns:
            Image array data
        """
        return funcs.imread_dicom(filename)


class AndorSIFImageFormat(MultipleImagesFormatBase):
    """Object representing an Andor SIF image file type"""

    FORMAT_INFO = FormatInfo(
        name="Andor SIF",
        extensions="*.sif",
        readable=True,
        writeable=False,
    )

    @staticmethod
    def read_data(filename: str) -> np.ndarray:
        """Read data and return it

        Args:
            filename: File name

        Returns:
            Image array data
        """
        return funcs.imread_sif(filename)


# Generate classes based on the information above:
def generate_imageio_format_classes(
    imageio_formats: list[list[str, str]]
    | list[tuple[str, str]]
    | tuple[tuple[str, str]]
    | tuple[list[str, str]]
    | None = None,
) -> None:
    """Generate classes based on the information above"""
    if imageio_formats is None:
        imageio_formats = options.imageio_formats.get()

    for extensions, name in imageio_formats:
        class_dict = {
            "FORMAT_INFO": FormatInfo(
                name=name, extensions=extensions, readable=True, writeable=False
            ),
            "read_data": staticmethod(
                lambda filename: iio.imread(filename, index=None)
            ),
        }
        class_name = extensions.split()[0].split(".")[1].upper() + "ImageFormat"
        globals()[class_name] = type(
            class_name, (MultipleImagesFormatBase,), class_dict
        )


generate_imageio_format_classes()


class SpiriconImageFormat(SingleImageFormatBase):
    """Object representing Spiricon image file type"""

    FORMAT_INFO = FormatInfo(
        name="Spiricon",
        extensions="*.scor-data",
        readable=True,
        writeable=False,
    )

    @staticmethod
    def read_data(filename: str) -> np.ndarray:
        """Read data and return it

        Args:
            filename: File name

        Returns:
            Image array data
        """
        return funcs.imread_scor(filename)


class XYZImageFormat(SingleImageFormatBase):
    """Object representing Dürr NDT XYZ image file type"""

    FORMAT_INFO = FormatInfo(
        name="Dürr NDT",
        extensions="*.xyz",
        readable=True,
        writeable=False,
    )

    @staticmethod
    def read_data(filename: str) -> np.ndarray:
        """Read data and return it

        Args:
            filename: File name

        Returns:
            Image array data
        """
        with open(filename, "rb") as fdesc:
            cols = int(np.fromfile(fdesc, dtype=np.uint16, count=1)[0])
            rows = int(np.fromfile(fdesc, dtype=np.uint16, count=1)[0])
            arr = np.fromfile(fdesc, dtype=np.uint16, count=cols * rows)
            arr = arr.reshape((rows, cols))
        return np.fliplr(arr)


class FTLabImageFormat(SingleImageFormatBase):
    """FT-Lab image file."""

    FORMAT_INFO = FormatInfo(
        name="FT-Lab",
        extensions="*.ima",
        readable=True,
        writeable=False,
    )

    @staticmethod
    def read_data(filename: str) -> np.ndarray:
        """Read and return data.

        Args:
            filename: Path to FT-Lab file.

        Returns:
            Image data.
        """
        return ftlab.imread_ftlabima(filename)
