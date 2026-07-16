from __future__ import annotations

import pathlib
import tempfile
import unittest

import numpy as np
from plyfile import PlyData, PlyElement

from backend.gsplat_training_worker import SPLAT_ROW_BYTES, convert_ply_to_splat


class SplatExportTest(unittest.TestCase):
    def test_ply_fields_are_encoded_in_viewer_32_byte_layout(self) -> None:
        vertex_dtype = [(name, "f4") for name in (
            "x", "y", "z", "scale_0", "scale_1", "scale_2",
            "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
            "rot_0", "rot_1", "rot_2", "rot_3",
        )]
        vertex = np.zeros(1, dtype=vertex_dtype)
        vertex["x"], vertex["y"], vertex["z"] = 1.0, 2.0, 3.0
        vertex["scale_1"], vertex["scale_2"] = np.log(2.0), np.log(3.0)
        vertex["rot_0"] = 1.0

        with tempfile.TemporaryDirectory() as directory:
            source = pathlib.Path(directory) / "source.ply"
            output = pathlib.Path(directory) / "result.splat"
            PlyData([PlyElement.describe(vertex, "vertex")]).write(source)
            metadata = convert_ply_to_splat(source, output)

            self.assertEqual({"numGaussians": 1, "splatBytes": SPLAT_ROW_BYTES}, metadata)
            row_dtype = np.dtype([
                ("xyz", "<f4", (3,)),
                ("scale", "<f4", (3,)),
                ("rgba", "u1", (4,)),
                ("rotation", "u1", (4,)),
            ])
            row = np.fromfile(output, dtype=row_dtype)[0]
            np.testing.assert_allclose(row["xyz"], [1.0, 2.0, 3.0])
            np.testing.assert_allclose(row["scale"], [1.0, 2.0, 3.0], rtol=1e-6)
            np.testing.assert_array_equal(row["rgba"], [128, 128, 128, 128])
            np.testing.assert_array_equal(row["rotation"], [255, 128, 128, 128])


if __name__ == "__main__":
    unittest.main()
