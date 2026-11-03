"""Working-size and output-shape contracts without GPU or model downloads."""
import importlib.util
from pathlib import Path
import unittest

from PIL import Image, ImageDraw

spec = importlib.util.spec_from_file_location('fill_size', Path(__file__).resolve().parents[1] / 'backend/fill_size.py')
size_policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(size_policy)


class FillSizeTests(unittest.TestCase):
    def test_native_limit_requires_a_complete_ascii_integer(self):
        self.assertEqual(size_policy.parse_max_side('0'), 0)
        self.assertEqual(size_policy.parse_max_side('640'), 640)
        for value in ('', '-1', '+640', '6_40', '640.0', ' 640 ', '６４０'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                size_policy.parse_max_side(value)

    def test_uncapped_preserves_existing_objects_and_dimensions(self):
        image = Image.new('RGB', (960, 544))
        mask = Image.new('L', image.size)
        source, region = size_policy.prepare_fill_input(image, mask, 0)
        self.assertIs(source, image)
        self.assertIs(region, mask)

    def test_bounded_working_images_keep_binary_mask(self):
        for dimensions in ((1920, 1080), (1080, 1920), (960, 544), (640, 640)):
            with self.subTest(dimensions=dimensions):
                image = Image.new('RGB', dimensions)
                mask = Image.new('L', dimensions)
                ImageDraw.Draw(mask).rectangle((0, 0, dimensions[0] // 2, dimensions[1] // 2), fill=255)
                source, region = size_policy.prepare_fill_input(image, mask, 640)
                self.assertEqual(source.size, region.size)
                self.assertLessEqual(max(source.size), 640)
                self.assertTrue(all(value % 64 == 0 for value in source.size))
                self.assertEqual(set(region.tobytes()), {0, 255})
                self.assertEqual(size_policy.restore_fill_output(source, image.size).size, image.size)

    def test_invalid_or_mismatched_inputs_fail(self):
        image = Image.new('RGB', (640, 640))
        for limit in (-1, 1, 63, 641):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                size_policy.prepare_fill_input(image, Image.new('L', image.size), limit)
        with self.assertRaises(ValueError):
            size_policy.prepare_fill_input(image, Image.new('L', (64, 64)), 640)


if __name__ == '__main__':
    unittest.main()
