import unittest

import craco


class VersionTestCase(unittest.TestCase):
    """ Version tests """

    def test_version(self):
        """ check craco exposes a version attribute """
        self.assertTrue(hasattr(craco, "__version__"))
        self.assertIsInstance(craco.__version__, str)


if __name__ == "__main__":
    unittest.main()
