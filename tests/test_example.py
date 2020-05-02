import unittest

import subprocess

class TestExample(unittest.TestCase):
    def test_all_example(self):
        res = subprocess.run(["python", "examples/machine_learning.py"])


if __name__ == '__main__':
    unittest.main()
