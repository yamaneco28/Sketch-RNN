import unittest

import time

import sys
sys.path.append('.')
sys.path.append('..')
from scripts.print_progress_bar import print_progress_bar


class TestPrintProgressBar(unittest.TestCase):
    def test_print_progress_bar(self):
        for i, _ in enumerate(range(100)):
            time.sleep(0.05)
            header = 'epoch: {}'.format(0)
            print_progress_bar(i=i, length=100, header=header, end='')
        print('\r\033[K' + 'finish')


if __name__ == "__main__":
    unittest.main()
