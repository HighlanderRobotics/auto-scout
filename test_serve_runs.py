import tempfile
import threading
import unittest
import urllib.request
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path

from serve_runs import RunHandler,parse_range


class RunServerTests(unittest.TestCase):
    def test_range_bounds_and_suffixes(self):
        self.assertEqual(parse_range('bytes=3-6',10),(3,6))
        self.assertEqual(parse_range('bytes=7-',10),(7,9))
        self.assertEqual(parse_range('bytes=-4',10),(6,9))
        for value in ['bytes=20-','bytes=5-3','bytes=-0','bytes=0-1,3-4']:
            with self.assertRaises(ValueError):parse_range(value,10)

    def test_video_request_returns_requested_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory,'video.mp4').write_bytes(b'0123456789')
            server=ThreadingHTTPServer(('127.0.0.1',0),partial(RunHandler,directory=directory))
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            try:
                request=urllib.request.Request(f'http://127.0.0.1:{server.server_port}/video.mp4',headers={'Range':'bytes=3-6'})
                with urllib.request.urlopen(request) as response:
                    self.assertEqual(response.status,206)
                    self.assertEqual(response.headers['Content-Range'],'bytes 3-6/10')
                    self.assertEqual(response.read(),b'3456')
            finally:server.shutdown();server.server_close();thread.join()


if __name__=='__main__':unittest.main()
