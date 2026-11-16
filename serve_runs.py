"""Serve local run artifacts with byte-range support for video seeking."""
import argparse
import os
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def parse_range(value, size):
    if not value.startswith('bytes=') or ',' in value or size<=0:
        raise ValueError('Unsupported range')
    left,right=value[6:].split('-',1)
    if not left:
        suffix=int(right)
        if suffix<=0:raise ValueError('Invalid suffix')
        return max(0,size-suffix),size-1
    start=int(left);end=min(size-1,int(right)) if right else size-1
    if not 0<=start<=end<size:raise ValueError('Unsatisfiable range')
    return start,end


class RunHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        self.byte_range=None
        path=Path(self.translate_path(self.path))
        if not path.is_file():return super().send_head()
        try:
            source=path.open('rb');stat=os.fstat(source.fileno());size=stat.st_size
            value=self.headers.get('Range')
            if value:
                try:self.byte_range=parse_range(value,size)
                except (ValueError,TypeError):
                    source.close();self.send_response(416);self.send_header('Content-Range',f'bytes */{size}');self.send_header('Content-Length','0');self.end_headers();return None
            self.send_response(206 if self.byte_range else 200)
            self.send_header('Content-Type',self.guess_type(str(path)))
            self.send_header('Accept-Ranges','bytes')
            if self.byte_range:
                start,end=self.byte_range;self.send_header('Content-Range',f'bytes {start}-{end}/{size}');length=end-start+1
            else:length=size
            self.send_header('Content-Length',str(length));self.send_header('Last-Modified',self.date_time_string(stat.st_mtime));self.end_headers()
            return source
        except OSError:
            self.send_error(404,'File not found');return None

    def copyfile(self,source,outputfile):
        if self.byte_range is None:return super().copyfile(source,outputfile)
        start,end=self.byte_range;source.seek(start);remaining=end-start+1
        while remaining:
            data=source.read(min(65536,remaining))
            if not data:break
            try:outputfile.write(data)
            except (BrokenPipeError,ConnectionResetError):break
            remaining-=len(data)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,default=Path('runs'));p.add_argument('--port',type=int,default=8766)
    args=p.parse_args()
    if not args.directory.is_dir():p.error('Run directory does not exist')
    with ThreadingHTTPServer(('127.0.0.1',args.port),partial(RunHandler,directory=str(args.directory.resolve()))) as server:
        print(f'Local run viewer: http://127.0.0.1:{args.port}/',flush=True);server.serve_forever()


if __name__=='__main__':main()
