"""Provide loopback HTTP links for the two closed-chain tutorial documents."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


DOCUMENTS = Path(__file__).resolve().parents[1] / 'docs'
FILES = {
    '/closed-chain-tutorial-v3.pdf': (
        DOCUMENTS / '开链转闭链图文教程_v3.pdf', 'application/pdf', 'inline'),
    '/closed-chain-tutorial-v3.docx': (
        DOCUMENTS / '开链转闭链图文教程_v3.docx',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'attachment'),
    '/training-record.html': (
        DOCUMENTS / 'training_record_latest.html', 'text/html; charset=utf-8', 'inline'),
}


class Handler(BaseHTTPRequestHandler):
    def send_document(self, head=False):
        route = urlsplit(self.path).path
        if route not in FILES or not FILES[route][0].is_file():
            self.send_error(404)
            return
        path, mime, disposition = FILES[route]
        with path.open('rb') as stream:
            self.send_response(200)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(path.stat().st_size))
            self.send_header('Content-Disposition',
                             f'{disposition}; filename="{route[1:]}"')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            if not head:
                while chunk := stream.read(65536):
                    self.wfile.write(chunk)

    def do_GET(self):
        self.send_document()

    def do_HEAD(self):
        self.send_document(head=True)


if __name__ == '__main__':
    ThreadingHTTPServer(('127.0.0.1', 8766), Handler).serve_forever()
