"""Certificate verification and restart recovery for the actual import client."""
import io
import shutil
import ssl
import subprocess
import tempfile
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest import skipUnless
from unittest.mock import patch

from django.core.management import call_command
from django.test import SimpleTestCase, TestCase

from . import pmc_client
from .models import Document, ImportJob
from .forms import get_or_create_topic


class NetworkErrorTests(SimpleTestCase):
    def setUp(self):
        pmc_client._ssl_context.cache_clear()
        pmc_client._last_request = 0
        self.addCleanup(pmc_client._ssl_context.cache_clear)

    def test_certificate_error_reports_recovery_without_retrying(self):
        error = ssl.SSLCertVerificationError(1, 'CERTIFICATE_VERIFY_FAILED: self signed certificate in certificate chain')
        with patch('search.pmc_client.urllib.request.urlopen', side_effect=urllib.error.URLError(error)) as request, patch('search.pmc_client.time.sleep') as sleep:
            with self.assertRaisesRegex(RuntimeError, 'prepare_certs.ps1'):
                pmc_client._get('https://example.invalid/')
        self.assertEqual(request.call_count, 1)
        sleep.assert_not_called()
        context = request.call_args.kwargs['context']
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(context.check_hostname)

    def test_direct_certificate_error_is_also_specific(self):
        with patch('search.pmc_client.urllib.request.urlopen', side_effect=ssl.SSLCertVerificationError(1, 'certificate verify failed')):
            with self.assertRaisesRegex(RuntimeError, 'SSL certificate verification failed'):
                pmc_client._get('https://example.invalid/')

    def test_temporary_server_error_retries_and_returns_response(self):
        response = io.BytesIO(b'abstract XML')
        error = urllib.error.HTTPError('https://example.invalid/', 503, 'busy', {}, None)
        with patch('search.pmc_client.urllib.request.urlopen', side_effect=[error, response]) as request, patch('search.pmc_client.time.sleep'):
            self.assertEqual(pmc_client._get('https://example.invalid/'), b'abstract XML')
        self.assertEqual(request.call_count, 2)

    def test_permanent_network_error_preserves_reason(self):
        with patch('search.pmc_client.urllib.request.urlopen', side_effect=urllib.error.URLError('proxy unreachable')) as request, patch('search.pmc_client.time.sleep'):
            with self.assertRaisesRegex(RuntimeError, 'proxy unreachable'):
                pmc_client._get('https://example.invalid/')
        self.assertEqual(request.call_count, 4)

    def test_invalid_custom_ca_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'invalid.crt').write_text('not a certificate')
            with self.settings(NCBI_CA_DIR=directory, NCBI_CA_BUNDLE=''):
                with self.assertRaisesRegex(RuntimeError, 'invalid.crt.*PEM CA certificate'):
                    pmc_client._ssl_context()

    def test_truncated_search_response_is_retryable(self):
        with patch('search.pmc_client._get', return_value=b'{'):
            with self.assertRaises(pmc_client.NCBITransientError):
                pmc_client.search_pubmed('GLP-1')

    def test_truncated_batch_response_is_retryable(self):
        with patch('search.pmc_client._get', return_value=b'<PubmedArticleSet>'):
            with self.assertRaises(pmc_client.NCBITransientError):
                pmc_client.fetch_pubmed_batch(['123'])

    def test_empty_or_book_only_batches_do_not_abort_import(self):
        for xml in (b'<PubmedArticleSet/>', b'<PubmedArticleSet><PubmedBookArticle/></PubmedArticleSet>'):
            with self.subTest(xml=xml), patch('search.pmc_client._get', return_value=xml):
                self.assertEqual(pmc_client.fetch_pubmed_batch(['123']), [])


@skipUnless(shutil.which('openssl'), 'The optional local TLS integration check needs OpenSSL.')
class TrustedTLSIntegrationTests(SimpleTestCase):
    def test_private_ca_is_required_and_hostname_is_verified(self):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'trusted abstract')

            def log_message(self, *args):
                pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trusted = root / 'trusted'
            trusted.mkdir()
            empty = root / 'empty'
            empty.mkdir()
            extension = root / 'extensions.txt'
            extension.write_text('subjectAltName=DNS:localhost\nbasicConstraints=CA:FALSE\nkeyUsage=digitalSignature,keyEncipherment\nextendedKeyUsage=serverAuth\n')
            commands = [
                ['req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', '1', '-subj', '/CN=BioMedIR disposable test CA', '-addext', 'basicConstraints=critical,CA:TRUE', '-keyout', 'ca.key', '-out', str(trusted / 'test-root.crt')],
                ['req', '-newkey', 'rsa:2048', '-nodes', '-subj', '/CN=localhost', '-keyout', 'server.key', '-out', 'server.csr'],
                ['x509', '-req', '-in', 'server.csr', '-CA', str(trusted / 'test-root.crt'), '-CAkey', 'ca.key', '-CAcreateserial', '-days', '1', '-extfile', str(extension), '-out', 'server.crt'],
            ]
            for arguments in commands:
                subprocess.run(['openssl', *arguments], cwd=root, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
            server = HTTPServer(('127.0.0.1', 0), Handler)
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(str(root / 'server.crt'), str(root / 'server.key'))
            server.socket = context.wrap_socket(server.socket, server_side=True)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                port = server.server_address[1]
                with self.settings(NCBI_CA_DIR=str(empty), NCBI_CA_BUNDLE=''):
                    pmc_client._ssl_context.cache_clear()
                    with self.assertRaisesRegex(RuntimeError, 'SSL certificate verification failed'):
                        pmc_client._get(f'https://localhost:{port}/', timeout=5)
                with self.settings(NCBI_CA_DIR=str(trusted), NCBI_CA_BUNDLE=''):
                    pmc_client._ssl_context.cache_clear()
                    self.assertEqual(pmc_client._get(f'https://localhost:{port}/', timeout=5), b'trusted abstract')
                    with self.assertRaisesRegex(RuntimeError, 'SSL certificate verification failed'):
                        pmc_client._get(f'https://127.0.0.1:{port}/', timeout=5)
            finally:
                pmc_client._ssl_context.cache_clear()
                server.shutdown()
                thread.join(timeout=5)
                server.server_close()


class ContainerRecoveryTests(TestCase):
    def test_restart_keeps_articles_and_recovers_incomplete_job(self):
        topic = get_or_create_topic('GLP-1')
        document = Document.objects.create(title='Retained article', abstract='Retained abstract.', raw_text='Retained abstract.', source_file='existing.xml')
        document.topics.add(topic)
        job = ImportJob.objects.create(topic=topic, requested=1000, status='running', imported=12, offset=50)
        call_command('prepare_container', stdout=io.StringIO())
        job.refresh_from_db()
        self.assertEqual(job.status, 'interrupted')
        self.assertEqual((job.imported, job.offset), (12, 50))
        self.assertEqual(Document.objects.get().pk, document.pk)
        self.assertEqual(document.topics.get().pk, topic.pk)
        call_command('prepare_container', stdout=io.StringIO())
        self.assertEqual(Document.objects.count(), 1)

    def test_restart_removes_old_bundled_samples_only(self):
        sample = Document.objects.create(title='Sample', abstract='x', raw_text='x', source_file='39932906.xml')
        retained = Document.objects.create(title='Uploaded', abstract='y', raw_text='y', source_file='uploaded.xml')
        call_command('prepare_container', stdout=io.StringIO())
        self.assertFalse(Document.objects.filter(pk=sample.pk).exists())
        self.assertTrue(Document.objects.filter(pk=retained.pk).exists())
