# A fix for the the problem when there are bad certificates in the Windows CA store.
# See discussion on https://github.com/pystatgen/sgkit/issues/733 and linked issues.
# TODO: remove this when fixed upstream https://github.com/pystatgen/sgkit/issues/737

import ssl


def _ssl_load_windows_store_certs(self, storename, purpose):  # type: ignore[no-untyped-def] # pragma: no cover
    # Code adapted from _load_windows_store_certs in https://github.com/python/cpython/blob/main/Lib/ssl.py
    try:
        certs = [
            cert
            for cert, encoding, trust in ssl.enum_certificates(storename)  # type: ignore
            if encoding == "x509_asn" and (trust is True or purpose.oid in trust)
        ]
    except PermissionError:
        return
    for cert in certs:
        try:
            self.load_verify_locations(cadata=cert)
        except ssl.SSLError:
            pass


ssl.SSLContext._load_windows_store_certs = _ssl_load_windows_store_certs  # type: ignore
