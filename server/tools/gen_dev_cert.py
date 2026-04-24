"""Generate a self-signed TLS cert + key for local development.

Browsers require a 'secure context' (HTTPS or localhost) for camera + IMU APIs.
When you open the PWA from your phone over the LAN, you need HTTPS.

This generates a cert valid for: localhost, 127.0.0.1, and all detected LAN IPs.
The cert lives in server/data/dev_cert/ (gitignored).

Usage:
    python tools/gen_dev_cert.py
"""
from __future__ import annotations

import datetime as dt
import ipaddress
import socket
import sys
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "dev_cert"
KEY_PATH = OUT_DIR / "key.pem"
CERT_PATH = OUT_DIR / "cert.pem"


def detect_lan_ips() -> list[str]:
    ips: set[str] = {"127.0.0.1"}
    try:
        ips.add(socket.gethostbyname(socket.gethostname()))
    except Exception:
        pass
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ips.add(s.getsockname()[0])
    except Exception:
        pass
    finally:
        s.close()
    return sorted(ips)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ips = detect_lan_ips()
    print(f"detected LAN IPs: {ips}")
    print(f"writing self-signed cert to {OUT_DIR}")

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    KEY_PATH.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )

    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "Road Anomaly Analysis (dev)"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "UGP_amarb"),
    ])

    san_entries: list[x509.GeneralName] = [x509.DNSName("localhost")]
    for ip in ips:
        try:
            san_entries.append(x509.IPAddress(ipaddress.ip_address(ip)))
        except ValueError:
            pass

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=5))
        .not_valid_after(dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=365))
        .add_extension(x509.SubjectAlternativeName(san_entries), critical=False)
        .add_extension(
            x509.BasicConstraints(ca=False, path_length=None), critical=True
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=True,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .sign(key, hashes.SHA256())
    )
    CERT_PATH.write_bytes(cert.public_bytes(serialization.Encoding.PEM))

    print()
    print("Open the PWA from your phone using one of:")
    for ip in ips:
        if ip != "127.0.0.1":
            print(f"  https://{ip}:8443/app/")
    print()
    print("First time you'll see a browser cert warning — tap 'Advanced' -> 'Proceed'.")
    print("This is normal for a self-signed dev cert.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
