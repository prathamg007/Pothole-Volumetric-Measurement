"""Launch the FastAPI server over HTTPS using the self-signed dev cert.

Run `python tools/gen_dev_cert.py` once first to create the cert.
"""
from pathlib import Path

import uvicorn

CERT_DIR = Path(__file__).resolve().parent / "data" / "dev_cert"


def main():
    keyfile = CERT_DIR / "key.pem"
    certfile = CERT_DIR / "cert.pem"
    if not (keyfile.exists() and certfile.exists()):
        raise SystemExit(
            "Dev cert not found. Generate one first:\n  python tools/gen_dev_cert.py"
        )
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8443,
        reload=False,
        ssl_keyfile=str(keyfile),
        ssl_certfile=str(certfile),
    )


if __name__ == "__main__":
    main()
