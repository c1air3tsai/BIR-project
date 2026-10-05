Additional trusted CA certificates (*.crt or *.pem, PEM format) go here.
Windows: run prepare_certs.ps1 before docker compose up --build.
The script exports public certificates from Windows trusted root stores.
No private keys are exported, and TLS certificate/hostname verification stays enabled.
