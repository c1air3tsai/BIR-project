# Export only public CA certificates already trusted by this Windows account/machine.
# Nothing is installed into Windows, and no private key is exported.
$ErrorActionPreference = 'Stop'
if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) {
    throw 'This helper is for Windows. On other systems, put your trusted proxy CA in certs/.'
}
$certificateDirectory = Join-Path $PSScriptRoot 'certs'
New-Item -ItemType Directory -Force -Path $certificateDirectory | Out-Null
$now = Get-Date
$certificates = @(Get-ChildItem -Path 'Cert:\CurrentUser\Root', 'Cert:\LocalMachine\Root' |
    Where-Object { $_.NotBefore -le $now -and $_.NotAfter -gt $now } |
    Sort-Object -Property Thumbprint -Unique)
if ($certificates.Count -eq 0) {
    throw 'No valid trusted Windows root certificates were found.'
}
Get-ChildItem -LiteralPath $certificateDirectory -Filter 'windows-trusted-*.crt' |
    Remove-Item -Force
foreach ($certificate in $certificates) {
    $body = [Convert]::ToBase64String($certificate.RawData, [Base64FormattingOptions]::InsertLineBreaks)
    $pem = "-----BEGIN CERTIFICATE-----`r`n$body`r`n-----END CERTIFICATE-----`r`n"
    $path = Join-Path $certificateDirectory ("windows-trusted-" + $certificate.Thumbprint + '.crt')
    [IO.File]::WriteAllText($path, $pem, [Text.Encoding]::ASCII)
}
Write-Host ("Prepared " + $certificates.Count + " trusted public CA certificates in certs/.")
Write-Host 'Next: docker compose up --build'
