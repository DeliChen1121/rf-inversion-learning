"""Fetch the exact upstream source; never download model weights or credentials."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = json.loads((ROOT / 'source.json').read_text())
    target = ROOT / 'pipeline_flux_rf_inversion.py'
    if target.exists():
        data = target.read_bytes()
    else:
        with urlopen(source['url'], timeout=60) as response:
            data = response.read()
    if hashlib.sha256(data).hexdigest() != source['sha256']:
        raise SystemExit('Pipeline hash mismatch. Existing files were not overwritten.')
    if not target.exists():
        # Exclusive creation also protects against another process creating it.
        with target.open('xb') as handle:
            handle.write(data)
    print('Verified:', target)


if __name__ == '__main__':
    main()
