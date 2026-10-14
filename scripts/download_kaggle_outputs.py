"""Download private notebook outputs through the SDK's resumable streaming path."""

from __future__ import annotations

import argparse
from pathlib import Path

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest


def download(kernel, destination):
    api = KaggleApi()
    api.authenticate()
    owner, slug, _ = api.parse_kernel_string(kernel)
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    files, logs, token = [], [], None
    with api.build_kaggle_client() as client:
        while True:
            request = ApiListKernelSessionOutputRequest()
            request.user_name, request.kernel_slug = owner, slug
            api._set_paging(request, 200, token)
            response = client.kernels.kernels_api_client.list_kernel_session_output(request)
            files.extend(response.files or [])
            if response.log:
                logs.append(response.log)
            token = response.next_page_token
            if not token:
                break
    # Small manifests/configuration arrive first; large weights never enter response.content.
    files.sort(key=lambda item: item.file_name.endswith(".safetensors"))
    print(f"Listed {len(files)} outputs", flush=True)
    with requests.Session() as http:
        http.mount("https://", requests.adapters.HTTPAdapter(max_retries=3))
        for item in files:
            target = (destination / item.file_name).resolve()
            if not target.is_relative_to(destination):
                raise ValueError("Remote output path escapes the destination")
            with http.get(item.url, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                if not api.download_needed(response, str(target), quiet=True):
                    continue
                partial = target.with_name(target.name + ".part")
                print(f"Downloading {item.file_name}", flush=True)
                api.download_file(response, str(partial), http, quiet=True, resume=True,
                                  chunk_size=1024 * 1024, timeout=120)
                partial.replace(target)
                print(f"Saved {item.file_name}: {target.stat().st_size} bytes", flush=True)
    if logs:
        (destination / f"{slug}.log").write_text("\n".join(logs), encoding="utf-8")
    print(f"Complete: {len(files)} output files", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        download(args.kernel, args.output)
    except Exception as error:
        # Signed download URLs must not be exposed through HTTP exception tracebacks.
        parser.exit(1, f"Download failed: {type(error).__name__}\n")
