"""mitmproxy addon that records the photos shown while voting on Photofeeler and the requests sent when voting.

Run with: mitmdump -s personal_ratings/photofeeler_capture_addon.py

Output (relative to the current directory):
  captured_images/<photo file>     photos as served by photos.photofeeler.com
  captured_events.jsonl            one JSON object per event, in chronological order:
                                     {"type": "image", ...}    a photo was downloaded
                                     {"type": "request", ...}  a non-GET request was sent to photofeeler.com (votes)
The events log is later turned into a ratings CSV once the vote request format is known.
"""
import json
import time
from pathlib import Path

IMAGES_DIRECTORY = Path("captured_images")
EVENTS_FILE = Path("captured_events.jsonl")


def _append_event(event):
    event["time"] = time.time()
    with EVENTS_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event, ensure_ascii=False) + "\n")


def response(flow):
    url = flow.request.url
    host = flow.request.pretty_host

    if url.startswith("https://photos.photofeeler.com/p/") and flow.response.headers.get("content-type", "").startswith("image/"):
        IMAGES_DIRECTORY.mkdir(exist_ok=True)
        filename = url.split("/")[-1].split("?")[0] or f"image_{abs(hash(url))}.jpg"
        path = IMAGES_DIRECTORY / filename
        path.write_bytes(flow.response.content)
        _append_event({"type": "image", "url": url, "path": str(path)})
        print(f"Saved image: {path}")
    elif host.endswith("photofeeler.com") and flow.request.method != "GET":
        _append_event({
            "type": "request",
            "method": flow.request.method,
            "url": url,
            "request_body": flow.request.get_text(strict=False),
            "status": flow.response.status_code,
            "response_body": (flow.response.get_text(strict=False) or "")[:2000],
        })
