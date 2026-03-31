#!/usr/bin/env python3
"""
Fix metadata on existing Zenodo records to match .zenodo.json.

Reads the canonical metadata from .zenodo.json, fetches all versions linked to
the concept record, and updates each one (except the already-correct v0.6.0)
with the correct creators, contributors, keywords, description, and license.

Version-specific release notes from CHANGELOG.md are appended to the base
project description for each record.

Usage:
    export ZENODO_TOKEN="your-personal-access-token"
    python3 scripts/fix_zenodo_records.py [--dry-run]

Get a token at: https://zenodo.org/account/settings/applications/
Required scopes: deposit:write, deposit:actions
"""

import argparse
import html
import json
import os
import re
import sys
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError


API = "https://zenodo.org/api"
SKIP_VERSION = "v0.6.0"  # already correct

# Map .zenodo.json license values to Zenodo deposit API license IDs
LICENSE_MAP = {
    "MIT": "mit",
    "mit": "mit",
}


def api_request(method, path, token, data=None):
    """Make an authenticated Zenodo API request."""
    url = f"{API}{path}" if path.startswith("/") else path
    headers = {"Authorization": f"Bearer {token}"}
    body = None
    if data is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(data).encode()

    req = Request(url, data=body, headers=headers, method=method)
    try:
        with urlopen(req) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else {}
    except HTTPError as e:
        error_body = e.read().decode()
        print(f"  HTTP {e.code}: {error_body}", file=sys.stderr)
        raise


def get_all_versions(token, concept_recid):
    """Fetch all version records for a concept."""
    records = api_request(
        "GET",
        f"/records?q=conceptrecid:{concept_recid}&all_versions=true&size=50&sort=version",
        token,
    )
    return records["hits"]["hits"]


def extract_changelog_html(changelog_path, version):
    """Extract a version's changelog section and convert to HTML."""
    with open(changelog_path) as f:
        changelog = f.read()

    # Strip leading 'v' if present for matching
    ver = version.lstrip("v")
    pattern = rf"## \[{re.escape(ver)}\][^\n]*\n(.*?)(?=\n## \[|\Z)"
    match = re.search(pattern, changelog, re.DOTALL)
    if not match:
        return None

    notes = match.group(1).strip()
    lines = notes.split("\n")
    html_parts = []
    in_list = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("### "):
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            html_parts.append(f"<h3>{html.escape(stripped[4:])}</h3>")
        elif stripped.startswith("- "):
            if not in_list:
                html_parts.append("<ul>")
                in_list = True
            html_parts.append(f"<li>{html.escape(stripped[2:])}</li>")
        elif stripped:
            html_parts.append(f"<p>{html.escape(stripped)}</p>")

    if in_list:
        html_parts.append("</ul>")

    return "\n".join(html_parts)


def build_description(base_description, version, changelog_path):
    """Combine base project description with version-specific release notes."""
    release_html = extract_changelog_html(changelog_path, version)
    if release_html:
        ver = version.lstrip("v")
        return (
            base_description
            + f"\n<hr>\n<h2>v{html.escape(ver)} Release Notes</h2>\n"
            + release_html
        )
    return base_description


def build_title(base_title, version):
    """Build a versioned title like 'guv-calcs v0.7.0: subtitle'."""
    ver = version.lstrip("v")
    # Split on first colon to get "guv-calcs" and the subtitle
    if ":" in base_title:
        name, subtitle = base_title.split(":", 1)
        return f"{name.strip()} v{ver}:{subtitle}"
    return f"{base_title} v{ver}"


def update_record(token, record_id, version, zenodo_meta, changelog_path, dry_run=False):
    """Update a single Zenodo record's metadata."""
    title = build_title(zenodo_meta["title"], version)
    description = build_description(
        zenodo_meta["description"], version, changelog_path
    )

    # Map license to deposit API format
    raw_license = zenodo_meta.get("license", "MIT")
    license_id = LICENSE_MAP.get(raw_license, raw_license.lower())

    new_metadata = {
        "metadata": {
            "title": title,
            "description": description,
            "upload_type": "software",
            "creators": zenodo_meta["creators"],
            "contributors": zenodo_meta.get("contributors", []),
            "keywords": zenodo_meta.get("keywords", []),
            "license": license_id,
            "access_right": zenodo_meta.get("access_right", "open"),
        }
    }

    if dry_run:
        print(f"  [DRY RUN] Would update with:")
        print(f"    title: {title}")
        print(f"    creators: {[c['name'] for c in new_metadata['metadata']['creators']]}")
        print(f"    contributors: {[c['name'] for c in new_metadata['metadata']['contributors']]}")
        print(f"    keywords: {len(new_metadata['metadata']['keywords'])} keywords")
        has_notes = "Release Notes" in description
        print(f"    description: base + {'release notes' if has_notes else 'NO release notes found'}")
        print(f"    license: {license_id}")
        return True

    # Step 1: Unlock the published record for editing
    print(f"  Unlocking record for editing...")
    api_request("POST", f"/deposit/depositions/{record_id}/actions/edit", token)

    # Step 2: Update metadata
    print(f"  Updating metadata...")
    api_request("PUT", f"/deposit/depositions/{record_id}", token, data=new_metadata)

    # Step 3: Re-publish
    print(f"  Publishing...")
    api_request("POST", f"/deposit/depositions/{record_id}/actions/publish", token)

    return True


def main():
    parser = argparse.ArgumentParser(description="Fix Zenodo record metadata")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--only",
        help="Only update this version (e.g. v0.6.1) or record ID (e.g. 18573616)",
    )
    parser.add_argument(
        "--set-version",
        help="Override the version string (for records missing it, e.g. --set-version v0.5.0.1)",
    )
    args = parser.parse_args()

    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        print("Error: Set ZENODO_TOKEN environment variable", file=sys.stderr)
        print(
            "Get one at: https://zenodo.org/account/settings/applications/",
            file=sys.stderr,
        )
        print("Required scopes: deposit:write, deposit:actions", file=sys.stderr)
        sys.exit(1)

    # Load canonical metadata
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    zenodo_path = os.path.join(repo_root, ".zenodo.json")
    changelog_path = os.path.join(repo_root, "CHANGELOG.md")

    with open(zenodo_path) as f:
        zenodo_meta = json.load(f)

    # Fetch all versions
    concept_recid = "18573615"
    print(f"Fetching all versions for concept {concept_recid}...")
    records = get_all_versions(token, concept_recid)
    print(f"Found {len(records)} versions\n")

    success = 0
    skipped = 0
    failed = 0

    for record in records:
        rec_id = record["id"]
        version = record["metadata"].get("version")
        # Fallback: extract version from related_identifiers tag URL or title
        if not version:
            for ri in record["metadata"].get("related_identifiers", []):
                m = re.search(r"/tree/(v[\d.]+)", ri.get("identifier", ""))
                if m:
                    version = m.group(1)
                    break
        if not version:
            title = record["metadata"].get("title", "")
            m = re.search(r"(v[\d.]+)", title)
            if m:
                version = m.group(1)
        if not version:
            version = "unknown"
        doi = record.get("doi", "no DOI")

        # Apply version override (by record ID match) before any filtering
        if args.set_version and str(rec_id) == args.only:
            version = args.set_version

        if version == SKIP_VERSION and not (args.only and (version == args.only or str(rec_id) == args.only)):
            print(f"[SKIP] {version} (record {rec_id}) — already correct")
            skipped += 1
            continue

        if args.only and version != args.only and str(rec_id) != args.only:
            print(f"[SKIP] {version} (record {rec_id}) — not selected")
            skipped += 1
            continue

        print(f"[UPDATE] {version} (record {rec_id}, DOI: {doi})")
        try:
            update_record(
                token, rec_id, version, zenodo_meta, changelog_path, dry_run=args.dry_run
            )
            success += 1
            print(f"  Done!\n")
            if not args.dry_run:
                time.sleep(1)  # be polite to the API
        except Exception as e:
            print(f"  FAILED: {e}\n", file=sys.stderr)
            failed += 1

    print(f"\nSummary: {success} updated, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
