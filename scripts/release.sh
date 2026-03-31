#!/usr/bin/env bash
set -euo pipefail

VERSION_FILE="src/guv_calcs/_version.py"
CHANGELOG="CHANGELOG.md"

die() { echo "Error: $*" >&2; exit 1; }

# --- Read current version ---
current=$(grep -oP '(?<=__version__ = ")[^"]+' "$VERSION_FILE")
IFS='.' read -r major minor patch <<< "$current"

# --- Determine new version ---
case "${1:-}" in
    major) new_version="$((major + 1)).0.0" ;;
    minor) new_version="$major.$((minor + 1)).0" ;;
    patch) new_version="$major.$minor.$((patch + 1))" ;;
    [0-9]*) new_version="$1" ;;
    *)
        echo "Usage: $0 <major|minor|patch|X.Y.Z>"
        echo ""
        echo "Current version: $current"
        echo ""
        echo "Examples:"
        echo "  $0 patch   # $current -> $major.$minor.$((patch + 1))"
        echo "  $0 minor   # $current -> $major.$((minor + 1)).0"
        echo "  $0 major   # $current -> $((major + 1)).0.0"
        echo "  $0 1.0.0   # $current -> 1.0.0"
        exit 1
        ;;
esac

echo "Releasing: $current -> $new_version"

# --- Preflight checks ---
git diff --quiet && git diff --cached --quiet \
    || die "Working tree is dirty. Commit or stash changes first."

branch=$(git rev-parse --abbrev-ref HEAD)
[ "$branch" = "main" ] || die "Must be on main branch (currently on $branch)"

grep -q "## \[Unreleased\]" "$CHANGELOG" \
    || die "CHANGELOG.md missing [Unreleased] section"

# Check that Unreleased section has content
unreleased_content=$(sed -n '/## \[Unreleased\]/,/## \[/{/## \[/d; /^$/d; p;}' "$CHANGELOG")
[ -n "$unreleased_content" ] \
    || die "No content in [Unreleased] section. Add changelog entries first."

# --- Update version file ---
sed -i "s/__version__ = \".*\"/__version__ = \"$new_version\"/" "$VERSION_FILE"

# --- Update changelog ---
today=$(date +%Y-%m-%d)
sed -i "s/## \[Unreleased\]/## [Unreleased]\n\n## [$new_version] - $today/" "$CHANGELOG"

# --- Update .zenodo.json description with release notes ---
ZENODO_FILE=".zenodo.json"
if [ -f "$ZENODO_FILE" ]; then
    # Extract this version's changelog as HTML and inject into .zenodo.json
    python3 -c "
import json, re, html

# Read the changelog section for this version
version = '$new_version'
with open('$CHANGELOG') as f:
    changelog = f.read()

# Extract section between this version's header and the next
pattern = rf'## \[{re.escape(version)}\][^\n]*\n(.*?)(?=\n## \[|\Z)'
match = re.search(pattern, changelog, re.DOTALL)
if not match:
    print(f'Warning: no changelog section found for {version}, skipping .zenodo.json update')
    exit(0)

notes = match.group(1).strip()

# Convert markdown changelog to simple HTML
lines = notes.split('\n')
html_parts = []
in_list = False
for line in lines:
    stripped = line.strip()
    if stripped.startswith('### '):
        if in_list:
            html_parts.append('</ul>')
            in_list = False
        html_parts.append(f'<h3>{html.escape(stripped[4:])}</h3>')
    elif stripped.startswith('- '):
        if not in_list:
            html_parts.append('<ul>')
            in_list = True
        html_parts.append(f'<li>{html.escape(stripped[2:])}</li>')
    elif stripped:
        html_parts.append(f'<p>{html.escape(stripped)}</p>')
if in_list:
    html_parts.append('</ul>')

release_html = '\n'.join(html_parts)

# Update .zenodo.json
with open('$ZENODO_FILE') as f:
    data = json.load(f)

# Build versioned title: 'guv-calcs v0.8.0: subtitle'
base_title = data['title']
if ':' in base_title:
    name, subtitle = base_title.split(':', 1)
    data['title'] = f'{name.strip()} v{version}:{subtitle}'
else:
    data['title'] = f'{base_title} v{version}'

base_desc = data['description']
data['description'] = base_desc + f'\n<hr>\n<h2>v{html.escape(version)} Release Notes</h2>\n' + release_html

with open('$ZENODO_FILE', 'w') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
    f.write('\n')

print(f'Updated .zenodo.json with v{version} title and release notes')
"
    git add "$ZENODO_FILE"
fi

# --- Commit, tag, push ---
git add "$VERSION_FILE" "$CHANGELOG"
git commit -m "Release v$new_version"
git tag -a "v$new_version" -m "Release v$new_version"

# Restore .zenodo.json to base description for next release
if [ -f "$ZENODO_FILE" ]; then
    git checkout HEAD~1 -- "$ZENODO_FILE"
    git commit -m "Restore .zenodo.json base description"
fi

git push origin main
git push origin "v$new_version"

echo ""
echo "Released v$new_version"
echo "Tagged and pushed v$new_version. Now uploading to PyPI and creating GitHub release..."
