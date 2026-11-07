"""Run only after observing the exact successful R14 integration merge."""
import hashlib
import io
import json
import os
import pathlib
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

REPO = 'OVVO-Financial/NNS-python'
MERGE = os.environ['VERIFIED_MERGE_SHA']
HEAD = 'f2bda92aa006503d5f1ee26c8f9085e2bae0ee7c'
CI_RUN = 42402380593
ARTIFACT = 13586100516
ARTIFACT_DIGEST = '91c7136af5ff91f75428e39453367834b8fcd3f3fc49e0f01028d1a896110a2f'
STAGING = 'automation/r14-publish-20261107'
OUT = pathlib.Path('/tmp/r14-retirement-evidence')
OUT.mkdir(exist_ok=True)
TOKEN = os.environ['GH_TOKEN']
assert re.fullmatch('[0-9a-f]{40}', MERGE)

class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        result = super().redirect_request(req, fp, code, msg, headers, newurl)
        if result and urllib.parse.urlsplit(req.full_url).netloc != urllib.parse.urlsplit(newurl).netloc:
            result.remove_header('Authorization')
        return result

opener = urllib.request.build_opener(SafeRedirect())

def request(path, data=None, method=None, missing_ok=False, binary=False):
    req = urllib.request.Request(
        'https://api.github.com/repos/' + REPO + path,
        data=None if data is None else json.dumps(data).encode(),
        headers={'Authorization': 'Bearer ' + TOKEN,
                 'Accept': 'application/vnd.github+json',
                 'X-GitHub-Api-Version': '2022-11-28'},
        method=method or ('GET' if data is None else 'POST'))
    try:
        with opener.open(req, timeout=60) as response:
            raw = response.read()
            if req.get_method() != 'GET':
                time.sleep(1.1)
            return raw if binary else (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as error:
        if error.code == 404 and missing_ok:
            return None
        raise

def pages(path):
    out = []
    sep = '&' if '?' in path else '?'
    for page in range(1, 101):
        batch = request(path + sep + 'per_page=100&page=' + str(page))
        out.extend(batch)
        if len(batch) < 100:
            return out
    raise RuntimeError('Unexpected pagination size')

def ancestor(sha, descendant):
    if subprocess.run(['git', 'cat-file', '-e', sha], capture_output=True).returncode:
        subprocess.run(['git', 'fetch', 'origin', sha], check=True)
    return subprocess.run(['git', 'merge-base', '--is-ancestor', sha, descendant]).returncode == 0

def archive(name, sha):
    path = '/git/ref/tags/' + urllib.parse.quote(name, safe='')
    ref = request(path, missing_ok=True)
    if ref:
        assert ref['object']['sha'] == sha, 'Refusing to move existing archive tag'
    else:
        request('/git/refs', {'ref': 'refs/tags/' + name, 'sha': sha})
    assert request(path)['object']['sha'] == sha

pr192 = request('/pulls/192')
assert pr192['merged'] and pr192['merge_commit_sha'] == MERGE
assert pr192['head']['sha'] == HEAD and pr192['base']['ref'] == 'main'
run = request('/actions/runs/' + str(CI_RUN))
assert run['head_sha'] == HEAD and run['status'] == 'completed' and run['conclusion'] == 'success'
jobs = request('/actions/runs/' + str(CI_RUN) + '/jobs')['jobs']
assert {j['name'] for j in jobs if j['conclusion'] == 'success'} >= {'Python 3.11', 'Python 3.12', 'Python 3.13', 'Python 3.14'}
main = request('/git/ref/heads/main')['object']['sha']
subprocess.run(['git', 'fetch', 'origin', 'main'], check=True)
assert ancestor(MERGE, main) and ancestor(HEAD, main)
raw = request('/actions/artifacts/' + str(ARTIFACT) + '/zip', binary=True)
assert hashlib.sha256(raw).hexdigest() == ARTIFACT_DIGEST
with zipfile.ZipFile(io.BytesIO(raw)) as z:
    inventory = json.loads(z.read('open-pr-inventory.json'))
assert len(inventory) == 44
assert {r['number'] for r in inventory} == (set(range(148, 190)) - {177}) | {190, 191, 192}
receipt = {'merge_sha': MERGE, 'head_sha': HEAD, 'main_sha': main, 'ci_run': CI_RUN,
           'retired': [], 'retained': [], 'deleted_branches': [], 'archives': []}

def save():
    (OUT / 'retirement.json').write_text(json.dumps(receipt, indent=2))

save()
approved = []
# Close dependent PRs before deleting any branch used as their base.
for snapshot in sorted(inventory, key=lambda p: p['number'], reverse=True):
    number = snapshot['number']
    pr = request('/pulls/' + str(number))
    expected = HEAD if number == 192 else snapshot['head_sha']
    if pr['head']['sha'] != expected or (pr['head'].get('repo') or {}).get('full_name') != REPO:
        receipt['retained'].append({'number': number, 'reason': 'head changed since inventory'})
        save()
        continue
    if number == 190:
        changed = pages('/pulls/190/files')
        if not all(f['filename'] in {'sync/nns_source.json', 'tests/_r.py'} or f['filename'].startswith('tests/_r_cache/') for f in changed):
            receipt['retained'].append({'number': number, 'reason': 'unexpected unique non-cache work'})
            save()
            continue
        tag = 'archive/r14/pr-190-20261107'
        archive(tag, expected)
        receipt['archives'].append({'ref': 'refs/tags/' + tag, 'sha': expected})
        comment = ('Superseded by the complete NNS R 14.0 integration in #192, merged as `' + MERGE + '`. '
                   'This initial cache-only PR was NOT merged unchanged. Its snapshot is preserved at `' + tag + '` '
                   '(`' + expected + '`). The complete source-authored cache and aligned Python implementations '
                   'landed together after the full Python 3.11-3.14 GitHub matrix passed.')
    else:
        if not ancestor(expected, main):
            receipt['retained'].append({'number': number, 'reason': 'head not contained in main'})
            save()
            continue
        comment = ('Incorporated through NNS R 14.0 integration #192, merged as `' + MERGE + '`. '
                   'Verified exact head `' + expected + '` is an ancestor of `main`; original commits and authorship '
                   'are preserved. The integration includes the subsequent R14 compatibility corrections and passed '
                   'the complete GitHub Python 3.11-3.14 matrix. This supersedes the separate landing, not the work or review history.')
    existing = pages('/issues/' + str(number) + '/comments')
    duplicate = any(MERGE in (c.get('body') or '') and (
        'Incorporated through NNS R 14.0 integration' in (c.get('body') or '') or
        'Superseded by the complete NNS R 14.0 integration' in (c.get('body') or '')) for c in existing)
    if not duplicate:
        request('/issues/' + str(number) + '/comments', {'body': comment})
    if pr['state'] == 'open':
        assert number != 192, 'Landing PR must already be merged'
        request('/pulls/' + str(number), {'state': 'closed'}, method='PATCH')
    closed = request('/pulls/' + str(number))
    assert closed['state'] == 'closed'
    receipt['retired'].append({'number': number, 'sha': expected, 'merged_flag': closed['merged']})
    approved.append((number, pr['head']['ref'], expected))
    save()

# Delete only exact preserved heads that no still-open PR references.
open_prs = pages('/pulls?state=open')
for number, branch, sha in approved:
    if branch == 'main' or any(p['head']['ref'] == branch or p['base']['ref'] == branch for p in open_prs):
        receipt['retained'].append({'number': number, 'branch': branch, 'reason': 'branch still referenced'})
        continue
    path = '/git/ref/heads/' + urllib.parse.quote(branch, safe='')
    ref = request(path, missing_ok=True)
    if ref is None:
        continue
    if ref['object']['sha'] != sha:
        receipt['retained'].append({'number': number, 'branch': branch, 'reason': 'branch advanced'})
        continue
    meta = request('/branches/' + urllib.parse.quote(branch, safe=''))
    if meta['protected']:
        receipt['retained'].append({'number': number, 'branch': branch, 'reason': 'protected branch'})
        continue
    request('/git/refs/heads/' + urllib.parse.quote(branch, safe=''), method='DELETE')
    assert request(path, missing_ok=True) is None
    receipt['deleted_branches'].append({'number': number, 'branch': branch, 'sha': sha})
    save()
receipt['remaining_open_prs'] = [{'number': p['number'], 'head': p['head']['ref']} for p in pages('/pulls?state=open')]
save()
print(json.dumps({k: v for k, v in receipt.items() if k not in ('retired', 'deleted_branches')}, indent=2))
print('Retired:', len(receipt['retired']), 'Deleted branches:', len(receipt['deleted_branches']))

# Preserve our publication evidence before removing the temporary staging ref.
staging_path = '/git/ref/heads/' + urllib.parse.quote(STAGING, safe='')
staging = request(staging_path, missing_ok=True)
if staging and staging['object']['sha'] == os.environ['GITHUB_SHA']:
    tag = 'archive/r14/publication-evidence-20261107'
    archive(tag, os.environ['GITHUB_SHA'])
    receipt['archives'].append({'ref': 'refs/tags/' + tag, 'sha': os.environ['GITHUB_SHA']})
    request('/git/refs/heads/' + urllib.parse.quote(STAGING, safe=''), method='DELETE')
    receipt['deleted_branches'].append({'number': None, 'branch': STAGING, 'sha': os.environ['GITHUB_SHA']})
else:
    receipt['retained'].append({'branch': STAGING, 'reason': 'staging head changed or already removed'})
save()
with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as summary:
    summary.write('R14 integration ' + MERGE + ': retired ' + str(len(receipt['retired'])) + ' PRs; removed ' + str(len(receipt['deleted_branches'])) + ' preserved branches. Remaining open PRs: ' + str(receipt['remaining_open_prs']) + '.\n')
