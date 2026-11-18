"""Retire lessons: point every reference at a replacement, add redirects for the old URLs, delete the files.

    uv run --no-project scripts/retire_lessons.py map.json [--dry-run]

map.json: {"<key>:<id>": "<key>:<id>" | null, ...}   e.g. {"air-fanout-v1:04.02": "dlcore:03.02"}
A null target means "no replacement": references are dropped and the old URL goes to the course page.

Touches: practice/src/problems/*.py (META lessons=[...]), practice/src/{builds,rounds,quizzes}/*.json,
site/src/content/concepts.json (replaced only when the target is in the concept's course and unowned, else
dropped), `lesson:` links in lesson bodies, site/redirects.json. Hand-laid-out JSON is edited as text.
"""
import html, json, re, sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
COURSES = ROOT / 'site/src/content/courses'
REF = r'[a-z0-9-]+:[0-9A-Z]{2}\.\d\d'


def lessons():
    """{ref: (file, course dir, module slug, lesson slug, "Course: Title")} for every lesson on the site."""
    out = {}
    for cj in sorted(COURSES.glob('*/course.json')):
        c = json.loads(cj.read_text()); mods = {m['n']: m['slug'] for m in c['modules']}
        for f in sorted(cj.parent.glob('lessons/*.md')):
            fm = f.read_text().split('\n---', 1)[0]
            get = lambda k: re.search(rf'^{k}: "?([^"\n]+)"?', fm, re.M).group(1)
            out[f"{c['key']}:{get('id')}"] = (f, cj.parent.name, mods[get('module')], get('slug'), f"{c['title']}: {get('title')}")
    return out


def sub_list(text, old, new):
    """Replace "old" with "new" inside JSON/Python string lists, dropping it if new is None or already present."""
    def fix(m):
        items = re.findall(r'''(["'])([^"']+)\1''', m.group(2)); q = items[0][0] if items else '"'
        vals = []
        for _, v in items:
            v = new if v == old else v
            if v and v not in vals: vals.append(v)
        return m.group(1) + ', '.join(f'{q}{v}{q}' for v in vals) + m.group(3)
    return re.sub(r'(\[)([^\[\]]*\b' + re.escape(old) + r'\b[^\[\]]*)(\])', fix, text)


def main(mapfile, dry):
    M = json.loads(pathlib.Path(mapfile).read_text()); L = lessons(); log = []
    missing = [o for o in M if o not in L] + [n for n in M.values() if n and n not in L]
    assert not missing, f'unknown lessons: {missing}'
    assert not any(n in M for n in M.values() if n), 'a target is itself being retired'
    files = [*ROOT.glob('practice/src/problems/*.py'), *ROOT.glob('practice/src/builds/*.json'),
             *ROOT.glob('practice/src/rounds/*.json'), *ROOT.glob('practice/src/quizzes/*.json')]
    for f in files:
        t = s = f.read_text()
        for old, new in M.items():
            if old in t: t = sub_list(t, old, new)
        if t != s:
            log.append(f'refs  {f.relative_to(ROOT)}')
            if not dry: f.write_text(t)
    # lesson bodies: [text](lesson:old) → lesson:new, or plain text when there is no replacement; and plain
    # "lesson 01.11" mentions of a retired lesson in its own course → a link to the replacement, titled
    for ref, (f, cdir, *_) in L.items():
        if ref in M: continue
        t = s = f.read_text()
        for old, new in M.items():
            t = re.sub(r'\[([^\]]*)\]\(lesson:' + re.escape(old) + r'\)', (lambda m: f'[{m.group(1)}](lesson:{new})') if new else r'\1', t)
            if not new or L[old][1] != cdir: continue
            title = html.escape(L[new][4]).replace('"', '&quot;'); pat = r'\b[Ll]esson ' + re.escape(old.split(':')[1]) + r'\b'
            fm, sep, body = t.partition('\n---\n')
            fm = re.sub(pat, f'<a href=\\"lesson:{new}\\">{title}</a>', fm)   # inside the html: "…" frontmatter string
            t = fm + sep + re.sub(pat, f'[{L[new][4]}](lesson:{new})', body)
        if t != s:
            log.append(f'link  {f.relative_to(ROOT)}')
            if not dry: f.write_text(t)
    # concepts: one owner per lesson, lessons stay in the concept's course
    cp = ROOT / 'site/src/content/concepts.json'; C = json.loads(cp.read_text())['concepts']
    owned = {r for c in C for r in c['lessons']}; key = {L[r][1]: r.split(':')[0] for r in L}
    t = cp.read_text()
    for c in C:
        for old in [r for r in c['lessons'] if r in M]:
            new = M[old]
            ok = new and new.split(':')[0] == key[c['course']] and new not in owned
            if ok: owned.add(new)
            t = t.replace(json.dumps(c['lessons']), json.dumps([new if (r == old and ok) else r for r in c['lessons'] if r != old or ok]))
            c['lessons'] = [new if (r == old and ok) else r for r in c['lessons'] if r != old or ok]
            log.append(f"concept {c['id']}: {old} → {new if ok else 'dropped'}" + ('' if c['lessons'] else '  !! NO LESSONS LEFT'))
    if not dry: cp.write_text(t)
    # redirects + delete
    rp = ROOT / 'site/redirects.json'; R = json.loads(rp.read_text())
    for old, new in M.items():
        f, cdir, mslug, slug, _ = L[old]
        R[f'/library/{cdir}/{mslug}/{slug}/'] = f'lesson:{new}' if new else f'/library/{cdir}/'
        log.append(f'gone  {f.relative_to(ROOT)}')
        if not dry: f.unlink()
    if not dry: rp.write_text(json.dumps(R, indent=1, ensure_ascii=False) + '\n')
    print('\n'.join(log)); print(f"{len(M)} lessons retired{' (dry run)' if dry else ''}")


if __name__ == '__main__':
    main(sys.argv[1], '--dry-run' in sys.argv)
