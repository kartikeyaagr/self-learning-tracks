"""Check interactive lesson steps (the `steps:` frontmatter line) before they ship.

    uv run --no-project scripts/check_steps.py <lesson.md> [...]     check these lessons
    uv run --no-project scripts/check_steps.py --concept attention   list a concept's lesson files, in order

Format: site/src/lib/steps.js. Writer brief: tasks/steps-writer-prompt.md.
Errors fail the check; warnings are printed for a human to judge.
"""
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / 'site'
WIDGETS = {p.stem for p in (SITE / 'src/lib/widgets').glob('*.js') if not p.stem.endswith('.test')}
KINDS = {'explain', 'interact', 'question'}


def frontmatter(path):
    head = path.read_text(encoding='utf-8').split('\n---\n', 1)[0]
    return {k: v for k, v in re.findall(r'^(\w+): (.*)$', head, re.M)}


def check(path):
    errs, warns = [], []
    fm = frontmatter(path)
    if 'steps' not in fm:
        return ['no `steps:` line'], []
    try:
        steps = json.loads(fm['steps'])
    except json.JSONDecodeError as e:
        return [f'steps is not valid JSON on one line: {e}'], []
    if not isinstance(steps, list) or not 5 <= len(steps) <= 8:
        errs.append(f'give 5-8 steps (found {len(steps) if isinstance(steps, list) else "not a list"})')
        steps = steps if isinstance(steps, list) else []
    qs = []
    for i, s in enumerate(steps, 1):
        at = f'step {i}'
        if not isinstance(s, dict):
            errs.append(f'{at}: not an object'); continue
        extra = set(s) - {'kind', 'title', 'body', 'widget', 'question'}
        if extra: errs.append(f'{at}: unknown keys {sorted(extra)}')
        if s.get('kind') not in KINDS: errs.append(f'{at}: kind must be one of {sorted(KINDS)}')
        t = s.get('title', '')
        if not t or len(t) > 40: errs.append(f'{at}: title is required, at most 40 characters')
        body = s.get('body', '')
        if s.get('kind') != 'question' and len(body.split()) < 15: errs.append(f'{at}: body too short for an {s.get("kind")} step')
        if len(body.split()) > 160: warns.append(f'{at}: body is {len(body.split())} words (aim for 40-120)')
        if body.count('$$') % 2 or (body.replace('$$', '').count('$') - body.count('\\$')) % 2: errs.append(f'{at}: unbalanced $ maths')
        if re.search(r'<[a-z][^>]*>', body): errs.append(f'{at}: no HTML in body, use Markdown')
        w = s.get('widget')
        if s.get('kind') == 'interact' and not w: errs.append(f'{at}: interact steps need a widget')
        if w and w not in WIDGETS: errs.append(f'{at}: unknown widget {w!r} (available: {sorted(WIDGETS)})')
        q = s.get('question')
        if s.get('kind') == 'question' and not q: errs.append(f'{at}: question steps need a question')
        if q:
            o = q.get('options', [])
            if len(o) != 3 or len(set(o)) != 3: errs.append(f'{at}: exactly 3 distinct options')
            a = q.get('answer')
            if not isinstance(a, int) or not 0 <= a < len(o): errs.append(f'{at}: answer is the 0-based index of the right option')
            for k in ('q', 'why', 'hint'):
                if not str(q.get(k, '')).strip(): errs.append(f'{at}: question needs {k}')
            if isinstance(a, int) and 0 <= a < len(o) == 3: qs.append((a, o))
    if len(qs) < 3: errs.append(f'at least 3 steps should ask a question (found {len(qs)})')
    if steps and not steps[-1].get('question'): errs.append('the last step must end with a question (recap and a check)')
    if qs:
        if len({a for a, _ in qs}) == 1 and len(qs) > 2: warns.append('every right answer is in the same position')
        longest = sum(len(o[a]) == max(map(len, o)) and len(o[a]) > 1.3 * sorted(map(len, o))[1] for a, o in qs)
        if longest > len(qs) / 2: warns.append(f'the right option is clearly the longest in {longest}/{len(qs)} questions')
    return errs, warns


def concept_files(cid):
    C = {c['id']: c for c in json.loads((SITE / 'src/content/concepts.json').read_text())['concepts']}
    keys = {json.loads(p.read_text())['key']: p.parent for p in (SITE / 'src/content/courses').glob('*/course.json')}
    for ref in C[cid]['lessons']:
        key, lid = ref.split(':')
        for f in sorted((keys[key] / 'lessons').glob(lid.replace('.', '-') + '-*.md')):
            print(f.relative_to(ROOT))


if __name__ == '__main__':
    if sys.argv[1:2] == ['--concept']:
        concept_files(sys.argv[2]); sys.exit(0)
    bad = 0
    for raw in sys.argv[1:]:
        e, w = check(Path(raw))
        bad += bool(e)
        print(('FAIL  ' if e else 'OK    ') + raw)
        for x in e: print('      error:', x)
        for x in w: print('      warn: ', x)
    print(f'{len(sys.argv) - 1 - bad}/{len(sys.argv) - 1} passed')
    sys.exit(1 if bad else 0)
