import json,glob,re,sys
d=json.load(open('data.json'))
full={}
for f in sorted(glob.glob('full/m*.txt')):
  cur=None
  for line in open(f).read().split('\n'):
    if line.startswith('@'):
      cur=line[1:].strip(); full[cur]={'a':[],'s':''}
    elif cur is None: continue
    elif line.startswith('?? '): full[cur]['s']=line[3:]
    else: full[cur]['a'].append(line)
for m in d:
  for L in m['lessons']:
    if 'a' in L and isinstance(L['a'],str) and not L.get('ak') and L['a'] and not L['a'].startswith('##'):
      L['ak']=L.pop('a')
    elif 'a' in L and L['a']=='' : L.pop('a'); L['ak']=''
    L.setdefault('ak','')
    if L['id'] in full:
      L['a']='\n'.join(full[L['id']]['a']).strip(); L['s']=full[L['id']]['s']
import os
FL=set(json.load(open('flags.json'))) if os.path.exists('flags.json') else set()
for m in d:
  for L in m['lessons']:
    if L['id'] in FL: L['g']=1
    else: L.pop('g',None)
json.dump(d,open('data.json','w'),separators=(',',':'),ensure_ascii=False)
t=open('app.html').read()
open('course.html','w').write(t.replace('__DATA__',json.dumps(d,separators=(',',':'),ensure_ascii=False).replace('</','<\\/')))
print('full notes:',len(full),'| sample answers:',sum(1 for v in full.values() if v['s']))
# also write ../index.html with the site's title, description and favicon
h=open('course.html').read().replace('<meta name="claude:viewport-fit-sides" content="cover">\n','').replace('<title>System Design Course</title>','''<meta name="description" content="A 200-lesson system design course with diagrams, worked examples, self-check questions and flashcard review.">
<meta property="og:title" content="System Design 101">
<meta property="og:description" content="200 short lessons on how large systems are built, with diagrams and review cards.">
<meta name="theme-color" content="#F7F7F8">
<link rel="icon" href="favicon.svg" type="image/svg+xml">
<title>System Design 101</title>''',1)
open('../index.html','w').write(h)
print('wrote ../index.html')
