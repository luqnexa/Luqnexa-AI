# SPDX-License-Identifier: AGPL-3.0-only
"""Flag word-substitution duplicates; these counts do not measure meaning."""
import argparse
from collections import Counter
from difflib import SequenceMatcher
import json
import re
from pathlib import Path

def frame(text):
    return re.sub(r'[A-Za-z0-9_]+','<EN>',text).replace(' ','')

def audit(rows, previous=()):
    groups=Counter(frame(r['good']) for r in rows)
    old_frames={frame(r['good']) for r in previous}
    unique=list(dict.fromkeys(frame(r['good']) for r in rows))
    near=[]
    for i, left in enumerate(unique):
        for right in unique[:i]:
            # Literal slots alone must not dominate the similarity score.
            a=left.replace('<EN>', '')
            b=right.replace('<EN>', '')
            similarity=SequenceMatcher(None,a,b,autojunk=False).ratio()
            if min(len(a),len(b))>=6 and similarity>=0.8:
                near.append(dict(left=right,right=left,similarity=round(similarity,3)))
    words=Counter(word for r in rows for word in
                  {w.lower() for w in re.findall(r'[A-Za-z][A-Za-z0-9_]*',r['good'])})
    return dict(examples=len(rows),surface_frames=len(groups),
        substitution_variants=sum(n-1 for n in groups.values()),
        repeated_frames=[dict(frame=k,count=n) for k,n in groups.most_common() if n>1],
        reused_previous_frames=[dict(frame=k,count=n) for k,n in groups.items() if k in old_frames],
        near_frames_for_review=near,
        english_word_exposure=[dict(word=k,examples=n) for k,n in words.most_common(20)],
        limits='Surface structure and lexical exposure only; similarity flags require review. Different frames may still repeat the same meaning. Not a semantic diversity score.')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('lessons',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--require-distinct-frames',action='store_true',help='For new batches; reject simple English-word substitution duplicates')
    parser.add_argument('--against',type=Path,action='append',default=[],help='Prior training lessons; reject reuse of their frames in strict mode')
    args=parser.parse_args()
    rows=[json.loads(l) for l in args.lessons.read_text(encoding='utf-8').splitlines() if l]
    previous=[json.loads(l) for path in args.against for l in path.read_text(encoding='utf-8').splitlines() if l]
    report=audit(rows,previous)
    if args.output:args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))
    if args.require_distinct_frames and (report['substitution_variants'] or report['reused_previous_frames']):
        raise SystemExit('Repeated frames: revise the new lessons rather than count substitutions as diverse problems')

if __name__=='__main__':main()
