# SPDX-License-Identifier: AGPL-3.0-only
"""Compile lexical annotations into inflected words in a separate dictionary."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

GODAN={
    'う':('わ','い','って','った'), 'く':('か','き','いて','いた'),
    'ぐ':('が','ぎ','いで','いだ'), 'す':('さ','し','して','した'),
    'つ':('た','ち','って','った'), 'ぬ':('な','に','んで','んだ'),
    'ぶ':('ば','び','んで','んだ'), 'む':('ま','み','んで','んだ'),
    'る':('ら','り','って','った'),
}

def forms(reading,surface,kind):
    if not reading or not surface or reading[-1]!=surface[-1]:raise ValueError('Unaligned lemma ending')
    def pair(suffix):return reading[:-1]+suffix,surface[:-1]+suffix
    result=[(reading,surface,'adjective' if kind=='adjective' else 'verb')]
    if kind=='adjective':
        if reading[-1]!='い':raise ValueError('Expected i-adjective')
        endings=['かった','くない','くなかった','くて','ければ','くても','くなる','くなりました']
    else:
        if kind=='ichidan':
            if reading[-1]!='る':raise ValueError('Expected ichidan ending')
            negative,polite,te,past='','','て','た'
            conditional='れば'
            passive='られ'
        elif kind in ('godan','iku'):
            negative,polite,te,past=GODAN[reading[-1]]
            conditional=dict(zip('うくぐすつぬぶむる','えけげせてねべめれ'))[reading[-1]]+'ば'
            passive=negative+'れ'
            if kind=='iku':
                if reading!='いく' or surface!='行く':raise ValueError('Invalid iku annotation')
                te,past='って','った'
        else:raise ValueError('Unknown inflection class')
        endings=[negative+'ない',negative+'なかった',negative+'なくて',negative+'なくてもいい',negative+'なくてもよい',
            negative+'なければ',negative+'なければいけません',polite+'ます',polite+'ました',polite+'ません',
            polite+'たい',polite+'たかった',polite+'たくない',te,te+'いる',te+'いた',te+'いない',te+'いなかった',
            te+'います',te+'いました',te+'ください',te+'もいい',past]
        endings.append(conditional)
        endings.extend(passive+ending for ending in
            ['る','た','て','ている','ていた','ていない','ない','なかった','ます','ました','れば'])
    result.extend((*pair(ending),'adjective' if kind=='adjective' else 'verb') for ending in endings)
    return sorted(set(result))

def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--lexemes',type=Path,required=True)
    parser.add_argument('--seed-journal',type=Path,required=True)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--cli',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Refusing to overwrite journal')
    lexemes=[line.split('\t') for line in args.lexemes.read_text(encoding='utf-8').splitlines() if line and not line.startswith('#')]
    entries=sorted({entry for reading,surface,kind in lexemes for entry in forms(reading,surface,kind)})
    before=sha(args.base);seed=sha(args.seed_journal)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(args.seed_journal,args.output)
    # Batch append avoids starting and opening the full dictionary for each form.
    # The CLI validates the complete TSV before writing to this experiment.
    batch=args.output.with_suffix('.forms.tsv')
    batch.write_text(''.join('\t'.join([reading,surface,pos,'9000'])+'\n' for reading,surface,pos in entries),encoding='utf-8')
    subprocess.run([str(args.cli),'dict','add-tsv',str(args.base),str(args.output),str(batch)],check=True,stdout=subprocess.DEVNULL,timeout=300)
    assert sha(args.base)==before and sha(args.seed_journal)==seed
    report=dict(lexemes=len(lexemes),generated_word_forms=len(entries),lexemes_sha256=sha(args.lexemes),generated_forms_sha256=sha(batch),seed_sha256=seed,base_sha256=before,journal_sha256=sha(args.output),installed=False,test_sentences_used=False,weight_model_updated=False,
        limits='Productive conjugation expansion of authored lexical annotations, not independent sentence problems, not neural weight training. Counts are not measured frequencies.')
    args.output.with_suffix('.report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))

if __name__=='__main__':main()
