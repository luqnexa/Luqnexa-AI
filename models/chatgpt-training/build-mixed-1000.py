# SPDX-License-Identifier: AGPL-3.0-only
"""Reviewed vocabulary/frame combinations, not 1,000 independently authored sentences."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
NAME='mixed-20261013-1000.jsonl'

# Exact English labels are typed as literal spans. Japanese suffixes are romanized.
# Every frame is applicable to every item in its own vocabulary category.
NAMES='Python Rust JavaScript TypeScript Java Kotlin Swift Ruby Perl PHP Lua Go Windows Linux Android Ubuntu Debian Fedora Firefox Chrome Chromium GitHub GitLab Docker Kubernetes'.split()
IDENTIFIERS='user_id request_id file_path output_path input_text result_count error_code item_count session_id access_token user_name file_name folder_name current_value previous_value target_path source_path start_time end_time max_count min_count data_size buffer_size retry_count page_number'.split()
WORDS=[('hello','ハロー'),('world','ワールド'),('book','ブック'),('music','ミュージック'),('coffee','コーヒー'),('tea','ティー'),('water','ウォーター'),('apple','アップル'),('orange','オレンジ'),('table','テーブル'),('window','ウィンドウ'),('door','ドア'),('school','スクール'),('teacher','ティーチャー'),('friend','フレンド'),('family','ファミリー'),('morning','モーニング'),('evening','イブニング'),('summer','サマー'),('winter','ウィンター'),('happy','ハッピー'),('thank','サンク'),('please','プリーズ'),('welcome','ウェルカム'),('message','メッセージ')]
ACRONYMS='API CPU GPU RAM ROM USB HTML CSS HTTP HTTPS JSON XML SQL CSV PDF PNG JPEG GIF TCP UDP DNS SSD HDD LAN URL'.split()

NAME_FRAMES=[
 ('nitsuiteshiraberu','についてしらべる','について調べる'),
 ('notukaikatawooboeru','のつかいかたをおぼえる','の使い方を覚える'),
 ('noshiryouwosagasu','のしりょうをさがす','の資料を探す'),
 ('nosetteiwokakunin','のせっていをかくにん','の設定を確認'),
 ('nomeishouwokirokusuru','のめいしょうをきろくする','の名称を記録する'),
 ('notokuchouwosetsumeisuru','のとくちょうをせつめいする','の特徴を説明する'),
 ('nosousawotamesu','のそうさをためす','の操作を試す'),
 ('noshitsumonwokangaeru','のしつもんをかんがえる','の質問を考える'),
 ("noun'youwokentousuru",'のうんようをけんとうする','の運用を検討する'),
 ('nogaiyouwoyomu','のがいようをよむ','の概要を読む'),
 ('nokaisetsuwosagasu','のかいせつをさがす','の解説を探す')]
ID_FRAMES=[
 ('noataiwokakunin','のあたいをかくにん','の値を確認'),
 ('noataiwokoushinsuru','のあたいをこうしんする','の値を更新する'),
 ('noataiwoyomitoru','のあたいをよみとる','の値を読み取る'),
 ('nomeishouwokaeru','のめいしょうをかえる','の名称を変える'),
 ('woshokikasuru','をしょきかする','を初期化する'),
 ('wokirokushitehozon','をきろくしてほぞん','を記録して保存'),
 ('nosengenwomiru','のせんげんをみる','の宣言を見る'),
 ('nosanshouwosagasu','のさんしょうをさがす','の参照を探す'),
 ('wokansuuniwatasu','をかんすうにわたす','を関数に渡す'),
 ('nohikakuwookonau','のひかくをおこなう','の比較を行う'),
 ('noimiwosetsumeisuru','のいみをせつめいする','の意味を説明する')]
WORD_FRAMES=[
 ('tokaitamemo','とかいためも','と書いたメモ'),
 ('womemonikaku','をめもにかく','をメモに書く'),
 ('notsuduriwokakunin','のつづりをかくにん','の綴りを確認'),
 ('noimiwoshiraberu','のいみをしらべる','の意味を調べる'),
 ('wohatsuonsuru','をはつおんする','を発音する'),
 ('nohatsuonworenshuu','のはつおんをれんしゅう','の発音を練習'),
 ('wokaitenokosu','をかいてのこす','を書いて残す'),
 ('tokaitakami','とかいたかみ','と書いた紙'),
 ('wobunshouniireru','をぶんしょうにいれる','を文章に入れる'),
 ('noyomikatawokiku','のよみかたをきく','の読み方を聞く'),
 ('woreibunniireru','をれいぶんにいれる','を例文に入れる')]
ACRONYM_FRAMES=[
 ('noshiyouwoshiraberu','のしようをしらべる','の仕様を調べる'),
 ('nomeishouwokakunin','のめいしょうをかくにん','の名称を確認'),
 ('noimiwosetsumeisuru','のいみをせつめいする','の意味を説明する'),
 ('noyakuwariwooboeru','のやくわりをおぼえる','の役割を覚える'),
 ('noshiryouwoyomu','のしりょうをよむ','の資料を読む'),
 ('noshitsumonwosuru','のしつもんをする','の質問をする'),
 ('nokaisetsuwokiku','のかいせつをきく','の解説を聞く'),
 ('nomeishouwomemoninokosu','のめいしょうをめもにのこす','の名称をメモに残す'),
 ('notokuchouwokiroku','のとくちょうをきろく','の特徴を記録'),
 ('noshiyoureiwomiru','のしようれいをみる','の使用例を見る'),
 ('nokijutsuwosagasu','のきじゅつをさがす','の記述を探す')]

def read(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]

def build(known):
    used_inputs={r.get('input') for r in known}
    used_pairs={(r['context'],r['good'],r['bad']) for r in known}
    result=[]
    categories=[('proper_names',NAMES,NAME_FRAMES,'正式名称のケースを保持して説明する'),
                ('identifiers',IDENTIFIERS,ID_FRAMES,'ソースコードの指定変数名を小文字のまま説明する'),
                ('english_literals',WORDS,WORD_FRAMES,'入力した英単語を英字のまま記し翻訳やかな置換をしない'),
                ('acronyms',ACRONYMS,ACRONYM_FRAMES,'正式略称を大文字で記して説明する')]
    for category,words,frames,hint in categories:
        count=0
        # Frame-first ordering covers all 25 vocabulary items before adding variants.
        for index,(roman,kana,surface) in enumerate(frames):
            roman=roman.replace(' ','')
            for word in words:
                token,replacement=word if isinstance(word,tuple) else (word,None)
                if replacement is None:
                    replacement=token.upper() if category=='identifiers' else token.lower()
                record=dict(input=token.lower()+roman,reading=token+kana,
                    context=f'{token}: {hint}',good=token+surface,bad=replacement+surface,
                    topic=category,split='train',group=f'mixed1000-{category}-{token}',language_pair='ja-en',
                    review=f'語彙{token}の綴りと指定ケース、枠{index}の読み・自然な接続を点検。英語を保持し日本語を変換する。',
                    provenance='Assistant-reviewed vocabulary/frame synthetic combination; not independently human-verified',
                    template_id=f'{category}-{index}')
                pair=(record['context'],record['good'],record['bad'])
                if record['input'] in used_inputs or pair in used_pairs:continue
                assert record['good']!=record['bad']
                used_inputs.add(record['input']);used_pairs.add(pair);result.append(record);count+=1
                if count==250:break
            if count==250:break
        assert count==250,(category,count)
    assert len(result)==1000
    return result

def main():
    state=json.loads((ROOT/'.cache/chatgpt-training/status.json').read_text(encoding='utf-8'))
    known=read(ROOT/'models/native-training.jsonl')+read(HERE/'examples.jsonl')
    for name in state['known_lessons']:known+=read(HERE/name)
    rows=build(known)
    path=HERE/NAME
    if path.exists():raise FileExistsError('Do not overwrite an existing reviewed lesson')
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    with (HERE/'mixed-20261013-1000.tsv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream,delimiter='\t');writer.writerow(['問題（ローマ字）','期待する変換','誤答候補','文脈','分野'])
        writer.writerows([r['input'],r['good'],r['bad'],r['context'],r['topic']] for r in rows)
    review=dict(examples=len(rows),category_counts=dict(Counter(r['topic'] for r in rows)),
        unique_inputs=len({r['input'] for r in rows}),unique_surfaces=len({r['good'] for r in rows}),
        templates=len({r['template_id'] for r in rows}),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        independently_reviewed=False,limits='Template combinations; not 1,000 independent semantic situations. Wrong alternatives are spelling/case substitutions, not free generation.')
    (HERE/'mixed-20261013-1000-review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(review,ensure_ascii=False))

if __name__=='__main__':main()
