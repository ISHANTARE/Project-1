import json
import sys
from pathlib import Path

PROJECT_ROOT = Path("d:/Project-1-main/Project-1-main")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from scripts.benchmark import run_laps_system, run_few_shot_icl, run_static_dict_system, run_unconstrained_nmt

with open(PROJECT_ROOT / "data/evaluation/gold_corpus.json", encoding="utf-8") as f:
    sents = json.load(f)["sentences"]

sub = sents[:15]
laps = run_laps_system(sub)
few = run_few_shot_icl(sub)
stat = run_static_dict_system(sub)
uncon = run_unconstrained_nmt(sub)

for i in range(15):
    print("=" * 60)
    print(f"SENTENCE {sub[i]['id']}: {sub[i]['sentence']}")
    print(f"REF:   {sub[i].get('reference')}")
    print(f"LAPS:  {laps[i]['rewritten']}")
    print(f"FEW:   {few[i]['rewritten']}")
    print(f"STAT:  {stat[i]['rewritten']}")
    print(f"UNCON: {uncon[i]['rewritten']}")
