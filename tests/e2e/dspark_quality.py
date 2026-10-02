"""Bounded functional checks for code, mathematics and exact-fact prose."""
from __future__ import annotations
import json
import re
import subprocess
import sys

CASES = [
 ("code","merge","Return only a complete Python function merge_intervals(intervals) that merges closed overlapping intervals into a sorted list of [start,end]. Do not mutate input. No imports, explanations or examples.",
  [[[[1,3],[2,6],[8,10],[9,12]],[[1,6],[8,12]]],[[],[]],[[[3,4],[1,2]],[[1,2],[3,4]]]],"merge_intervals"),
 ("code","substring","Return only a complete Python function longest_unique_substring(text) returning the longest substring length without repeated characters. No imports, explanations or examples.",
  [["abcabcbb",3],["bbbbb",1],["",0],["pwwkew",3]],"longest_unique_substring"),
 ("code","bound","Return only a complete Python function lower_bound(values,target) returning the first index of an element >= target in a sorted list, or len(values). No imports, explanations or examples.",
  [[[[1,2,2,4],2],1],[[[],7],0],[[[1,3],4],2]],"lower_bound"),
 ("math","squares","计算从1到100的整数平方和。只输出 FINAL: 后跟十进制答案，不解释。",338350,None),
 ("math","binomial","从20个不同元素中无序选6个，有多少种组合？只输出 FINAL: 后跟十进制答案，不解释。",38760,None),
 ("math","symmetric","已知a+b=13、ab=40，求a平方加b平方。只输出 FINAL: 后跟十进制答案，不解释。",89,None),
 ("prose","archive","独立事实：项目AURORA，维护人Lin，版本7.4，校验码COBALT-8426。用一句中文完整重复四项事实。",["AURORA","Lin","7.4","COBALT-8426"],None),
 ("prose","library","独立事实：图书馆名晨光，借阅期21天，联系人陈雨，预约号AMBER-7193。用一句中文完整重复四项事实。",["晨光","21天","陈雨","AMBER-7193"],None),
 ("prose","handoff","独立事实：实验EMERALD，样本5638，负责人Zhao，归档日2026-10-03。用一句中文完整重复四项事实。",["EMERALD","5638","Zhao","2026-10-03"],None),
]
CHECKER = r"""
import ast,json,re,resource,sys
resource.setrlimit(resource.RLIMIT_CPU,(2,2));resource.setrlimit(resource.RLIMIT_AS,(128*1024**2,128*1024**2))
d=json.load(sys.stdin);text=d['text'];blocks=re.findall(r'```(?:python)?\s*\n(.*?)```',text,re.S)
code=next((b for b in blocks if 'def '+d['name']+'(' in b),text)
tree=ast.parse(code);assert len(list(ast.walk(tree)))<=1000
builtins={k:__builtins__.__dict__[k] for k in ['range','len','sorted','sum','min','max','enumerate','zip','abs','set','list','dict','tuple','int','str','float','bool']}
for n in ast.walk(tree):
    assert not isinstance(n,(ast.Import,ast.ImportFrom,ast.Global,ast.Nonlocal,ast.ClassDef,ast.With,ast.AsyncFunctionDef))
    if isinstance(n,ast.Name):assert not n.id.startswith('__')
    if isinstance(n,ast.Attribute):assert n.attr in ('append','pop','get','sort','items','values','keys','extend','copy')
    if isinstance(n,ast.Call) and isinstance(n.func,ast.Name):assert n.func.id in builtins or n.func.id==d['name']
ns={'__builtins__':builtins};exec(compile(tree,'<generated>','exec'),ns)
for args,expected in d['cases']:
    before=json.dumps(args,sort_keys=True)
    actual=ns[d['name']](*args) if d['name']=='lower_bound' else ns[d['name']](args)
    assert actual==expected,(actual,expected)
    assert json.dumps(args,sort_keys=True)==before,'input mutated'
print('PASS')
"""

def grade(case,text):
    category,_,_,expected,name=case
    if category=="prose":return {"passed":all(f in text for f in expected),"oracle":"exact facts"}
    if category=="math":
        m=re.fullmatch(r"FINAL:\s*([0-9,]+)",text.strip())
        return {"passed":bool(m) and int(m.group(1).replace(",",""))==expected,"oracle":"exact known arithmetic"}
    try:
        r=subprocess.run([sys.executable,"-I","-S","-c",CHECKER],input=json.dumps({"text":text,"name":name,"cases":expected}),text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=4)
        return {"passed":r.returncode==0,"oracle":"bounded restricted AST + functional cases + immutable input","exit_code":r.returncode,"error":r.stderr[-800:]}
    except Exception as e:return {"passed":False,"oracle":"bounded restricted AST","error":str(e)}
