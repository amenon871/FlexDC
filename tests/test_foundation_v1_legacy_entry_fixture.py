"""Exercise retained historical stage APIs without routing through the new CLI.

The production CLI intentionally no longer resumes legacy databases/chunk flags.
These old regression fixtures still test current retained execute_scope() code;
new-format behavior is tested in test_foundation_v1_simple_collector.py.
"""
import ast
import subprocess
from functools import lru_cache
from am_flexdc_foundation_v1_common import ROOT
from foundation_v1 import campaign


@lru_cache(None)
def code():
    source=subprocess.check_output(['git','-C',str(ROOT),'show','7eae6d5ac893f14ad41fca6bf829f9f86411db14:foundation_v1/campaign.py'],text=True)
    tree=ast.parse(source);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    return compile(ast.Module(body=[node],type_ignores=[]),'<retired-stage-entry-fixture>','exec')


def invoke():
    namespace=dict(campaign.__dict__);exec(code(),namespace)
    return namespace['main']()
