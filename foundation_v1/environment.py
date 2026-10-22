"""Pinned numerical stack for deterministic Foundation adaptive collection."""
import sys
import platform
import importlib.metadata

EXPECTED={'numpy':'2.2.6','pandas':'2.3.3','scipy':'1.15.3'}

def versions(require=False):
    actual={'python':platform.python_version(),**{name:importlib.metadata.version(name) for name in EXPECTED}}
    mismatches={k:{'expected':v,'actual':actual[k]} for k,v in EXPECTED.items() if actual[k]!=v}
    if sys.version_info[:2]!=(3,10):mismatches['python']={'expected':'3.10.x','actual':actual['python']}
    if require and mismatches:raise ValueError('Foundation numerical stack mismatch: '+str(mismatches))
    return {'actual':actual,'expected':{'python':'3.10.x',**EXPECTED},'status':'PASS' if not mismatches else 'MISMATCH','mismatches':mismatches}
