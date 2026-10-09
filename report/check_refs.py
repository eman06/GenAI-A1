import os
import re

s = open('report/main_standalone.tex', encoding='utf-8').read()
refs = set(re.findall(r'\\ref\{([^}]+)\}', s))
labels = re.findall(r'\\label\{([^}]+)\}', s)
figs = set(re.findall(r'includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', s))
print('missing labels:', sorted(refs - set(labels)))
print('duplicate labels:', sorted({l for l in labels if labels.count(l) > 1}))
print('missing files:', sorted(f for f in figs if not os.path.exists('report/figures/' + f)))
print('control chars:', sum(s.count(chr(c)) for c in (8, 9)), '| video link:', 'YC3jGST1eOY' in s)
