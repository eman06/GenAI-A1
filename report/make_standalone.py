"""Builds report/main_standalone.tex: tables inlined, every figure optional (compiles with no image files)."""
import re

s = open('report/main.tex', encoding='utf-8').read()
s = s.replace('(YouTube link to be added)', 'YouTube link to be added')

# inline \input{tables/...}
def inline(m):
    name = m.group(1)
    path = 'report/' + name + ('' if name.endswith('.tex') else '.tex')
    return open(path, encoding='utf-8').read().strip()
s = re.sub(r'\\input\{(tables/[^}]+)\}', inline, s)

# remove existing IfFileExists wrappers, then wrap every figure / figure* environment
s = re.sub(r'\\IfFileExists\{figures/[^}]+\}\{(\\begin\{figure\}.*?\\end\{figure\})\}\{\}', r'\1', s, flags=re.S)

def wrap(m):
    block = m.group(0)
    imgs = re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', block)
    if not imgs:
        return block
    return '\\IfFileExists{figures/' + imgs[0] + '}{' + block + '}{}'
s = re.sub(r'\\begin\{figure\*?\}.*?\\end\{figure\*?\}', wrap, s, flags=re.S)

open('report/main_standalone.tex', 'w', encoding='utf-8').write(s)
print(len(s.splitlines()), 'lines')
