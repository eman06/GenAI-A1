import glob

BS = chr(92)
for f in glob.glob('report/tables/*.tex'):
    s = open(f, encoding='utf-8').read()
    s = s.replace(chr(8), BS + 'b').replace(chr(9), BS + 't')
    open(f, 'w', encoding='utf-8').write(s)
    print(f, repr(s[:30]))

for f in glob.glob('report/tables/*.tex'):
    s = open(f, encoding='utf-8').read()
    good = BS * 2 + ' ' + BS + 'midrule'
    s = s.replace(BS * 3 + 'midrule', good).replace(BS * 2 + 'midrule', good)
    open(f, 'w', encoding='utf-8').write(s)
