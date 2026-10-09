"""dist/report_overleaf.zip: main.tex + inlined tables + only the referenced figures, compressed (<10 MB)."""
import io, os, re, zipfile
from PIL import Image
s = open('report/main_standalone.tex', encoding='utf-8').read()
figs = sorted(set(re.findall(r'includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', s)))
os.makedirs('dist', exist_ok=True)
z = zipfile.ZipFile('dist/report_overleaf.zip', 'w', zipfile.ZIP_DEFLATED)
z.writestr('main.tex', s)
for f in figs:
    p = 'report/figures/' + f
    if not os.path.exists(p):
        print('missing (optional):', f); continue
    im = Image.open(p).convert('RGB'); im.thumbnail((1300, 1300))
    b = io.BytesIO(); im.quantize(256).save(b, 'PNG', optimize=True)
    z.writestr('figures/' + f, b.getvalue())
z.close()
print(round(os.path.getsize('dist/report_overleaf.zip') / 1e6, 2), 'MB,', len(figs), 'figures referenced')
