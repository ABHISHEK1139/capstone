with open('paper/paper_presentation.tex', 'r', encoding='utf-8') as f:
    text = f.read()

replacements = [
    ('( = 9{,}835$)', '($N = 9{,}835$)'),
    ('( = 169$)', '($M = 169$)'),
    ('( = 3{,}196$)', '($N = 3{,}196$)'),
    ('( = 88{,}162$)', '($N = 88{,}162$)'),
    ('( = 16{,}470$)', '($M = 16{,}470$)'),
    ('( = 990{,}002$)', '($N = 990{,}002$)'),
    ('( = 41{,}270$)', '($M = 41{,}270$)'),
]

for old, new in replacements:
    count = text.count(old)
    text = text.replace(old, new)
    print(f"Replaced {count} occurrences of {old} with {new}")

with open('paper/paper_presentation.tex', 'w', encoding='utf-8', newline='\n') as f:
    f.write(text)
print("Finished updating paper_presentation.tex")
