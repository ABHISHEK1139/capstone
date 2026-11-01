#!/usr/bin/env python3
"""Build a readable HTML preview of paper-draft.tex (no LaTeX engine here).

Parses the draft's own regular structure: title, authors, abstract,
sections, tables, figures (as captioned diagram boxes), placeholders,
and bibliography. This is a preview only — the compiled PDF remains the
canonical paper once results are filled in.
"""
import re
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "paper/paper.tex"
DST = "paper/paper-preview.html"

LABEL_MAP = {
    "sec:intro": "1", "sec:sklearn_scope": "1.2", "sec:determinism_dims": "1.3",
    "sec:rqs": "1.4", "sec:related": "2", "sec:methods": "3",
    "sec:formal_spec": "3.1", "sec:result_states": "3.2",
    "sec:api_contracts": "3.4", "sec:governor": "3.5",
    "sec:data": "4", "sec:setup": "5", "sec:rq_answers": "6.3",
    "sec:ablation": "6.4", "sec:limitations": "7", "app:repro": "A",
    "app:timing": "B",
    "tab:ecosystem": "1", "tab:metrics": "2", "tab:data": "3", "tab:thr": "4",
    "tab:timing": "5", "tab:correctness": "6", "tab:skips": "7",
    "tab:timingfull": "8",
    "fig:engines": "1", "fig:fptree": "2", "fig:arch": "3",
    "fig:family": "4", "fig:flow": "5",
    "fig:runtime": "6", "fig:memory": "7", "fig:scatter": "8",
    "fig:throughput": "9", "fig:itemsets": "10", "fig:outcomes": "11", "fig:ablation": "12",
}

MATH_MAP = [
    (r"\mathrm{supp}", "supp"), (r"\mathrm{conf}", "conf"),
    (r"\mathrm{confidence}", "confidence"),
    (r"\Rightarrow", "\u21d2"), (r"\to", "\u2192"),
    (r"\cup", "\u222a"), (r"\cap", "\u2229"),
    (r"\emptyset", "\u2205"), (r"\infty", "\u221e"), (r"\times", "\u00d7"),
    (r"\leq", "\u2264"), (r"\geq", "\u2265"), (r"\ge", "\u2265"), (r"\le", "\u2264"),
    (r"\neq", "\u2260"), (r"\ne", "\u2260"), (r"\equiv", "\u2261"),
    (r"\wedge", "\u2227"), (r"\vee", "\u2228"),
    (r"\forall", "\u2200"), (r"\exists", "\u2203"),
    (r"\bowtie", "\u22c8"), (r"\subset", "\u2282"), (r"\subseteq", "\u2286"),
    (r"\in", "\u2208"), (r"\notin", "\u2209"),
    (r"\mid", "|"), (r"\cdot", "\u00b7"), (r"\approx", "\u2248"), (r"\pm", "\u00b1"),
    (r"\setminus", "\\"), (r"\max", "max"), (r"\min", "min"),
    (r"\sigma", "\u03c3"), (r"\gamma", "\u03b3"), (r"\theta", "\u03b8"),
    (r"\Delta", "\u0394"),
    (r"\bar{L}", "L\u0304"), (r"\bar", ""),
    (r"{,}", ","), (r"\%", "%"), (r"\#", "#"), (r"\_", "_"),
    (r"\&", "&"), (r"\{", "{"), (r"\}", "}"),
    (r"\mathbf", ""), (r"\mathcal", ""),
]


def extract_braced(s, i):
    """s[i] must be '{'. Return (content, index_after_closing_brace)."""
    assert s[i] == "{"
    depth = 0
    j = i
    while j < len(s):
        c = s[j]
        if c == "\\" and j + 1 < len(s) and s[j + 1] in "{}\\%#&_":
            j += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    raise ValueError("unbalanced braces")


def cmd_arg(s, cmd, pos):
    """Find \\cmd{...} at/after pos; return (content, start, end) or None."""
    m = re.compile(r"\\" + cmd + r"\{").search(s, pos)
    if not m:
        return None
    content, end = extract_braced(s, m.end() - 1)
    return content, m.start(), end


def strip_comments(text):
    text = text.replace("\\%", "\x00PERCENT\x00")
    out = []
    for line in text.splitlines():
        if "%" in line:
            line = line[:line.index("%")]
        out.append(line)
    return "\n".join(out).replace("\x00PERCENT\x00", "\\%")


BIB_ORDER = {}


def inline(text):
    """Convert a chunk of LaTeX body text to HTML inline content."""
    text = re.sub(r"\\enlargethispage\*?\{[^}]*\}", "", text)
    text = re.sub(r"\\vspace\*?\{[^}]*\}", "", text)
    text = re.sub(r"\\hspace\*?\{[^}]*\}", "", text)
    text = re.sub(r"\\setlength\{[^}]*\}\{[^}]*\}", "", text)

    # display equations
    def eq_sub(m):
        inner = m.group(1)
        for a, b in MATH_MAP:
            inner = inner.replace(a, b)
        inner = re.sub(r"\\(?:mathrm|text|mathit)\{([^}]*)\}", r"\1", inner)
        inner = inner.replace("\\\\", "<br>")
        inner = re.sub(r"\\[a-zA-Z]+", "", inner)
        inner = inner.replace("{", "").replace("}", "")
        return f'<div style="text-align:center;margin:0.5em 0;"><i>{inner}</i></div>'
    text = re.sub(r"\\begin\{equation\*?\}(?:\s*\\begin\{split\})?(.*?)(?:\\end\{split\}\s*)?\\end\{equation\*?\}", eq_sub, text, flags=re.S)

    # math spans: apply map, then drop $ delimiters
    def math_sub(m):
        inner = m.group(1)
        for a, b in MATH_MAP:
            inner = inner.replace(a, b)
        inner = re.sub(r"\\(?:mathrm|text|mathit)\{([^}]*)\}", r"\1", inner)
        inner = re.sub(r"\\[a-zA-Z]+", "", inner)
        inner = inner.replace("{", "").replace("}", "")
        return inner
    text = re.sub(r"\$(.+?)\$", math_sub, text, flags=re.S)

    # commands with braced args (loop until none left, innermost-safe order)
    def replace_cmd(text, cmd, fmt):
        while True:
            r = cmd_arg(text, cmd, 0)
            if not r:
                return text
            content, a, b = r
            text = text[:a] + fmt(inline(content)) + text[b:]
        return text

    text = replace_cmd(text, "textbf", lambda c: f"<b>{c}</b>")
    text = replace_cmd(text, "textit", lambda c: f"<i>{c}</i>")
    text = replace_cmd(text, "emph", lambda c: f"<i>{c}</i>")
    text = replace_cmd(text, "texttt", lambda c: f"<code>{c}</code>")
    text = replace_cmd(text, "textsuperscript", lambda c: f"<sup>{c}</sup>")

    def cite_sub(m):
        nums = []
        for k in m.group(1).split(","):
            k = k.strip()
            if k in BIB_ORDER:
                nums.append(str(BIB_ORDER[k]))
        return "[" + ", ".join(nums) + "]" if nums else "[?]"
    text = re.sub(r"\\cite\{([^}]*)\}", cite_sub, text)
    text = re.sub(r"\\ref\{([^}]*)\}",
                  lambda m: LABEL_MAP.get(m.group(1), m.group(1)), text)
    text = re.sub(r"\\label\{[^}]*\}", "", text)
    text = text.replace("\\skarm", "skarm")
    text = text.replace("~", " ")
    text = text.replace("---", "\u2014").replace("--", "\u2013")
    text = text.replace("``", "\u201c").replace("''", "\u201d")
    for a, b in MATH_MAP:
        text = text.replace(a, b)
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    text = text.replace("{", "").replace("}", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_table(env):
    cap = cmd_arg(env, "caption", 0)
    if not cap:
        m_captab = re.search(r"\\captionof\{table\}\{", env)
        if m_captab:
            content, _ = extract_braced(env, m_captab.end() - 1)
            cap = (content, m_captab.start(), _)
    caption = inline(cap[0]) if cap else ""

    m = re.search(r"\\begin\{(tabular\*?|longtable)\}", env)
    if not m:
        return caption, []

    endtok = f"\\end{{{m.group(1)}}}"
    end_idx = env.index(endtok) if endtok in env else len(env)

    brace_start = env.find("{", m.end())
    if brace_start != -1 and brace_start < end_idx:
        _, body_start = extract_braced(env, brace_start)
        body = env[body_start:end_idx]
    else:
        body = env[m.end():end_idx]

    if m.group(1) == "longtable":
        if r"\endlastfoot" in body:
            pre, _, post = body.partition(r"\endlastfoot")
            header_part = pre.split(r"\endfirsthead")[0] if r"\endfirsthead" in pre else pre.split(r"\endhead")[0]
            header_part = re.sub(r"\\caption(?:of\{table\})?\{.*?\}", "", header_part, flags=re.S)
            header_part = re.sub(r"\\label\{.*?\}", "", header_part)
            body = header_part + "\n" + post
        else:
            body = re.sub(r"\\end(first|last)?head|\\end(first|last)?foot", "", body)

    body = re.sub(r"\\end(first|last)?head|\\end(first|last)?foot", "", body)

    rows = []
    for raw in body.split(r"\\"):
        raw = re.sub(r"\\(toprule|midrule|bottomrule|hline)\b", "", raw)
        raw = re.sub(r"\\cline\{[^}]*\}", "", raw)
        raw = re.sub(r"\\caption(?:of\{table\})?\{.*?\}", "", raw, flags=re.S)
        raw = re.sub(r"\\label\{.*?\}", "", raw)
        raw = raw.strip()
        if not raw or raw.startswith(r"\end"):
            continue
        raw = raw.replace(r"\&", "\x00AMP\x00")
        cells = [inline(c.replace("\x00AMP\x00", r"\&")) for c in raw.split("&")]
        if any(c.strip() for c in cells):
            rows.append(cells)
    return caption, rows


def find_matching_end(s, start_pos, env_name):
    """Find the index after \\end{env_name} matching \\begin{env_name} at start_pos."""
    begin_str = f"\\begin{{{env_name}}}"
    assert s.startswith(begin_str, start_pos)
    pattern = re.compile(rf"\\begin\{{{re.escape(env_name)}\}}|\\end\{{{re.escape(env_name)}\}}")
    depth = 0
    for m in pattern.finditer(s, start_pos):
        if m.group(0) == begin_str:
            depth += 1
        else:
            depth -= 1
            if depth == 0:
                return m.end()
    raise ValueError(f"unmatched \\begin{{{env_name}}}")


def render_list(env_text, list_tag="ul"):
    """Parse list items, supporting nested lists and stripping length commands."""
    env_text = re.sub(r"\\setlength\{[^}]*\}\{[^}]*\}", "", env_text)
    env_text = re.sub(r"\\vspace\*?\{[^}]*\}", "", env_text)

    sub_map = {}
    sub_idx = 0
    while True:
        m_sub = re.search(r"\\begin\{(itemize|enumerate)\}", env_text)
        if not m_sub:
            break
        sub_name = m_sub.group(1)
        sub_end = find_matching_end(env_text, m_sub.start(), sub_name)
        sub_inner = env_text[m_sub.start() + len(f"\\begin{{{sub_name}}}"):sub_end - len(f"\\end{{{sub_name}}}")]
        sub_tag = "ol" if sub_name == "enumerate" else "ul"
        sub_rendered = render_list(sub_inner, sub_tag)
        ph = f"\x00SUBLIST_{sub_idx}\x00"
        sub_map[ph] = sub_rendered
        sub_idx += 1
        env_text = env_text[:m_sub.start()] + ph + env_text[sub_end:]

    items = []
    parts = re.split(r"\\item\b\s*", env_text)
    for p in parts[1:]:
        p = p.strip()
        if not p:
            continue
        for ph, sub_html in sub_map.items():
            if ph in p:
                p = p.replace(ph, f"\n{sub_html}")
        if "\n<ul" in p or "\n<ol" in p:
            main_p, _, sub_p = p.partition("\n<")
            rendered = f"{inline(main_p)} <{sub_p}"
        else:
            rendered = inline(p)
        items.append(rendered)

    return f"<{list_tag}>" + "".join(f"<li>{it}</li>" for it in items if it.strip()) + f"</{list_tag}>"



def main():
    full = strip_comments(open(SRC, encoding='utf-8').read())
    # Title and authors live in the preamble, before \begin{document}.
    t = cmd_arg(full, "title", 0)
    title = inline(t[0]) if t else "Paper draft"
    au = cmd_arg(full, "author", 0)
    author_html = ""
    if au:
        author_html = inline(au[0].replace("\\\\", "<br>"))

    raw = full[full.index("\\begin{document}") + len("\\begin{document}"):]
    raw = raw.replace("\\end{document}", "")

    # bibliography first (for citation numbering)
    bib = []
    m = re.search(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", raw, flags=re.S)
    if m:
        bib_text = m.group(0)[:m.group(0).index("\\end{thebibliography}")]
        for part in re.split(r"\\bibitem\{", bib_text)[1:]:
            key, _, rest = part.partition("}")
            bib.append((key, rest.strip()))
            BIB_ORDER[key] = len(bib)
        raw = raw.replace(m.group(0), "\x00BIB\x00")

    ab = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", raw, flags=re.S)
    abstract = ab.group(1) if ab else ""
    raw = re.sub(r"\\maketitle", "", raw)
    raw = re.sub(r"\\twocolumn\[\{?", "", raw)
    raw = re.sub(r"\}\]\s*", "", raw, count=1)
    raw = re.sub(r"^\s*\}\]\s*$", "", raw, flags=re.M)
    raw = re.sub(r"\\begin\{abstract\}.*?\\end\{abstract\}", "", raw,
                 count=1, flags=re.S)
    raw = re.sub(r"\\enlargethispage\*?\{[^}]*\}", "", raw)
    raw = re.sub(r"\\vspace\*?\{[^}]*\}", "", raw)
    raw = re.sub(r"\\hspace\*?\{[^}]*\}", "", raw)
    raw = re.sub(r"\\setlength\{[^}]*\}\{[^}]*\}", "", raw)
    raw = re.sub(r"\\renewcommand\{[^}]*\}\{[^}]*\}", "", raw)
    raw = re.sub(r"\\par\b", "\n\n", raw)
    raw = re.sub(r"\\(noindent|smallskip|medskip|bigskip|centering|arraystretch)\b", "", raw)
    raw = re.sub(r"\\(footnotesize|scriptsize|small|normalsize|large|Large)\b", "", raw)

    raw = raw.replace("\\begin{landscape}", "").replace("\\end{landscape}", "")
    # token scan over the body
    token_re = re.compile(
        r"\\section\*\{|\\section\{|\\subsection\{|\\begin\{table\*?\}|\\begin\{longtable\}|"
        r"\\begin\{figure\*?\}|\\begin\{itemize\}|\\begin\{enumerate\}|\\placeholder\{|\\appendix|"
        r"\x00BIB\x00")
    out = []
    tab_no = fig_no = 0
    sec_no = 0
    app_no = 0
    in_appendix = False

    def flush_text(chunk):
        for para in re.split(r"\n\s*\n", chunk):
            p = para.strip()
            if not p or p.startswith("\\date"):
                continue
            if "\\begin{center}" in p or "\\end{center}" in p:
                continue
            p = inline(p)
            if not p or p in ("]", "}", "}]"):
                continue
            if p.startswith("<div") and p.endswith("</div>"):
                out.append(p)
            else:
                out.append(f"<p>{p}</p>")

    pos = 0
    for m in token_re.finditer(raw):
        if m.start() < pos:
            continue
        flush_text(raw[pos:m.start()])
        tok = m.group(0)
        if tok == "\\appendix":
            in_appendix = True
            pos = m.end()
            continue
        if tok == "\x00BIB\x00":
            out.append("<h1>References</h1><ol>")
            for key, rest in bib:
                out.append(f"<li>{inline(rest)}</li>")
            out.append("</ol>")
            pos = m.end()
            continue
        if tok in ("\\section*{", "\\section{", "\\subsection{"):
            content, end = extract_braced(raw, m.end() - 1)
            title_text = inline(content)
            if tok == "\\section*{":
                out.append(f"<h1>{title_text}</h1>")
            elif tok == "\\section{":
                if in_appendix:
                    app_no += 1
                    out.append(f"<h1>Appendix {chr(64 + app_no)}. {title_text}</h1>")
                else:
                    sec_no += 1
                    out.append(f"<h1>{sec_no}. {title_text}</h1>")
            else:
                out.append(f"<h2>{title_text}</h2>")
            pos = end
            continue
        if tok == "\\placeholder{":
            content, end = extract_braced(raw, m.end() - 1)
            out.append(f'<div class="ph"><b>EMPTY — to be filled from '
                       f'campaign results</b><br>{inline(content)}</div>')
            pos = end
            continue
        if tok in ("\\begin{table}", "\\begin{table*}"):
            endtok = "\\end{table*}" if tok.endswith("*}") else "\\end{table}"
            end = raw.index(endtok, m.start()) + len(endtok)
            env = raw[m.start():end]
            caption, rows = parse_table(env)
            tab_no += 1
            html = [f'<table><caption>Table {tab_no}. {caption}</caption>']
            for ri, row in enumerate(rows):
                tag = "th" if ri == 0 else "td"
                html.append("<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in row) + "</tr>")
            html.append("</table>")
            out.append("\n".join(html))

            m_capfig = re.search(r"\\captionof\{figure\}\{", env)
            if m_capfig:
                fig_cap_raw, _ = extract_braced(env, m_capfig.end() - 1)
                fig_caption = inline(fig_cap_raw)
                fig_no += 1
                ig = re.search(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", env)
                if ig:
                    out.append(f'<div class="fig"><img src="{ig.group(1)}" '
                               f'style="max-width:100%">'
                               f'<div class="cap">Figure {fig_no}. {fig_caption}</div></div>')
                else:
                    out.append(f'<div class="fig"><div class="figbox">Figure {fig_no} '
                               f'— diagram (renders in the compiled PDF)</div>'
                               f'<div class="cap">Figure {fig_no}. {fig_caption}</div></div>')
            pos = end
            continue
        if tok == "\\begin{longtable}":
            end = raw.index("\\end{longtable}", m.start()) + len("\\end{longtable}")
            caption, rows = parse_table(raw[m.start():end])
            tab_no += 1
            html = [f'<table><caption>Table {tab_no}. {caption}</caption>']
            for ri, row in enumerate(rows):
                tag = "th" if ri == 0 else "td"
                html.append("<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in row) + "</tr>")
            html.append("</table>")
            out.append("\n".join(html))
            pos = end
            continue
        if tok in ("\\begin{figure}", "\\begin{figure*}"):
            endtok = "\\end{figure*}" if tok.endswith("*}") else "\\end{figure}"
            end = raw.index(endtok, m.start()) + len(endtok)
            env = raw[m.start():end]
            cap = cmd_arg(env, "caption", 0)
            caption = inline(cap[0]) if cap else ""
            fig_no += 1
            ig = re.search(r"\\includegraphics\[[^\]]*\]\{([^}]*)\}", env)
            if ig:
                out.append(f'<div class="fig"><img src="{ig.group(1)}" '
                           f'style="max-width:100%">'
                           f'<div class="cap">Figure {fig_no}. {caption}</div></div>')
            else:
                out.append(f'<div class="fig"><div class="figbox">Figure {fig_no} '
                           f'— diagram (renders in the compiled PDF)</div>'
                           f'<div class="cap">Figure {fig_no}. {caption}</div></div>')
            pos = end
            continue
        if tok in ("\\begin{itemize}", "\\begin{enumerate}"):
            env_name = "itemize" if tok == "\\begin{itemize}" else "enumerate"
            list_tag = "ul" if env_name == "itemize" else "ol"
            end = find_matching_end(raw, m.start(), env_name)
            env_inner = raw[m.start() + len(tok):end - len(f"\\end{{{env_name}}}")]
            out.append(render_list(env_inner, list_tag))
            pos = end
            continue
        pos = m.end()
    flush_text(raw[pos:])

    # abstract may itself contain a placeholder
    abstract_html = abstract
    while "\\placeholder{" in abstract_html:
        r = cmd_arg(abstract_html, "placeholder", 0)
        content, a, b = r
        abstract_html = (abstract_html[:a] +
                         f'<div class="ph"><b>EMPTY — to be filled from campaign '
                         f'results</b><br>{inline(content)}</div>' +
                         abstract_html[b:])
    abstract_html = "<p>" + "</p><p>".join(
        inline(p) for p in abstract_html.split("\n\n") if p.strip()) + "</p>"
    abstract_html = abstract_html.replace("<p></p>", "")

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>{title} — preview</title>
<style>
body {{ font-family: Georgia, 'Times New Roman', serif; max-width: 780px;
  margin: 2em auto; padding: 0 1.2em; color: #111; line-height: 1.55; }}
h1 {{ font-size: 1.35em; margin-top: 1.6em; border-bottom: 1px solid #ccc;
  padding-bottom: 0.2em; }}
h2 {{ font-size: 1.12em; margin-top: 1.3em; }}
.title {{ text-align: center; font-size: 1.6em; font-weight: bold;
  line-height: 1.3; margin-bottom: 0.4em; }}
.authors {{ text-align: center; color: #333; margin-bottom: 1.5em; }}
.abstract {{ margin: 0 2em 1.5em 2em; }}
.abstract h1 {{ border: none; text-align: center; font-size: 1.1em; }}
table {{ border-collapse: collapse; margin: 1em auto; font-size: 0.85em; }}
caption {{ caption-side: top; font-style: italic; margin-bottom: 0.4em;
  text-align: left; }}
th, td {{ border: 1px solid #999; padding: 4px 8px; text-align: left; }}
th {{ background: #eee; }}
.ph {{ border: 2px dashed #c47f00; background: #fff8e6; padding: 0.8em 1em;
  margin: 1em 0; }}
.fig {{ margin: 1.2em 0; }}
.figbox {{ background: #f0f0f0; border: 1px solid #bbb; padding: 1.2em;
  text-align: center; color: #555; font-style: italic; }}
.cap {{ font-size: 0.9em; margin-top: 0.3em; }}
code {{ background: #f4f4f4; padding: 0 3px; }}
.note {{ background: #eef4ff; border-left: 4px solid #3366cc; padding: 0.7em 1em;
  margin-bottom: 1.5em; font-family: sans-serif; font-size: 0.92em; }}
</style></head><body>
<div class="note"><b>Paper preview</b> — readable rendering of paper.tex
(complete results for all 25 datasets, PC campaign 2026-10-30, two-column
journal layout). The compiled PDF remains the canonical version.</div>
<div class="title">{title}</div>
<div class="authors">{author_html}</div>
<div class="abstract"><h1>Abstract</h1>{abstract_html}</div>
{''.join(out)}
</body></html>
"""
    open(DST, "w", encoding="utf-8").write(html)
    print(f"wrote {DST} ({len(html)} chars)")


if __name__ == "__main__":
    main()
