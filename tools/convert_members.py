#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Rewrites trait-dialect IR text from the nested form to the members form.

Nested form: a `trait.trait` declares its associated types
(`trait.assoc_type`) and its methods (`trait.method`, bodiless when required,
bodied when a default) in its region, beside evidence ops over its block
argument; a `trait.impl` contains its bindings (`trait.assoc_type @A = T`),
its methods and its evidence; a method is called through
`trait.method.call %c @m(...)`.

Members form: a trait is a region-less declaration naming its member
functions in declaration order (`members [@T.m, default @T.n]`) and counting
each associated type's own parameters (`assoc {A = 0}`); an impl keeps its
evidence region, binds its associated types (`bindings {A = T}`) and pairs
each member of its trait with the function implementing it
(`members [@T.m -> @I.m, @T.n -> @T.n]`); every member is a module-level
`func.func` led by its declaration's claims, and a call names the member it
calls (`trait.func.call @T.m(%c, ...)`).

The rewrite reads text: strings, comments and brackets are tracked, nothing
is parsed by MLIR. Diagnostic-expectation comments, FileCheck lines and
anything the rewrite cannot decide are reported on stderr, one
`<kind> <file>:<line>` line each.

Usage:
  convert_members.py <in> <out>          (`-` for stdout)
  convert_members.py --in-place <file>...
"""

import bisect
import re
import sys

STRING = r'"(?:[^"\\\n]|\\.)*"'
COMMENT = r'//[^\n]*'
STRING_OR_COMMENT = re.compile(STRING + '|' + COMMENT)
COMMENT_ONLY = re.compile(STRING + '|(' + COMMENT + ')')
BRACKET = re.compile(r'[{}()\[\]<>]')
OPENERS = {'{': '}', '(': ')', '[': ']', '<': '>'}
CLOSERS = {'}': '{', ')': '(', ']': '[', '>': '<'}
SSA_NAME = re.compile(r'%[\w$.\-]+')
SSA_USE = re.compile(r'%[\w$.\-]+(?:#\d+)?')
SYMBOL = re.compile(r'@(?:' + STRING + r'|[\w$.\-]+)')
BARE_ID = re.compile(r'[A-Za-z_][\w$.]*\Z')
ALIAS_TOKEN = re.compile(r'!([A-Za-z_][\w$.]*)')
ALIAS_DEF = re.compile(r'^[ \t]*!([A-Za-z_][\w$.]*)[ \t]*=[ \t]*(.*?)[ \t]*$', re.M)
POLY = re.compile(r'(![A-Za-z_][\w$]*\.poly<)(\d+)(?=>)')
PARAMETER = re.compile(r'![A-Za-z_][\w$]*\.poly<(\d+)>')
SPLIT_MARK = re.compile(r'^// -----[^\n]*\n?', re.M)
MID_LINE_OP = re.compile(
    r'(?:%[\w$.\-]+(?::\d+)?\s*=\s*)?trait\.(?:method|assoc_type|return|allege|'
    r'derive|project|witness|coerce)\b')
OP_START = re.compile(
    r'%[\w$.\-]+(?::\d+)?(?:\s*,\s*%[\w$.\-]+(?::\d+)?)*\s*=(?!=)'
    r'|[A-Za-z_][\w$]*\.[\w$.]+'
    r'|"'
    r'|\^')
RESULTS_AND_NAME = re.compile(
    r'(?:(%[\w$.\-]+(?::\d+)?(?:\s*,\s*%[\w$.\-]+(?::\d+)?)*)\s*=\s*)?'
    r'("[^"]*"|[A-Za-z_][\w$]*(?:\.[\w$]+)+)')
EXPECTATION = re.compile(r'//\s*expected-(?:error|note|warning|remark)')
DIRECTIVE = re.compile(
    r'//\s*[A-Z][A-Z0-9_-]*(?:-NEXT|-LABEL|-DAG|-NOT|-SAME|-EMPTY|-COUNT-\d+)?:')
OLD_FORM_IN_CHECK = re.compile(
    r'trait\.method\b|trait\.assoc_type|_\{\{h\[0-9a-f\]\+\}\}_\w|_h[0-9a-f]{16}_\w')
VISIBILITY = re.compile(r'(private|public|nested)\b')


class ConvertError(Exception):
    pass


def unquote(symbol):
    """The name a symbol reference `@name` or `@"name"` names, and whether it
    was quoted."""
    body = symbol[1:]
    if body.startswith('"'):
        return body[1:-1], True
    return body, False


def symbol_ref(name, quoted=False):
    if quoted or not BARE_ID.match(name):
        return '@"' + name + '"'
    return '@' + name


def member_ref(owner, member):
    """The symbol of `member` of the declaration `owner` (two symbol
    references as written): the owner's name, a dot, the member's name."""
    owner_name, owner_quoted = unquote(owner)
    member_name, member_quoted = unquote(member)
    return symbol_ref(owner_name + '.' + member_name,
                      owner_quoted or member_quoted)


def dictionary_key(name):
    return name if BARE_ID.match(name) else '"' + name + '"'


class Text:
    """One chunk of IR text with its strings and comments blanked (`clean`),
    its comments alone blanked (`code`), and every bracket paired."""

    def __init__(self, text):
        self.text = text
        self.clean = STRING_OR_COMMENT.sub(lambda m: ' ' * len(m.group()), text)
        self.code = COMMENT_ONLY.sub(
            lambda m: ' ' * len(m.group()) if m.group(1) else m.group(), text)
        self.newlines = [m.start() for m in re.finditer('\n', text)]
        self.match = {}
        stack = []
        clean = self.clean
        for m in BRACKET.finditer(clean):
            c = m.group()
            p = m.start()
            if c == '>' and p > 0 and clean[p - 1] == '-':
                continue
            if c in OPENERS:
                stack.append(p)
                continue
            if not stack or clean[stack[-1]] != CLOSERS[c]:
                raise ConvertError(f'{self.line(p)}: unbalanced {c!r}')
            self.match[stack.pop()] = p
        if stack:
            raise ConvertError(f'{self.line(stack[-1])}: unclosed '
                               f'{clean[stack[-1]]!r}')

    def line(self, pos):
        return bisect.bisect_right(self.newlines, pos - 1) + 1

    def line_start(self, pos):
        return self.text.rfind('\n', 0, pos) + 1

    def line_end(self, pos):
        end = self.text.find('\n', pos)
        return len(self.text) if end < 0 else end

    def skip_ws(self, p):
        code = self.code
        n = len(code)
        while p < n and code[p] in ' \t\r\n':
            p += 1
        return p

    def skip_blank(self, p):
        """Skips spaces and tabs on one line."""
        clean = self.clean
        n = len(clean)
        while p < n and clean[p] in ' \t':
            p += 1
        return p

    def closing(self, p):
        if p not in self.match:
            raise ConvertError(f'{self.line(p)}: expected a bracket')
        return self.match[p]

    def expect(self, p, token):
        if not self.clean.startswith(token, p):
            got = self.text[p:p + 20].split('\n')[0]
            raise ConvertError(f'{self.line(p)}: expected {token!r}, found '
                               f'{got!r}')
        return p + len(token)

    def type_end(self, p):
        """The end of the type starting at `p`: its brackets skipped, it ends
        at whitespace, a comma or an enclosing closer."""
        clean = self.clean
        n = len(clean)
        while p < n:
            c = clean[p]
            if c in OPENERS and p in self.match:
                p = self.match[p] + 1
                continue
            if c in ' \t\r\n,)]}>':
                break
            p += 1
        return p

    def split_top(self, a, b):
        """The comma-separated entries of [a, b) at bracket depth zero, as
        stripped texts."""
        out = []
        start = p = a
        clean = self.clean
        while p < b:
            c = clean[p]
            if c in OPENERS and p in self.match:
                p = self.match[p] + 1
                continue
            if c == ',':
                out.append(self.text[start:p].strip())
                start = p + 1
            p += 1
        last = self.text[start:b].strip()
        if last:
            out.append(last)
        return out

    def keyword_at(self, p, word):
        end = p + len(word)
        return (self.clean.startswith(word, p) and
                (end >= len(self.clean) or not
                 (self.clean[end].isalnum() or self.clean[end] in '_$.')))

    def items(self, a, b):
        """The ops of the region interior [a, b), each as a (start, end)
        span: an op starts at a line's first token when that token begins an
        op (results and `=`, a dotted op name, a generic op's string, a block
        label), or after whitespace before an op of this dialect, and runs
        over its brackets and any continuation lines."""
        out = []
        p = a
        clean = self.clean
        while True:
            p = self.skip_ws(p)
            if p >= b:
                break
            start = q = p
            while q < b:
                c = clean[q]
                if c in OPENERS and q in self.match:
                    q = self.match[q] + 1
                    continue
                if c == '\n':
                    r = self.skip_ws(q)
                    if r >= b or OP_START.match(self.text, r):
                        break
                elif c in ' \t' and q > start and clean[q - 1] not in ' \t=':
                    r = self.skip_blank(q)
                    if r < b and clean[r] != '\n' and MID_LINE_OP.match(clean, r):
                        q = r
                        break
                q += 1
            end = min(q, b)
            while end > start and self.text[end - 1] in ' \t\r\n':
                end -= 1
            out.append((start, end))
            p = q
        return out


def chunk_aliases(t):
    """The type aliases a chunk defines at its top level, name to text."""
    aliases = {}
    for m in ALIAS_DEF.finditer(t.text):
        if t.clean[m.start(1) - 1] != '!':
            continue
        value = t.clean[m.start(2):m.end(2)].strip()
        aliases[m.group(1)] = value
    return aliases


def expand_aliases(text, aliases, depth=0):
    if depth > 32:
        raise ConvertError('alias expansion does not end')

    def repl(m):
        name = m.group(1)
        if name in aliases:
            return expand_aliases(aliases[name], aliases, depth + 1)
        return m.group(0)
    return ALIAS_TOKEN.sub(repl, text)


def claim_trait(type_text, aliases):
    """The trait symbol reference a claim type's application names, the claim
    written in full, short (`<@T[...]>`) or as an alias."""
    expanded = expand_aliases(type_text.strip(), aliases)
    m = re.match(r'(?:!trait\.claim)?<\s*(@(?:' + STRING + r'|[\w$.\-]+))',
                 expanded)
    return m.group(1) if m else None


def poly_labels(text, aliases):
    return [int(m.group(2)) for m in POLY.finditer(expand_aliases(text, aliases))]


class Report:
    def __init__(self, fname, out):
        self.fname = fname
        self.out = out
        self.line_offset = 0

    def __call__(self, kind, line=None, extra=None):
        where = self.fname if line is None else f'{self.fname}:{line + self.line_offset}'
        text = f'{kind} {where}' + (f' {extra}' if extra is not None else '')
        self.out.append(text)


class Method:
    """A `trait.method` read off its text: symbol, arguments, results,
    attributes, body and location spans."""


class EvidenceOp:
    """An op of a declaration's body other than a method, an associated type
    or the return: its span, the names it defines and the names it reads."""


class Declaration:
    """A `trait.trait` or `trait.impl` read off its text."""


def parse_argument(text):
    """(name, type, rest) of an argument `%name: type ...`, or (None, text, '')
    of a type-only argument."""
    m = re.match(r'(%[\w$.\-]+)\s*:\s*', text)
    if not m:
        return None, text, ''
    sub = Text(text)
    end = sub.type_end(m.end())
    return m.group(1), text[m.end():end], text[end:].strip()


def parse_method(t, start, end, report):
    """Reads `trait.method @m(args) [-> results] [attributes {..}] [{body}]
    [loc(..)]` spanning [start, end)."""
    m = Method()
    p = t.skip_ws(start + len('trait.method'))
    v = VISIBILITY.match(t.clean, p)
    if v:
        report('method-visibility', t.line(p))
        p = t.skip_ws(v.end())
    sym = SYMBOL.match(t.text, p)
    if not sym:
        raise ConvertError(f'{t.line(p)}: a method names no symbol')
    m.symbol = sym.group()
    p = t.skip_ws(sym.end())
    t.expect(p, '(')
    close = t.closing(p)
    m.args = t.split_top(p + 1, close)
    p = t.skip_ws(close + 1)
    m.results = None
    if t.clean.startswith('->', p):
        p = t.skip_ws(p + 2)
        r_end = t.closing(p) + 1 if t.clean[p] == '(' else t.type_end(p)
        m.results = t.text[p:r_end]
        p = t.skip_ws(r_end)
    m.attrs = None
    if t.keyword_at(p, 'attributes'):
        p = t.skip_ws(p + len('attributes'))
        t.expect(p, '{')
        m.attrs = t.text[p:t.closing(p) + 1]
        p = t.skip_ws(t.closing(p) + 1)
    m.body = None
    if p < end and t.clean[p] == '{':
        m.body = (p, t.closing(p))
        p = t.skip_ws(t.closing(p) + 1)
    m.loc = None
    if p < end and t.clean.startswith('loc(', p):
        m.loc = t.text[p:t.closing(p + 3) + 1]
        p = t.skip_ws(t.closing(p + 3) + 1)
    if p < end:
        raise ConvertError(f'{t.line(p)}: unexpected text after a method: '
                           f'{t.text[p:end][:40]!r}')
    m.span = (start, end)
    return m


def parse_assoc(t, start, end):
    """Reads `trait.assoc_type @A [<[params]>] [= type] [loc(..)]`: the name,
    the parameter texts or None, the bound type text or None."""
    p = t.skip_ws(start + len('trait.assoc_type'))
    sym = SYMBOL.match(t.text, p)
    if not sym:
        raise ConvertError(f'{t.line(p)}: an associated type names no symbol')
    name, _ = unquote(sym.group())
    p = sym.end()
    params = None
    if t.clean.startswith('<', p):
        close = t.closing(p)
        inner = t.skip_ws(p + 1)
        t.expect(inner, '[')
        params = t.split_top(inner + 1, t.closing(inner))
        p = close + 1
    p = t.skip_ws(p)
    bound = None
    if p < end and t.clean[p] == '=':
        p = t.skip_ws(p + 1)
        b_end = t.type_end(p)
        bound = t.text[p:b_end]
        p = t.skip_ws(b_end)
    if p < end and t.clean.startswith('loc(', p):
        p = t.skip_ws(t.closing(p + 3) + 1)
    if p < end:
        raise ConvertError(f'{t.line(p)}: unexpected text after an associated '
                           f'type: {t.text[p:end][:40]!r}')
    return name, params, bound


def parse_declaration(t, k, kind):
    """Reads the trait or impl op whose keyword stands at `k`."""
    d = Declaration()
    d.kind = kind
    d.keyword = k
    p = t.skip_ws(k + len('trait.' + kind))
    d.visibility = ''
    v = VISIBILITY.match(t.clean, p)
    if v:
        d.visibility = v.group(1)
        p = t.skip_ws(v.end())
    sym = SYMBOL.match(t.text, p)
    if not sym:
        raise ConvertError(f'{t.line(p)}: a trait.{kind} names no symbol')
    d.symbol = sym.group()
    p = t.skip_ws(sym.end())
    t.expect(p, '(')
    close = t.closing(p)
    d.args_text = t.text[p + 1:close]
    d.args = t.split_top(p + 1, close)
    p = t.skip_ws(close + 1)
    d.results = None
    if t.clean.startswith('->', p):
        p = t.skip_ws(p + 2)
        r_end = t.closing(p) + 1 if t.clean[p] == '(' else t.type_end(p)
        d.results = t.text[p:r_end]
        p = t.skip_ws(r_end)
    d.attrs = None
    if t.keyword_at(p, 'attributes'):
        p = t.skip_ws(p + len('attributes'))
        t.expect(p, '{')
        d.attrs = t.text[p:t.closing(p) + 1]
        p = t.skip_ws(t.closing(p) + 1)
    t.expect(p, '{')
    d.body = (p, t.closing(p))
    p = d.body[1] + 1
    d.loc = None
    q = t.skip_blank(p)
    if t.clean.startswith('loc(', q):
        d.loc = t.text[q:t.closing(q + 3) + 1]
        p = t.closing(q + 3) + 1
    d.end = p
    d.indent = t.text[t.line_start(k):k]
    if d.indent.strip():
        d.indent = re.match(r'[ \t]*', d.indent).group()
    return d


def classify(t, start, end):
    m = RESULTS_AND_NAME.match(t.text, start)
    if not m:
        return None, [], None
    names = SSA_NAME.findall(m.group(1) or '')
    return m.group(2), names, m.end()


def read_body(t, d, report):
    """Splits a declaration's body into methods, associated types, evidence
    ops and its return."""
    d.methods = []
    d.assocs = []
    d.evidence = []
    d.returns = []
    d.others = []
    d.block_args = [parse_argument(arg)[0] for arg in d.args]
    a, b = d.body[0] + 1, d.body[1]
    for start, end in t.items(a, b):
        name, results, after = classify(t, start, end)
        if name == 'trait.method':
            meth = parse_method(t, start, end, report)
            # A method reads only what its declaration defines before it: a
            # value defined after it may share a name with one of its own.
            visible = set(d.block_args) | {n for item in d.evidence
                                           for n in item.results}
            meth.uses = set(SSA_NAME.findall(t.clean[start:end])) & visible
            d.methods.append(meth)
        elif name == 'trait.assoc_type':
            d.assocs.append((parse_assoc(t, start, end), (start, end)))
        elif name == 'trait.return':
            d.returns.append(((start, end),
                              set(SSA_NAME.findall(t.clean[start:end]))))
        elif name is None or name.startswith('"'):
            d.others.append((start, end))
        else:
            item = EvidenceOp()
            item.span = (start, end)
            item.op = name
            item.results = results
            item.uses = set(SSA_NAME.findall(t.clean[after:end]))
            d.evidence.append(item)


def attached_comments_start(t, start):
    """The start of the comment lines directly above the line of `start`, or that line's start where none stands there; `start` itself
    where something else stands before it on its line."""
    line = t.line_start(start)
    if t.text[line:start].strip():
        return start
    while line > 0:
        prev = t.line_start(line - 1)
        if not t.text[prev:line].strip().startswith('//'):
            break
        line = prev
    return line


def closure(seeds, by_name):
    """The evidence ops `seeds` read, transitively, in their original
    order."""
    chosen = set()
    pending = [n for n in seeds if n in by_name]
    while pending:
        item = by_name[pending.pop()]
        if id(item) in chosen:
            continue
        chosen.add(id(item))
        pending.extend(n for n in item.uses if n in by_name)
    return chosen


def rewrite_returns(t, a, b):
    """The text of [a, b) with each `trait.return` op named `return`."""
    out = []
    p = a
    for m in re.finditer(r'(?<![\w.$\-"])trait\.return\b', t.clean[a:b]):
        out.append(t.text[p:a + m.start()])
        out.append('return')
        p = a + m.end()
    out.append(t.text[p:b])
    return ''.join(out)


def dedent(text, shift):
    if shift <= 0:
        return text
    lines = text.split('\n')
    prefix = ' ' * shift
    for i in range(1, len(lines)):
        if lines[i].startswith(prefix):
            lines[i] = lines[i][shift:]
    return '\n'.join(lines)


def function_text(t, d, meth, symbol, leading):
    """The module-level function `symbol` the method `meth` of `d` becomes,
    led by the arguments `leading`, with the evidence ops of `d`'s body the
    method reads cloned at the top of its entry block.

    A cloned op keeps its result names: the method could read only names its
    declaration defines before it, and MLIR refuses a nested region that
    defines a name it can read, so no name the method defines is one a
    clone defines."""
    indent = d.indent
    if meth.body is None:
        args = list(leading)
        for arg in meth.args:
            name, ty, rest = parse_argument(arg)
            args.append(arg if name is None else (ty + (' ' + rest if rest else '')))
    else:
        args = list(leading) + list(meth.args)
    head = f'{indent}func.func private {symbol}({", ".join(args)})'
    if meth.results is not None:
        head += f' -> {meth.results}'
    if meth.attrs is not None:
        head += f' attributes {meth.attrs}'
    if meth.body is None:
        return head + (f' {meth.loc}' if meth.loc else '')
    method_indent = len(t.text[t.line_start(meth.span[0]):meth.span[0]])
    shift = method_indent - len(indent)
    open_, close = meth.body
    inner = dedent(rewrite_returns(t, open_ + 1, close), shift)
    by_name = {}
    for item in d.evidence:
        for n in item.results:
            by_name[n] = item
    chosen = closure(meth.uses, by_name)
    clones = [item for item in d.evidence if id(item) in chosen]
    if clones:
        body_lines = inner.split('\n')
        first = next((l for l in body_lines[1:] if l.strip()), indent + '  ')
        body_indent = re.match(r'[ \t]*', first).group()
        clone_text = ''.join(body_indent + t.text[item.span[0]:item.span[1]] + '\n'
                             for item in clones)
        if '\n' in inner:
            nl = inner.index('\n') + 1
            inner = inner[:nl] + clone_text + inner[nl:]
        else:
            inner = '\n' + clone_text + body_indent + inner.strip() + '\n' + indent
    elif '\n' not in inner:
        inner = '\n' + indent + '  ' + inner.strip() + '\n' + indent
    return head + ' {' + inner + '}' + (f' {meth.loc}' if meth.loc else '')


def note_expectations(t, a, b, kind, report):
    for m in EXPECTATION.finditer(t.text, a, b):
        report(kind, t.line(m.start()))


def convert_trait(t, d, aliases, report):
    self_name, self_type, _ = parse_argument(d.args[0]) if d.args else (None, None, '')
    if len(d.args) != 1 or self_name is None:
        raise ConvertError(f'{t.line(d.keyword)}: a trait takes one named block '
                           f'argument')
    header = f'trait.trait {d.visibility + " " if d.visibility else ""}{d.symbol}({self_type})'
    if d.results is not None:
        header += f' -> {d.results}'
    if d.assocs:
        entries = []
        for (name, params, bound), span in d.assocs:
            if bound is not None:
                report('trait-assoc-binding', t.line(span[0]))
            entries.append(f'{dictionary_key(name)} = {len(params) if params else 0}')
        header += ' assoc {' + ', '.join(entries) + '}'
    if d.methods:
        entries = []
        for meth in d.methods:
            ref = member_ref(d.symbol, meth.symbol)
            entries.append(('default ' if meth.body is not None else '') + ref)
        header += ' members [' + ', '.join(entries) + ']'
    if d.attrs is not None:
        header += f' attributes {d.attrs}'
    if d.loc:
        header += f' {d.loc}'

    # The evidence a trait's body computes travels into the defaults reading
    # it; what no default reads is dropped and named.
    read = set()
    by_name = {n: item for item in d.evidence for n in item.results}
    for meth in d.methods:
        if meth.body is not None:
            read |= closure(meth.uses, by_name)
    for item in d.evidence:
        if id(item) not in read:
            report('unused-trait-evidence', t.line(item.span[0]))
    for start, end in d.others:
        report('dropped-trait-op', t.line(start))

    # Comments in the body: those directly above a method travel with it;
    # every other comment line is kept above the trait.
    moved = []
    functions = []
    for meth in d.methods:
        c0 = attached_comments_start(t, meth.span[0])
        lead = t.text[c0:t.line_start(meth.span[0])]
        note_expectations(t, c0, meth.span[1], 'moved-expectation', report)
        moved.append((c0, meth.span[1]))
        leading = [d.args[0]] if meth.body is not None else [self_type]
        text = function_text(t, d, meth, member_ref(d.symbol, meth.symbol),
                             leading)
        functions.append(reindent_comments(lead, d.indent) + text)
    a, b = d.body[0] + 1, d.body[1]
    orphan = []
    for m in re.finditer(COMMENT, t.text[a:b]):
        pos = a + m.start()
        if t.clean[pos:pos + 2] != '  ' or t.text[pos:pos + 2] != '//':
            continue
        if any(s <= pos < e for s, e in moved):
            continue
        if t.text[t.line_start(pos):pos].strip():
            continue
        orphan.append(d.indent + m.group())
        if EXPECTATION.match(m.group()):
            report('moved-expectation', t.line(pos))
    before = ''.join(o + '\n' for o in orphan)
    return before, header, functions


def reindent_comments(lead, indent):
    if not lead:
        return ''
    return ''.join(indent + line.strip() + '\n' for line in lead.splitlines())


def impl_parameter_count(d, aliases):
    """The impl's own parameter count: its parameters are labelled 0 to
    n - 1, and its header and where clause spell each."""
    labels = []
    for arg in d.args:
        labels.extend(poly_labels(arg, aliases))
    return max(labels) + 1 if labels else 0


def positional_binding(bound, params, m, aliases):
    """The binding `bound` of a generic associated type whose own parameters
    `params` are written as type variables, with those parameters relabelled
    m, m + 1, ...: a binding's own parameters follow the impl's m parameters
    by position. None where a parameter is no type variable."""
    labels = []
    for param in params:
        label = PARAMETER.fullmatch(expand_aliases(param, aliases).strip())
        if not label:
            return None
        labels.append(int(label.group(1)))
    wanted = list(range(m, m + len(labels)))
    if labels == wanted:
        return bound
    mapping = dict(zip(labels, wanted))
    return POLY.sub(lambda mm: mm.group(1) + str(mapping.get(int(mm.group(2)),
                                                             int(mm.group(2)))),
                    expand_aliases(bound, aliases))


def convert_impl(t, d, traits, aliases, report):
    self_arg = d.args[0] if d.args else ''
    _, self_type, _ = parse_argument(self_arg)
    trait_ref = claim_trait(self_type or '', aliases)
    if trait_ref is None:
        raise ConvertError(f'{t.line(d.keyword)}: cannot read the trait of impl '
                           f'{d.symbol}')
    trait_name, _ = unquote(trait_ref)
    header = f'trait.impl {d.visibility + " " if d.visibility else ""}{d.symbol}({d.args_text})'
    if d.results is not None:
        report('impl-with-results', t.line(d.keyword))
        header = header + f' -> {d.results}'
    removals = []
    # Comment lines directly above an associated type or an evidence op the
    # body loses move above the impl, where a diagnostic about the binding
    # now stands.
    hoisted = []

    def hoist(span_start):
        c0 = attached_comments_start(t, span_start)
        line = t.line_start(span_start)
        if c0 < line:
            hoisted.append(t.text[c0:line])
            note_expectations(t, c0, line, 'moved-expectation', report)
        return c0

    if d.assocs:
        m = impl_parameter_count(d, aliases)
        entries = []
        for (name, params, bound), span in d.assocs:
            value = bound
            if bound is not None and params:
                value = positional_binding(bound, params, m, aliases)
                if value is None:
                    # The old line stays in the body, for a hand edit.
                    report('undecided-gat', t.line(span[0]))
                    continue
                if value != bound:
                    report('relabelled-gat', t.line(span[0]))
            removals.append((hoist(span[0]), span[1]))
            if bound is None:
                report('impl-assoc-without-binding', t.line(span[0]))
                continue
            entries.append(f'{dictionary_key(name)} = {value}')
        if entries:
            header += ' bindings {' + ', '.join(entries) + '}'

    own = {unquote(meth.symbol)[0]: meth for meth in d.methods}
    declared = traits.get(trait_name)
    pairs = []
    if declared is None:
        if d.methods:
            report('unknown-trait', t.line(d.keyword))
        for meth in d.methods:
            pairs.append(f'{member_ref(trait_ref, meth.symbol)} -> '
                         f'{member_ref(d.symbol, meth.symbol)}')
    else:
        for method_symbol, _ in declared:
            left = member_ref(trait_ref, method_symbol)
            name = unquote(method_symbol)[0]
            right = member_ref(d.symbol, method_symbol) if name in own else left
            pairs.append(f'{left} -> {right}')
        names = {unquote(s)[0] for s, _ in declared}
        for meth in d.methods:
            if unquote(meth.symbol)[0] not in names:
                report('impl-method-not-in-trait', t.line(meth.span[0]))
                pairs.append(f'{member_ref(trait_ref, meth.symbol)} -> '
                             f'{member_ref(d.symbol, meth.symbol)}')
    if pairs:
        header += ' members [' + ', '.join(pairs) + ']'
    if d.attrs is not None:
        header += f' attributes {d.attrs}'

    # Evidence a method reads moves into the method's function; an op stays
    # in the body while it has no use, or the return or an op staying reads
    # it.
    by_name = {n: item for item in d.evidence for n in item.results}
    return_uses = set()
    for _, uses in d.returns:
        return_uses |= uses
    staying = set()
    for item in reversed(d.evidence):
        users = [other for other in d.evidence
                 if other is not item and set(item.results) & other.uses]
        method_users = [meth for meth in d.methods
                        if set(item.results) & meth.uses]
        if set(item.results) & return_uses or any(id(u) in staying for u in users):
            staying.add(id(item))
        elif not users and not method_users:
            staying.add(id(item))
            report('unused-impl-evidence', t.line(item.span[0]))
    for item in d.evidence:
        if id(item) not in staying:
            removals.append((hoist(item.span[0]), item.span[1]))

    functions = []
    for meth in d.methods:
        c0 = attached_comments_start(t, meth.span[0])
        lead = t.text[c0:t.line_start(meth.span[0])]
        note_expectations(t, c0, meth.span[1], 'moved-expectation', report)
        removals.append((c0, meth.span[1]))
        text = function_text(t, d, meth, member_ref(d.symbol, meth.symbol),
                             d.args)
        functions.append(reindent_comments(lead, d.indent) + text)

    # The body keeps what is not removed.
    a, b = d.body[0] + 1, d.body[1]
    removals.sort()
    pieces = []
    p = a
    for s, e in removals:
        s = max(s, a)
        if s < p:
            continue
        pieces.append(t.text[p:s])
        p = e
        if t.text[p:t.line_end(p)].strip() == '':
            p = min(t.line_end(p) + 1, b) if t.line_end(p) < b else b
    pieces.append(t.text[p:b])
    interior = ''.join(pieces)
    if not interior.strip():
        interior = '\n' + d.indent
    elif not interior.endswith('\n' + d.indent) and '\n' in t.text[a:b]:
        interior = interior.rstrip() + '\n' + d.indent
    tail = f' {d.loc}' if d.loc else ''
    before = ''.join(reindent_comments(h, d.indent) for h in hoisted)
    return before, header + ' {' + interior + '}' + tail, functions


def collect_traits(t, decls):
    traits = {}
    for d in decls:
        if d.kind != 'trait':
            continue
        name, _ = unquote(d.symbol)
        members = [(meth.symbol, meth.body is not None) for meth in d.methods]
        traits.setdefault(name, [])
        traits[name].append(members)
    return traits


def convert_declarations(text, report):
    t = Text(text)
    aliases = chunk_aliases(t)
    for m in re.finditer(r'"trait\.(?:method|assoc_type|method\.call|trait|impl)"',
                         t.code):
        report('generic-form', t.line(m.start()))
    decls = []
    for m in re.finditer(r'(?<![\w.$\-"])trait\.(trait|impl)(?![\w.$\-])', t.clean):
        d = parse_declaration(t, m.start(), m.group(1))
        read_body(t, d, report)
        decls.append(d)
    for m in re.finditer(r'(?<![\w.$\-"])trait\.(?:method|assoc_type)(?![\w$.\-])', t.clean):
        if not any(d.body[0] < m.start() < d.body[1] for d in decls):
            report('stray-nested-op', t.line(m.start()))
    by_trait = collect_traits(t, decls)
    traits = {}
    for name, versions in by_trait.items():
        if len(versions) > 1 and any(v != versions[0] for v in versions):
            report('duplicate-trait-name', None, name)
        traits[name] = versions[0]
    out = []
    p = 0
    for d in decls:
        if d.keyword < p:
            raise ConvertError(f'{t.line(d.keyword)}: a declaration nests in '
                               f'another')
        if d.kind == 'trait':
            before, header, functions = convert_trait(t, d, aliases, report)
            start = t.line_start(d.keyword)
            if t.text[start:d.keyword].strip():
                start = d.keyword
                before = ''
            out.append(t.text[p:start])
            out.append(before)
            out.append(t.text[start:d.keyword])
            out.append(header)
        else:
            before, header, functions = convert_impl(t, d, traits, aliases, report)
            start = t.line_start(d.keyword)
            if t.text[start:d.keyword].strip():
                start = d.keyword
                before = ''
            out.append(t.text[p:start])
            out.append(before)
            out.append(t.text[start:d.keyword])
            out.append(header)
        for f in functions:
            out.append('\n' + f)
        p = d.end
    out.append(t.text[p:])
    return ''.join(out)


def convert_calls(text, report):
    """Rewrites every `trait.method.call` to a `trait.func.call` of the member
    it calls."""
    t = Text(text)
    aliases = chunk_aliases(t)
    out = []
    p0 = 0
    for m in re.finditer(r'(?<![\w.$\-"])trait\.method\.call\b', t.clean):
        k = m.start()
        p = t.skip_ws(m.end())
        recv = SSA_USE.match(t.text, p)
        if not recv:
            raise ConvertError(f'output line {t.line(p)}: a method call names '
                               f'no receiver')
        p = t.skip_ws(recv.end())
        sym = SYMBOL.match(t.text, p)
        if not sym:
            raise ConvertError(f'output line {t.line(p)}: a method call names '
                               f'no method')
        p = t.skip_ws(sym.end())
        t.expect(p, '(')
        args = t.text[p + 1:t.closing(p)].strip()
        p = t.skip_ws(t.closing(p) + 1)
        p = t.skip_ws(t.expect(p, ':'))
        claim_end = t.type_end(p)
        claim = t.text[p:claim_end]
        p = t.skip_ws(claim_end)
        p = t.skip_ws(t.expect(p, ','))
        t.expect(p, '(')
        inputs = t.text[p + 1:t.closing(p)].strip()
        p = t.skip_ws(t.closing(p) + 1)
        p = t.skip_ws(t.expect(p, '->'))
        r_end = t.closing(p) + 1 if t.clean[p] == '(' else t.type_end(p)
        results = t.text[p:r_end]
        end = r_end
        attrs = None
        q = t.skip_ws(end)
        if t.keyword_at(q, 'attributes'):
            q = t.skip_ws(q + len('attributes'))
            t.expect(q, '{')
            attrs = t.text[q:t.closing(q) + 1]
            end = t.closing(q) + 1
        q = t.skip_blank(end)
        loc = None
        if t.clean.startswith('loc(', q):
            loc = t.text[q:t.closing(q + 3) + 1]
            end = t.closing(q + 3) + 1
        trait_ref = claim_trait(claim, aliases)
        if trait_ref is None:
            raise ConvertError(f'output line {t.line(k)}: cannot read the trait '
                               f'of the receiver type {claim!r}')
        full_claim = '!trait.claim' + claim if claim.startswith('<') else claim
        operands = recv.group() + (', ' + args if args else '')
        types = full_claim + (', ' + inputs if inputs else '')
        new = (f'trait.func.call {member_ref(trait_ref, sym.group())}({operands})'
               + (f' {attrs}' if attrs else '')
               + f' : ({types}) -> {results}'
               + (f' {loc}' if loc else ''))
        out.append(t.text[p0:k])
        out.append(new)
        p0 = end
    out.append(t.text[p0:])
    return ''.join(out)


def report_checks(text, report):
    count = 0
    for line in text.split('\n'):
        stripped = line.strip()
        if DIRECTIVE.match(stripped) and OLD_FORM_IN_CHECK.search(stripped):
            count += 1
    if count:
        report('check-lines-mentioning-old-form', None, count)


def convert(text, fname, reports):
    report = Report(fname, reports)
    report_checks(text, report)
    out = []
    pos = 0
    line = 0
    # A `// -----` line separates modules only where a RUN line splits the
    # input file.
    split = re.search(r'^//\s*RUN:.*split-input-file', text, re.M)
    marks = list(SPLIT_MARK.finditer(text)) if split else []
    bounds = []
    for m in marks:
        bounds.append((pos, m.start(), m.group()))
        pos = m.end()
    bounds.append((pos, len(text), ''))
    for start, end, mark in bounds:
        chunk = text[start:end]
        report.line_offset = text.count('\n', 0, start)
        try:
            converted = convert_declarations(chunk, report)
            converted = convert_calls(converted, report)
        except ConvertError as e:
            raise ConvertError(f'{fname}:{e} (chunk at line {report.line_offset + 1})')
        out.append(converted)
        out.append(mark)
    return ''.join(out)


def main(argv):
    reports = []
    status = 0
    if argv[1:2] == ['--in-place']:
        jobs = [(f, f) for f in argv[2:]]
    elif len(argv) == 3:
        jobs = [(argv[1], argv[2])]
    else:
        sys.stderr.write(__doc__)
        return 2
    for src, dst in jobs:
        with open(src, encoding='utf-8', errors='surrogateescape') as f:
            text = f.read()
        try:
            converted = convert(text, src, reports)
        except ConvertError as e:
            sys.stderr.write(f'error {e}\n')
            status = 1
            continue
        if dst == '-':
            sys.stdout.write(converted)
        else:
            with open(dst, 'w', encoding='utf-8', errors='surrogateescape') as f:
                f.write(converted)
    for r in reports:
        sys.stderr.write(r + '\n')
    return status


if __name__ == '__main__':
    sys.exit(main(sys.argv))
