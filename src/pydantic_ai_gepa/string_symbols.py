"""Bridge Git commits and CandidateMaps through declared Python string constants.

A Git candidate whose editable surface is a set of top-level string constants
(``[acceptance.python_string_symbols]``) can be optimized by the in-process
text engines: :func:`candidate_from_commit` reads the constants into a
``CandidateMap`` and :func:`commit_from_candidate` writes a ``CandidateMap``
back as a new commit on top of a base, rewriting only those constants' source
spans. Commits are built with Git plumbing and a private index, so the
caller's working tree, index and branches are never touched.

Symbol paths are relative to ``repo``, the project directory inside the Git
repository, matching ``gepa.toml``. Version 1 supports exactly
``NAME = "..."`` and ``NAME: T = "..."`` at module level, including a
parenthesized implicit concatenation; anything else is refused.
"""

from __future__ import annotations

import ast
import codecs
import io
import os
import subprocess
import tempfile
import tokenize
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .gepa_graph.models import CandidateMap, ComponentValue

# `git rev-parse --local-env-vars`: variables that relocate the repository,
# index or object store. A parent hook's values must never redirect us.
_REPOSITORY_ENV = frozenset(
    {
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_CONFIG",
        "GIT_CONFIG_COUNT",
        "GIT_CONFIG_PARAMETERS",
        "GIT_DIR",
        "GIT_GRAFT_FILE",
        "GIT_IMPLICIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_NO_REPLACE_OBJECTS",
        "GIT_OBJECT_DIRECTORY",
        "GIT_PREFIX",
        "GIT_REPLACE_REF_BASE",
        "GIT_SHALLOW_FILE",
        "GIT_WORK_TREE",
    }
)
_FALLBACK_IDENTITY = ("pydantic-ai-gepa", "pydantic-ai-gepa@localhost")
_SELECTION_DOC = "v1 supports only top-level NAME = '...' string assignments"


class StringSymbolError(ValueError):
    """A declared string symbol cannot be read, rewritten or verified."""


def string_symbol_component(path: str, symbol: str) -> str:
    """Return the stable CandidateMap key for ``symbol`` in ``path``."""
    return f"py:{path}:{symbol}"


def masked_string_symbols(source: str, symbols: list[str]) -> str:
    """Dump ``source``'s AST with every string inside the named symbols masked.

    Two sources with equal masks differ at most in those strings. This is the
    trusted text scoring scope check shared with ``gepa`` git candidates.
    """
    tree = ast.parse(source)
    pending = set(symbols)
    for node in tree.body:
        names = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names = [node.name]
            doc_symbol = node.name + ".__doc__"
            if doc_symbol in pending:
                if ast.get_docstring(node) is None:
                    raise ValueError
                node.body[0] = ast.Expr(value=ast.Constant(value="<docstring>"))
                pending.remove(doc_symbol)
        elif isinstance(node, ast.Assign):
            names = [
                target.id for target in node.targets if isinstance(target, ast.Name)
            ]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        if not pending.intersection(names):
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Constant) and isinstance(child.value, str):
                child.value = "<instruction>"
        pending.difference_update(names)
    if pending:
        raise ValueError
    return ast.dump(tree, include_attributes=False)


def normalize_string_symbols(
    symbols: Mapping[str, Sequence[str]],
) -> dict[str, tuple[str, ...]]:
    """Validate a ``python_string_symbols`` mapping for the commit bridge."""
    if not isinstance(symbols, Mapping) or not symbols:
        raise StringSymbolError("Declare at least one Python file and symbol.")
    normalized: dict[str, tuple[str, ...]] = {}
    for path, names in symbols.items():
        if not _relative_python_path(path):
            raise StringSymbolError(
                f"{path!r} must be a relative .py path inside the project."
            )
        if isinstance(names, str) or not isinstance(names, Sequence) or not names:
            raise StringSymbolError(f"{path}: declare a non-empty list of symbols.")
        if len(set(names)) != len(names):
            raise StringSymbolError(f"{path}: symbols are declared more than once.")
        for name in names:
            if not isinstance(name, str) or not name.isidentifier():
                raise StringSymbolError(f"{path}:{name}: {_SELECTION_DOC}.")
        normalized[path] = tuple(names)
    return normalized


def resolve_commit(repo: str | Path, rev: str) -> str:
    """Resolve ``rev`` in ``repo`` to a full commit SHA."""
    return _Git(Path(repo)).commit(rev)


def candidate_from_commit(
    repo: str | Path,
    commit: str,
    symbols: Mapping[str, Sequence[str]],
) -> CandidateMap:
    """Read each declared string constant at ``commit`` into a CandidateMap.

    Keys are ``py:<path>:<NAME>`` and texts are the constants' values.
    Raises :class:`StringSymbolError` for a missing symbol or one that is not
    a single plain string constant assignment.
    """
    declared = normalize_string_symbols(symbols)
    git = _Git(Path(repo))
    sha = git.commit(commit)
    candidate: CandidateMap = {}
    for path, names in declared.items():
        spans = _string_spans(git.blob(sha, path)[1], path, names)
        for name in names:
            key = string_symbol_component(path, name)
            candidate[key] = ComponentValue(name=key, text=spans[name].value)
    return candidate


def commit_from_candidate(
    repo: str | Path,
    base_commit: str,
    symbols: Mapping[str, Sequence[str]],
    candidate: Mapping[str, ComponentValue],
    message: str,
) -> str:
    """Commit ``candidate``'s texts onto ``base_commit`` and return the new SHA.

    Only the declared constants' source spans are rewritten; components the
    candidate omits keep their base value, and leading/trailing whitespace of
    a base value (which ``ComponentValue`` strips) is kept around the new
    text. A candidate equal to the base returns ``base_commit``'s SHA rather
    than an empty commit. The result is verified with
    :func:`verify_string_symbol_commit` and read back before it is returned.
    """
    declared = normalize_string_symbols(symbols)
    known = {
        string_symbol_component(path, name)
        for path, names in declared.items()
        for name in names
    }
    unknown = sorted(set(candidate) - known)
    if unknown:
        raise StringSymbolError(f"Candidate has undeclared components: {unknown}.")
    git = _Git(Path(repo))
    base = git.commit(base_commit)
    expected: dict[str, str] = {}
    changed: dict[str, tuple[str, bytes]] = {}
    for path, names in declared.items():
        mode, source = git.blob(base, path)
        crlf = b"\r\n" in source
        spans = _string_spans(source, path, names)
        edits: list[tuple[int, int, bytes]] = []
        for name in names:
            span = spans[name]
            key = string_symbol_component(path, name)
            component = candidate.get(key)
            expected[key] = span.value.strip()
            if component is None or component.text == span.value.strip():
                continue
            expected[key] = component.text
            value = _reframe(span.value, component.text)
            # Keep a ``u`` prefix: the masked-AST scope check compares it.
            literal = (span.kind or "") + _literal(value)
            if crlf:
                # Raw newlines in a triple-quoted literal follow the file's
                # line endings; the parser reads both as "\n".
                literal = literal.replace("\n", "\r\n")
            edits.append((span.start, span.end, literal.encode()))
        if edits:
            for start, end, literal in sorted(edits, reverse=True):
                source = source[:start] + literal + source[end:]
            changed[path] = (mode, source)
    if not changed:
        return base
    sha = git.commit_tree(base, changed, message)
    verify_string_symbol_commit(repo, base, sha, symbols)
    read_back = candidate_from_commit(repo, sha, symbols)
    if {key: value.text for key, value in read_back.items()} != expected:
        raise StringSymbolError("Rewritten constants did not read back as written.")
    return sha


def verify_string_symbol_commit(
    repo: str | Path,
    base_commit: str,
    commit: str,
    symbols: Mapping[str, Sequence[str]],
) -> None:
    """Refuse a commit that differs from its base beyond the declared strings.

    The commit must descend from the base; only declared files may change,
    keeping their mode; each declared symbol must still be a plain string
    constant; the masked ASTs must match (the trusted text scoring check);
    and every byte outside the declared constants must be unchanged, so
    comments and formatting elsewhere are frozen too.
    """
    declared = normalize_string_symbols(symbols)
    git = _Git(Path(repo))
    base = git.commit(base_commit)
    sha = git.commit(commit)
    if not git.is_ancestor(base, sha):
        raise StringSymbolError("Commit does not descend from its base commit.")
    for status, old_mode, new_mode, path in git.changes(base, sha):
        relative = git.relative(path)
        if (
            relative is None
            or relative not in declared
            or status != "M"
            or old_mode != new_mode
            or new_mode not in ("100644", "100755")
        ):
            raise StringSymbolError(
                f"Commit changes {path} outside the declared string symbols."
            )
        names = declared[relative]
        before = git.blob(base, relative)[1]
        after = git.blob(sha, relative)[1]
        before_spans = _string_spans(before, relative, names)
        after_spans = _string_spans(after, relative, names)
        try:
            same_ast = masked_string_symbols(
                before.decode("utf-8-sig"), list(names)
            ) == masked_string_symbols(after.decode("utf-8-sig"), list(names))
        except (SyntaxError, ValueError, TypeError, RecursionError):
            same_ast = False
        if not same_ast or _outside(before, before_spans) != _outside(
            after, after_spans
        ):
            raise StringSymbolError(
                f"Commit changes {relative} outside the declared string symbols."
            )


@dataclass(frozen=True, slots=True)
class _Span:
    start: int
    end: int
    value: str
    kind: str | None = None


def _relative_python_path(path: object) -> bool:
    if not isinstance(path, str) or "\0" in path or "\\" in path:
        return False
    pure = PurePosixPath(path)
    return (
        not pure.is_absolute()
        and pure.suffix == ".py"
        and all(part not in ("", ".", "..", ".git") for part in path.split("/"))
    )


_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


def _scope_header(node: ast.AST) -> list[ast.AST]:
    """Expressions of a def/class/lambda header, evaluated in the outer scope."""
    header: list[ast.AST] = list(getattr(node, "decorator_list", []))
    if isinstance(node, ast.ClassDef):
        return [*header, *node.bases, *node.keywords]
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        arguments = node.args
        header.extend(arguments.defaults)
        header.extend(value for value in arguments.kw_defaults if value is not None)
        if not isinstance(node, ast.Lambda):
            parameters = [
                *arguments.posonlyargs,
                *arguments.args,
                *arguments.kwonlyargs,
                *(p for p in (arguments.vararg, arguments.kwarg) if p is not None),
            ]
            header.extend(p.annotation for p in parameters if p.annotation)
            if node.returns is not None:
                header.append(node.returns)
    return header


def _bound_names(node: ast.AST) -> set[str]:
    """Names a module-level statement binds or deletes in module scope.

    Nested function, class and lambda bodies are skipped, but their headers
    run in module scope. Comprehensions leak only assignment expressions.
    Over-counting is safe: it refuses a symbol rather than optimizing a
    constant the module may rebind.
    """
    names: set[str] = set()
    stack: list[ast.AST] = [node]
    while stack:
        current = stack.pop()
        if isinstance(
            current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
        ):
            if not isinstance(current, ast.Lambda):
                names.add(current.name)
            stack.extend(_scope_header(current))
            continue
        if isinstance(current, _COMPREHENSIONS):
            names.update(
                child.target.id
                for child in ast.walk(current)
                if isinstance(child, ast.NamedExpr)
                and isinstance(child.target, ast.Name)
            )
            continue
        if isinstance(current, ast.AnnAssign) and current.value is None:
            # A bare annotation (``X: str``) binds no value, but its annotation
            # expression is evaluated and may contain an assignment expression.
            stack.append(current.annotation)
            continue
        if isinstance(current, ast.Name) and isinstance(
            current.ctx, (ast.Store, ast.Del)
        ):
            names.add(current.id)
        elif isinstance(current, ast.alias):
            names.add(current.asname or current.name.split(".")[0])
        elif isinstance(current, ast.ExceptHandler) and current.name:
            names.add(current.name)
        elif isinstance(current, (ast.MatchAs, ast.MatchStar)) and current.name:
            names.add(current.name)
        elif isinstance(current, ast.MatchMapping) and current.rest:
            names.add(current.rest)
        stack.extend(ast.iter_child_nodes(current))
    return names


def _string_spans(source: bytes, path: str, names: Sequence[str]) -> dict[str, _Span]:
    """Locate each name's single top-level string constant by AST position."""
    try:
        tree = ast.parse(source, filename=path)
    except (SyntaxError, ValueError) as error:
        raise StringSymbolError(f"{path} is not valid Python: {error}") from None
    bindings: dict[str, list[ast.stmt]] = {}
    for statement in tree.body:
        for name in _bound_names(statement):
            bindings.setdefault(name, []).append(statement)
    starts = [0]
    for line in source.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))
    if source.startswith(codecs.BOM_UTF8):
        # The parser's first-line offsets exclude a UTF-8 byte-order mark.
        starts[0] = len(codecs.BOM_UTF8)
    rebound = {
        name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Global, ast.Nonlocal))
        for name in node.names
    }
    spans: dict[str, _Span] = {}
    for name in names:
        statements = bindings.get(name, [])
        if not statements:
            raise StringSymbolError(f"{path}:{name} is not assigned at module level.")
        if len(statements) > 1 or name in rebound:
            raise StringSymbolError(
                f"{path}:{name} is bound more than once at module level; "
                f"{_SELECTION_DOC}."
            )
        statement = statements[0]
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target: ast.expr | None = statement.targets[0]
        elif isinstance(statement, ast.AnnAssign):
            target = statement.target
        elif isinstance(statement, (ast.If, ast.Try, ast.With, ast.For, ast.While)):
            raise StringSymbolError(
                f"{path}:{name} is assigned inside a block, not at top-level; "
                f"{_SELECTION_DOC}."
            )
        else:
            target = None
        value = getattr(statement, "value", None)
        if not (
            isinstance(target, ast.Name)
            and target.id == name
            and isinstance(value, ast.Constant)
            and type(value.value) is str
            and value.end_lineno is not None
            and value.end_col_offset is not None
        ):
            raise StringSymbolError(
                f"{path}:{name} is not a plain string constant; {_SELECTION_DOC}."
            )
        start = starts[value.lineno - 1] + value.col_offset
        end = starts[value.end_lineno - 1] + value.end_col_offset
        try:
            # Parenthesize: an implicit concatenation may span lines.
            located = ast.literal_eval(f"({source[start:end].decode()})")
        except (SyntaxError, ValueError, UnicodeDecodeError):
            located = None
        if located != value.value:
            raise StringSymbolError(
                f"{path}:{name}: could not locate the constant's source span "
                "(only UTF-8 sources are supported)."
            )
        if _has_comment(source[start:end].decode()):
            # A rewrite replaces the whole span and would drop the comment.
            raise StringSymbolError(
                f"{path}:{name} has a comment between its concatenated parts; "
                "move the comment outside the constant."
            )
        spans[name] = _Span(start=start, end=end, value=value.value, kind=value.kind)
    return spans


def _has_comment(fragment: str) -> bool:
    readline = io.StringIO(f"({fragment})").readline
    try:
        return any(
            token.type == tokenize.COMMENT
            for token in tokenize.generate_tokens(readline)
        )
    except (tokenize.TokenError, SyntaxError):
        return True


def _outside(source: bytes, spans: Mapping[str, _Span]) -> list[bytes]:
    """Return the source bytes between the declared constants, in order."""
    pieces: list[bytes] = []
    cursor = 0
    for span in sorted(spans.values(), key=lambda item: item.start):
        pieces.append(source[cursor : span.start])
        cursor = span.end
    pieces.append(source[cursor:])
    return pieces


def _reframe(base: str, text: str) -> str:
    """Keep the base value's surrounding whitespace around a stripped text."""
    stripped = base.strip()
    if not stripped:
        return base + text
    leading = base[: len(base) - len(base.lstrip())]
    trailing = base[len(base.rstrip()) :]
    return leading + text + trailing


def _literal(text: str) -> str:
    """Format ``text`` as a Python literal, triple-quoted when multi-line."""
    if "\n" in text and all(char in "\n\t" or char.isprintable() for char in text):
        body = text.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
        if body.endswith('"'):
            body = body[:-1] + '\\"'
        literal = f'"""{body}"""'
        try:
            if ast.literal_eval(literal) == text:
                return literal
        except (SyntaxError, ValueError):
            pass
    return repr(text)


class _Git:
    """Plumbing-only Git access that never touches the checkout or its index."""

    def __init__(self, project: Path) -> None:
        # Plumbing pathspecs are cwd-relative: run from the repository root
        # and prefix declared paths with the project's location inside it.
        self.root = project.resolve()
        self.env = {
            key: value
            for key, value in os.environ.items()
            if key not in _REPOSITORY_ENV
            and not key.startswith(("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_"))
        }
        prefix = self.run("rev-parse", "--show-prefix").decode().strip()
        self.prefix = prefix.rstrip("/")
        self.root = Path(self.run("rev-parse", "--show-toplevel").decode().strip())

    def run(
        self,
        *args: str,
        input: bytes | None = None,
        env: Mapping[str, str] | None = None,
        check: bool = True,
    ) -> bytes:
        result = subprocess.run(
            ["git", "-C", str(self.root), *args],
            input=input,
            env=dict(env or self.env),
            capture_output=True,
        )
        if check and result.returncode:
            detail = result.stderr.decode(errors="replace").strip()
            raise StringSymbolError(f"git {args[0]} failed: {detail}")
        return result.stdout

    def path(self, relative: str) -> str:
        return f"{self.prefix}/{relative}" if self.prefix else relative

    def relative(self, path: str) -> str | None:
        if not self.prefix:
            return path
        head = self.prefix + "/"
        return path[len(head) :] if path.startswith(head) else None

    def commit(self, rev: str) -> str:
        if rev.startswith("-"):
            raise StringSymbolError(f"{rev!r} is not a commit.")
        return self.run("rev-parse", "--verify", f"{rev}^{{commit}}").decode().strip()

    def blob(self, commit: str, relative: str) -> tuple[str, bytes]:
        path = self.path(relative)
        entry = self.run("ls-tree", "-z", commit, "--", path).rstrip(b"\0")
        meta, _, name = entry.partition(b"\t")
        fields = meta.split()
        if (
            name.decode(errors="surrogateescape") != path
            or len(fields) != 3
            or fields[1] != b"blob"
        ):
            raise StringSymbolError(f"{relative} is not a file at {commit[:12]}.")
        mode, _, oid = (field.decode() for field in fields)
        if mode not in ("100644", "100755"):
            raise StringSymbolError(f"{relative} is not a regular file.")
        return mode, self.run("cat-file", "blob", oid)

    def is_ancestor(self, base: str, commit: str) -> bool:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(self.root),
                "merge-base",
                "--is-ancestor",
                base,
                commit,
            ],
            env=self.env,
            capture_output=True,
        )
        if result.returncode not in (0, 1):
            raise StringSymbolError("git merge-base failed.")
        return result.returncode == 0

    def changes(self, base: str, commit: str) -> list[tuple[str, str, str, str]]:
        raw = self.run("diff-tree", "-r", "-z", "--no-renames", base, commit)
        fields = raw.split(b"\0")
        entries: list[tuple[str, str, str, str]] = []
        for index in range(0, len(fields) - 1, 2):
            meta = fields[index].decode().lstrip(":").split()
            path = fields[index + 1].decode(errors="surrogateescape")
            entries.append((meta[4], meta[0], meta[1], path))
        return entries

    def commit_tree(
        self, base: str, files: Mapping[str, tuple[str, bytes]], message: str
    ) -> str:
        with tempfile.TemporaryDirectory(prefix="gepa-string-symbols-") as private:
            env = {**self.env, "GIT_INDEX_FILE": str(Path(private) / "index")}
            if any(
                self.run("var", ident, env=env, check=False) == b""
                for ident in ("GIT_AUTHOR_IDENT", "GIT_COMMITTER_IDENT")
            ):
                name, email = _FALLBACK_IDENTITY
                for role in ("AUTHOR", "COMMITTER"):
                    env.setdefault(f"GIT_{role}_NAME", name)
                    env.setdefault(f"GIT_{role}_EMAIL", email)
            self.run("read-tree", base, env=env)
            for relative, (mode, content) in files.items():
                oid = (
                    self.run(
                        "hash-object", "-w", "--no-filters", "--stdin", input=content
                    )
                    .decode()
                    .strip()
                )
                self.run(
                    "update-index",
                    "--cacheinfo",
                    f"{mode},{oid},{self.path(relative)}",
                    env=env,
                )
            tree = self.run("write-tree", env=env).decode().strip()
            # Candidate commits are machine-made; never prompt for a signature.
            return (
                self.run(
                    "commit-tree",
                    "--no-gpg-sign",
                    tree,
                    "-p",
                    base,
                    "-F",
                    "-",
                    input=message.encode(),
                    env=env,
                )
                .decode()
                .strip()
            )


__all__ = [
    "StringSymbolError",
    "candidate_from_commit",
    "commit_from_candidate",
    "masked_string_symbols",
    "normalize_string_symbols",
    "resolve_commit",
    "string_symbol_component",
    "verify_string_symbol_commit",
]
