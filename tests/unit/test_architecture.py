"""The dependency rule: code may only import from its own layer or from layers further in.

    presentation -> infrastructure -> application -> domain
"""
import ast
from pathlib import Path

APP = Path(__file__).resolve().parents[2] / "app"
ALLOWED = {
    "domain": {"domain"},
    "application": {"domain", "application"},
    "infrastructure": {"domain", "application", "infrastructure"},
    "presentation": {"domain", "application", "infrastructure", "presentation"},
}


def imported_layers(path: Path) -> set[str]:
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        names = [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
        names += [a.name for a in node.names] if isinstance(node, ast.Import) else []
        for n in names:
            parts = n.split(".")
            if parts[0] == "app" and len(parts) > 1 and parts[1] in ALLOWED:
                found.add(parts[1])
    return found


def test_layers_only_depend_inward():
    problems = []
    for layer, allowed in ALLOWED.items():
        for path in (APP / layer).rglob("*.py"):
            bad = imported_layers(path) - allowed
            if bad:
                problems.append(f"{path.relative_to(APP)} imports from {sorted(bad)}")
    assert not problems, "\n".join(problems)


def test_domain_and_application_do_not_import_frameworks():
    banned = {"fastapi", "starlette", "pydantic", "sqlite3", "torch", "transformers", "urllib", "langdetect"}
    problems = []
    for layer in ("domain", "application"):
        for path in (APP / layer).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                mods = [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
                mods += [a.name for a in node.names] if isinstance(node, ast.Import) else []
                hit = {m.split(".")[0] for m in mods} & banned
                if hit:
                    problems.append(f"{path.relative_to(APP)} imports {sorted(hit)}")
    assert not problems, "\n".join(problems)
