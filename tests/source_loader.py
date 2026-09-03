"""Execute unmodified real methods without importing GPU-only dependencies.

These are control-flow tests, not full PyTorch integration tests. Only the module
imports and unrelated methods are omitted; selected method bodies are unchanged.
"""
import ast
from pathlib import Path


def load_class_methods(path, class_name, methods, namespace):
    path = Path(path)
    tree = ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
    node = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == class_name)
    node.body = [item for item in node.body if isinstance(item, ast.FunctionDef) and item.name in methods]
    scope = dict(namespace)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    return scope[class_name]


def load_function(path, name, namespace):
    path = Path(path)
    tree = ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
    node = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == name)
    scope = dict(namespace)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    return scope[name]
