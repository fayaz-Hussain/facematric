import os
import ast
import re

backend_dir = os.path.abspath("backend")

def scan_tests():
    no_assert_tests = []
    skipped_tests = []
    trivial_assert_tests = []
    clock_dependent_tests = []
    path_dependent_tests = []
    
    for root, _, files in os.walk(os.path.join(backend_dir, "tests")):
        for f in files:
            if not f.endswith(".py") or not f.startswith("test_"):
                continue
            fpath = os.path.join(root, f)
            rel_path = os.path.relpath(fpath, backend_dir)
            
            with open(fpath, "r", encoding="utf-8") as fp:
                content = fp.read()
                
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                    # Check decorators
                    for dec in node.decorator_list:
                        dec_str = ast.unparse(dec)
                        if "skip" in dec_str or "xfail" in dec_str:
                            skipped_tests.append((rel_path, node.name, dec_str))
                            
                    # Check assertions
                    asserts = [n for n in ast.walk(node) if isinstance(n, ast.Assert)]
                    if len(asserts) == 0:
                        # Check if raises is used
                        has_raises = False
                        for with_node in ast.walk(node):
                            if isinstance(with_node, ast.With):
                                for item in with_node.items:
                                    if "raises" in ast.unparse(item.context_expr):
                                        has_raises = True
                        if not has_raises:
                            no_assert_tests.append((rel_path, node.name))
                    else:
                        # Check for trivial assertions
                        for a in asserts:
                            code = ast.unparse(a)
                            if code in ["assert True", "assert 1", "assert True == True"]:
                                trivial_assert_tests.append((rel_path, node.name, code))
                            elif "in [200, 400]" in code or "in [200, 400, 408]" in code:
                                trivial_assert_tests.append((rel_path, node.name, code))
                                
                    # Check clock dependency
                    body_code = ast.unparse(node)
                    if any(c in body_code for c in ["time.time()", "time.sleep", "datetime.now", "datetime.utcnow", "asyncio.sleep"]):
                        clock_dependent_tests.append((rel_path, node.name))
                        
                    # Check file path dependency
                    if any(p in body_code for p in ["os.path.exists", "Path(", "open(", "settings.mediapipe_model_path", "celebrity_embeddings.pkl"]):
                        path_dependent_tests.append((rel_path, node.name))
                        
    return no_assert_tests, skipped_tests, trivial_assert_tests, clock_dependent_tests, path_dependent_tests

if __name__ == "__main__":
    no_assert, skipped, trivial, clock, path = scan_tests()
    print(f"Skipped / xfail tests: {len(skipped)}")
    for s in skipped:
        print(f"  {s[0]}::{s[1]} -> {s[2]}")
    print(f"\nTests with zero assertions: {len(no_assert)}")
    for na in no_assert:
        print(f"  {na[0]}::{na[1]}")
    print(f"\nTrivial assertions (cannot fail): {len(trivial)}")
    for t in trivial:
        print(f"  {t[0]}::{t[1]} -> {t[2]}")
    print(f"\nClock-dependent tests: {len(clock)}")
    for c in clock:
        print(f"  {c[0]}::{c[1]}")
    print(f"\nPath-dependent tests: {len(path)}")
    for p in path:
        print(f"  {p[0]}::{p[1]}")
