from pr_checker.identifier_extractor import (
    _extract_ast,
    extract_identifiers,
    extract_identifiers_with_ast,
)
from pr_checker.models import DiffHunk, DiffLine


def _hunk(file_path: str, lines: list[DiffLine]) -> DiffHunk:
    return DiffHunk(
        file_path=file_path,
        header="@@ -1,10 +1,10 @@",
        old_start=1,
        old_count=10,
        new_start=1,
        new_count=10,
        lines=lines,
    )


def _added(content: str, lineno: int = 1) -> DiffLine:
    return DiffLine(kind="+", content=content, old_lineno=None, new_lineno=lineno)


def _removed(content: str, lineno: int = 1) -> DiffLine:
    return DiffLine(kind="-", content=content, old_lineno=lineno, new_lineno=None)


def _context(content: str) -> DiffLine:
    return DiffLine(kind=" ", content=content, old_lineno=1, new_lineno=1)


class TestRegexExtraction:
    def test_python_function_def(self) -> None:
        hunks = [_hunk("foo.py", [_added("def process_webhook(payload):")])]
        result = extract_identifiers(hunks)
        names = {i.name for i in result}
        assert "process_webhook" in names

    def test_python_class_def(self) -> None:
        hunks = [_hunk("models.py", [_added("class GitHubClient:")])]
        result = extract_identifiers(hunks)
        names = {i.name for i in result}
        assert "GitHubClient" in names
        gh = next(i for i in result if i.name == "GitHubClient")
        assert gh.kind == "class_ref"

    def test_function_call(self) -> None:
        hunks = [_hunk("main.py", [_added("    result = fetch_data(url)")])]
        result = extract_identifiers(hunks)
        names = {i.name for i in result}
        assert "fetch_data" in names

    def test_method_call(self) -> None:
        hunks = [_hunk("main.py", [_added("    self.validate_payload(data)")])]
        result = extract_identifiers(hunks)
        names = {i.name for i in result}
        assert "validate_payload" in names

    def test_python_import(self) -> None:
        hunks = [_hunk("main.py", [_added("from pr_checker.github_client import GitHubClient")])]
        result = extract_identifiers(hunks)
        names = {i.name for i in result}
        assert "pr_checker.github_client" in names

    def test_cpp_include(self) -> None:
        hunks = [_hunk("main.cpp", [_added('#include "cuda_runtime.h"')])]
        result = extract_identifiers(hunks)
        names = {i.name for i in result}
        assert "cuda_runtime.h" in names

    def test_cpp_function_def(self) -> None:
        hunks = [_hunk("kernel.cu", [_added("void launch_kernel(int n) {")])]
        result = extract_identifiers(hunks)
        names = {i.name for i in result}
        assert "launch_kernel" in names

    def test_scope_resolution(self) -> None:
        hunks = [_hunk("main.cpp", [_added("    MyClass::do_thing();")])]
        result = extract_identifiers(hunks)
        names = {i.name for i in result}
        assert "MyClass" in names

    def test_decorator(self) -> None:
        hunks = [_hunk("views.py", [_added("@router.post")])]
        result = extract_identifiers(hunks)
        names = {i.name for i in result}
        assert "router" in names

    def test_isinstance_extracts_class(self) -> None:
        hunks = [_hunk("check.py", [_added("    if isinstance(obj, ReviewContext):")])]
        result = extract_identifiers(hunks)
        names = {i.name for i in result}
        assert "ReviewContext" in names

    def test_type_annotation(self) -> None:
        hunks = [_hunk("foo.py", [_added("def run(ctx: ReviewContext) -> Summary:")])]
        result = extract_identifiers(hunks)
        names = {i.name for i in result}
        assert "ReviewContext" in names
        assert "Summary" in names

    def test_removed_lines_extracted(self) -> None:
        hunks = [_hunk("old.py", [_removed("    old_function(x)")])]
        result = extract_identifiers(hunks)
        names = {i.name for i in result}
        assert "old_function" in names

    def test_context_lines_ignored(self) -> None:
        hunks = [_hunk("ctx.py", [_context("    only_in_context()")])]
        result = extract_identifiers(hunks)
        names = {i.name for i in result}
        assert "only_in_context" not in names


class TestFiltering:
    def test_keywords_filtered(self) -> None:
        hunks = [
            _hunk(
                "test.py",
                [
                    _added("if True:"),
                    _added("    return None"),
                    _added("for x in range(10):"),
                ],
            )
        ]
        result = extract_identifiers(hunks)
        names = {i.name for i in result}
        assert "if" not in names
        assert "True" not in names
        assert "return" not in names
        assert "for" not in names
        assert "None" not in names

    def test_single_char_filtered(self) -> None:
        hunks = [_hunk("t.py", [_added("x = f(a, b)")])]
        result = extract_identifiers(hunks)
        names = {i.name for i in result}
        assert "x" not in names
        assert "a" not in names
        assert "b" not in names

    def test_python_builtins_filtered(self) -> None:
        hunks = [_hunk("t.py", [_added("    print(len(items))")])]
        result = extract_identifiers(hunks)
        names = {i.name for i in result}
        assert "print" not in names
        assert "len" not in names

    def test_cpp_builtins_filtered(self) -> None:
        hunks = [_hunk("t.cpp", [_added("    std::cout << sizeof(x);")])]
        result = extract_identifiers(hunks)
        names = {i.name for i in result}
        assert "std" not in names
        assert "cout" not in names

    def test_dunder_methods_filtered(self) -> None:
        hunks = [_hunk("t.py", [_added("    self.__init__()")])]
        result = extract_identifiers(hunks)
        names = {i.name for i in result}
        assert "__init__" not in names


class TestDeduplication:
    def test_same_name_deduped(self) -> None:
        hunks = [
            _hunk(
                "a.py",
                [
                    _added("    process(x)"),
                    _added("    process(y)"),
                    _added("    process(z)"),
                ],
            )
        ]
        result = extract_identifiers(hunks)
        process_entries = [i for i in result if i.name == "process"]
        assert len(process_entries) == 1

    def test_definition_wins_over_call(self) -> None:
        hunks = [
            _hunk(
                "a.py",
                [
                    _added("    process(x)"),
                    _added("def process(data):"),
                ],
            )
        ]
        result = extract_identifiers(hunks)
        process = next(i for i in result if i.name == "process")
        assert process.kind == "function_def"


class TestRanking:
    def test_definitions_before_calls_before_imports(self) -> None:
        hunks = [
            _hunk(
                "a.py",
                [
                    _added("from utils import helper"),
                    _added("class Manager:"),
                    _added("    helper()"),
                ],
            )
        ]
        result = extract_identifiers(hunks)
        names = [i.name for i in result]
        manager_idx = names.index("Manager")
        helper_idx = names.index("helper")
        utils_idx = names.index("utils")
        assert manager_idx < helper_idx
        assert helper_idx < utils_idx


class TestAstRefinement:
    def test_ast_catches_nested_call(self) -> None:
        source = "result = outer(inner(x))\n"
        hunks = [_hunk("nested.py", [_added("result = outer(inner(x))", lineno=1)])]
        result = extract_identifiers_with_ast(hunks, {"nested.py": source})
        names = {i.name for i in result}
        assert "outer" in names
        assert "inner" in names

    def test_ast_resolves_import(self) -> None:
        source = (
            "from pr_checker.github_client import GitHubClient\n\nclient = GitHubClient(token)\n"
        )
        hunks = [_hunk("use.py", [_added("client = GitHubClient(token)", lineno=3)])]
        result = extract_identifiers_with_ast(hunks, {"use.py": source})
        names = {i.name for i in result}
        assert "GitHubClient" in names
        assert "pr_checker.github_client.GitHubClient" in names

    def test_ast_fallback_on_syntax_error(self) -> None:
        bad_source = "def broken(\n"
        hunks = [_hunk("bad.py", [_added("broken()", lineno=1)])]
        result = extract_identifiers_with_ast(hunks, {"bad.py": bad_source})
        names = {i.name for i in result}
        assert "broken" in names

    def test_ast_skipped_for_non_python(self) -> None:
        hunks = [_hunk("main.cpp", [_added("void run() {}", lineno=1)])]
        result = extract_identifiers_with_ast(hunks, {"main.cpp": "void run() {}"})
        names = {i.name for i in result}
        assert "run" in names

    def test_ast_call_in_comprehension(self) -> None:
        source = "items = [transform(x) for x in data]\n"
        hunks = [_hunk("comp.py", [_added("items = [transform(x) for x in data]", lineno=1)])]
        result = extract_identifiers_with_ast(hunks, {"comp.py": source})
        names = {i.name for i in result}
        assert "transform" in names


class TestMixedLanguageDiff:
    def test_python_and_cpp_hunks(self) -> None:
        hunks = [
            _hunk("handler.py", [_added("def on_event(evt):")]),
            _hunk(
                "kernel.cu",
                [
                    _added('#include "kernel.h"'),
                    _added("void compute_result(float* data) {"),
                ],
            ),
        ]
        result = extract_identifiers(hunks)
        names = {i.name for i in result}
        assert "on_event" in names
        assert "kernel.h" in names
        assert "compute_result" in names


class TestImportRegex:
    def test_no_trailing_comma(self) -> None:
        hunks = [_hunk("t.py", [_added("import os, sys")])]
        result = extract_identifiers(hunks)
        names = {i.name for i in result}
        assert "os" in names
        assert "os," not in names

    def test_dotted_import(self) -> None:
        hunks = [_hunk("t.py", [_added("import os.path")])]
        result = extract_identifiers(hunks)
        names = {i.name for i in result}
        assert "os.path" in names


class TestIsinstanceRegex:
    def test_dotted_name(self) -> None:
        hunks = [_hunk("t.py", [_added("isinstance(x, ast.FunctionDef)")])]
        result = extract_identifiers(hunks)
        names = {i.name for i in result}
        assert "ast.FunctionDef" in names

    def test_tuple_of_types(self) -> None:
        hunks = [_hunk("t.py", [_added("isinstance(x, (Foo, Bar))")])]
        result = extract_identifiers(hunks)
        names = {i.name for i in result}
        assert "Foo" in names
        assert "Bar" in names


class TestLanguageGating:
    def test_cpp_func_def_not_matched_in_python(self) -> None:
        hunks = [_hunk("utils.py", [_added("result = float(data)")])]
        result = extract_identifiers(hunks)
        kinds = {i.kind for i in result if i.name == "data"}
        assert "function_def" not in kinds

    def test_python_def_not_matched_in_cpp(self) -> None:
        hunks = [_hunk("main.cpp", [_added("def something();")])]
        result = extract_identifiers(hunks)
        kinds = {i.kind for i in result if i.name == "something"}
        assert "function_def" not in kinds

    def test_scope_resolution_not_matched_in_python(self) -> None:
        hunks = [_hunk("t.py", [_added("x = Foo::bar()")])]
        result = extract_identifiers(hunks)
        names = {i.name for i in result}
        assert "Foo" not in names


class TestAstOnlyNewLines:
    def test_deleted_line_not_matched_in_ast(self) -> None:
        source = "old_call()\nnew_call()\n"
        ast_results = _extract_ast("f.py", source, changed_lines={2})
        ast_names = {i.name for i in ast_results}
        assert "new_call" in ast_names
        assert "old_call" not in ast_names
