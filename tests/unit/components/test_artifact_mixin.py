"""Tests ArtifactMixin's source hash: what counts as a code change of a component."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import hashlib
import importlib.util
import inspect
import sys
import textwrap

import pytest

from pfund.components._mixins.artifact_mixin import ArtifactMixin, _normalize_source

BASE = '''
@dec(1)
class A(Base):
    """doc"""

    x = 1

    def f(self, a):
        """f doc"""
        return a + 1

    async def g(self):
        """only a docstring"""
'''


class TestNormalizeSource:
    @pytest.mark.parametrize(
        "variant",
        [
            pytest.param(
                '''
@dec(1)
class A(Base):  # trailing comment
    """doc"""
    # a comment
    x = 1
    def f(self, a):
        # a comment before the docstring, which is still the first statement
        """f doc"""
        return a + 1  # inline
    async def g(self):
        """only a docstring"""
''',
                id="comments",
            ),
            pytest.param(
                '''
@dec( 1 )
class A( Base ):
    """doc"""
    x  =  1
    def f(
        self,
        a,
    ):
        """f doc"""
        return (a
                + 1)



    async def g(self):
        """only a docstring"""
''',
                id="formatting",
            ),
            pytest.param(
                '''
@dec(1)
class A(Base):
    """a new class docstring,
    on two lines"""

    x = 1

    def f(self, a):
        'a single-quoted docstring'
        return a + 1

    async def g(self):
        """changed"""
''',
                id="docstrings",
            ),
            pytest.param(textwrap.indent(BASE, "    "), id="indented"),
        ],
    )
    def test_non_behavioral_change_is_ignored(self, variant: str):
        assert _normalize_source(variant) == _normalize_source(BASE)

    @pytest.mark.parametrize(
        "old, new",
        [
            pytest.param("a + 1", "a + 2", id="logic"),
            pytest.param("@dec(1)", "@dec(2)", id="decorator"),
            pytest.param("x = 1", "x = 'hello'", id="non-docstring string literal"),
            pytest.param("def f(self, a)", "def f(self, b)", id="param name"),
            pytest.param("class A(Base)", "class A(Other)", id="base class"),
        ],
    )
    def test_behavioral_change_is_detected(self, old: str, new: str):
        assert old in BASE
        assert _normalize_source(BASE.replace(old, new)) != _normalize_source(BASE)

    def test_string_moved_to_first_statement_is_a_change(self):
        # a string literal that isn't the first statement isn't a docstring, it's code
        changed = BASE.replace("    x = 1\n", "    x = 1\n    'not a docstring'\n")
        assert _normalize_source(changed) != _normalize_source(BASE)

    def test_dump_has_no_line_numbers(self):
        assert "lineno" not in _normalize_source(BASE)

    def test_unparsable_source_falls_back_to_raw(self):
        # a multi-line string at column 0 in a nested class: dedent() can't strip the indentation
        def make() -> type:
            class Nested:
                s = """
at column 0"""

            return Nested

        source = inspect.getsource(make())
        assert _normalize_source(source) == source


# the hash of no source at all, i.e. every class was skipped
EMPTY_SHA256 = hashlib.sha256().hexdigest()


def _load_class(
    path: Path,
    module_name: str,
    source: str,
    class_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> type:
    path.write_text(textwrap.dedent(source))
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # inspect finds a class's file through sys.modules, as it does for an imported user module
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


USER_MODULE = '''
from pfund.components._mixins.artifact_mixin import ArtifactMixin


class MyBase(ArtifactMixin):
    def helper(self):
        return 1


class Alpha(MyBase):
    """Alpha"""

    def signal(self):
        return self.helper() + 1
'''


class TestSourceSha256:
    @pytest.fixture
    def load(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        count = 0

        def _load(source: str) -> ArtifactMixin:
            nonlocal count
            count += 1
            # a new file and module per version, so inspect reads each version's own source
            klass = _load_class(
                tmp_path / f"user_{count}.py",
                f"user_{count}",
                source,
                "Alpha",
                monkeypatch,
            )
            component = klass()
            assert component._source_sha256 != EMPTY_SHA256, "no source was hashed"
            return component

        return _load

    def test_same_source_same_hash(self, load):
        assert load(USER_MODULE)._source_sha256 == load(USER_MODULE)._source_sha256

    def test_comments_and_docstrings_keep_the_hash(self, load):
        edited = USER_MODULE.replace(
            '    """Alpha"""', '    """Alpha, reworded"""\n    # a note'
        ).replace("return 1", "return 1  # why")
        assert load(edited)._source_sha256 == load(USER_MODULE)._source_sha256

    def test_code_outside_the_classes_keeps_the_hash(self, load):
        edited = USER_MODULE + "\nprint('running a script')\n"
        assert load(edited)._source_sha256 == load(USER_MODULE)._source_sha256

    def test_logic_change_changes_the_hash(self, load):
        edited = USER_MODULE.replace("self.helper() + 1", "self.helper() + 2")
        assert load(edited)._source_sha256 != load(USER_MODULE)._source_sha256

    def test_user_base_class_change_changes_the_hash(self, load):
        edited = USER_MODULE.replace("return 1", "return 2")
        assert load(edited)._source_sha256 != load(USER_MODULE)._source_sha256
