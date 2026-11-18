"""
Tests for the low-level _tsdate C module.
"""

import _tsdate
import pytest

ERR_GENERIC = -1
ERR_NO_MEMORY = -2
ERR_BAD_PARAM = -3
ALL_ERRORS = [ERR_GENERIC, ERR_NO_MEMORY, ERR_BAD_PARAM]


class TestModule:
    def test_name(self):
        assert _tsdate.__name__ == "_tsdate"

    def test_library_error_is_exception(self):
        assert issubclass(_tsdate.LibraryError, Exception)
        assert _tsdate.LibraryError.__name__ == "LibraryError"


class TestStrerror:
    def test_success_code(self):
        assert "not an error" in _tsdate.strerror(0)

    @pytest.mark.parametrize("code", ALL_ERRORS)
    def test_known_codes(self, code):
        message = _tsdate.strerror(code)
        assert isinstance(message, str)
        assert len(message) > 0

    def test_messages_distinct(self):
        messages = {_tsdate.strerror(code) for code in [0, *ALL_ERRORS]}
        assert len(messages) == len(ALL_ERRORS) + 1

    def test_unknown_code(self):
        assert _tsdate.strerror(-12345) == "Unknown error"

    @pytest.mark.parametrize("bad_arg", ["x", None, 1.5])
    def test_bad_argument(self, bad_arg):
        with pytest.raises(TypeError):
            _tsdate.strerror(bad_arg)

    def test_no_arguments(self):
        with pytest.raises(TypeError):
            _tsdate.strerror()


class TestRaiseLibraryError:
    @pytest.mark.parametrize("code", [ERR_GENERIC, ERR_BAD_PARAM])
    def test_raises_library_error(self, code):
        with pytest.raises(_tsdate.LibraryError, match=_tsdate.strerror(code)):
            _tsdate.raise_library_error(code)

    def test_no_memory_is_memory_error(self):
        with pytest.raises(MemoryError):
            _tsdate.raise_library_error(ERR_NO_MEMORY)

    def test_unknown_code(self):
        with pytest.raises(_tsdate.LibraryError, match="Unknown error"):
            _tsdate.raise_library_error(-12345)

    def test_bad_argument(self):
        with pytest.raises(TypeError):
            _tsdate.raise_library_error("x")
