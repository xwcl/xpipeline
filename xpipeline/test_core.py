def test_version_autogen():
    from . import version
    assert version.version
    assert version.version_tuple
