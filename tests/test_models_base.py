"""PointProcessModel protocol conformance tests."""

from eonet_cascades.models.base import PointProcessModel


def test_protocol_has_required_methods():
    methods = set(dir(PointProcessModel))
    for required in {"log_likelihood", "sample", "fit"}:
        assert required in methods, f"Protocol missing method: {required}"


def test_protocol_runtime_checkable():
    # @runtime_checkable lets isinstance(obj, PointProcessModel) work.
    class _Stub:
        name = "stub"

        def log_likelihood(self, events, window): ...
        def sample(self, history, window): ...
        def fit(self, events, window, **kwargs): ...

    assert isinstance(_Stub(), PointProcessModel)
