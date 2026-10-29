# Owner(s): ["module: dynamo"]
# The implicit __class__ cell must survive torch.compile. These are the
# CPython 3.13 shapes from test_super.py that used to come back None or as
# the wrong class.

import torch
from torch._dynamo.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import HardwareClassification


class _EnclosingHost:
    def read_enclosing(self):
        class X:
            value = __class__

            def f():
                __class__

        return X.value


class _PathologyHost:
    def pathologies(self):
        class A:
            def f(self):
                return "A"

        class X(A):
            def f(self):
                return super().f()

            __class__ = 413

        x = X()
        shadowed = x.__class__
        via_super = x.f()

        class Y:
            value = __class__

            def f():
                __class__

        enclosing = Y.value
        try:
            exec(
                "class X:\n    __class__\n    def f():\n        __class__\n",
                globals(),
                {},
            )
        except NameError:
            saw_name_error = True
        else:
            saw_name_error = False

        class Z:
            global __class__
            __class__ = 42

            def f():
                __class__

        stored = globals()["__class__"]
        del globals()["__class__"]
        leaked = "__class__" in Z.__dict__

        class W:
            nonlocal __class__
            __class__ = 42

            def f():
                __class__

        return shadowed, via_super, enclosing, saw_name_error, stored, leaked, __class__


class ClassCellTests(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def _compile(self, fn):
        return torch.compile(fn, backend="eager", fullgraph=True)

    @torch._dynamo.config.patch(enable_trace_load_build_class=True)
    def test_plain___class___is_the_class(self):
        def fn(t):
            class A:
                def f(self):
                    return __class__

            return t + 1, A().f(), A().__class__

        t = torch.ones(2)
        out, cls, inst_cls = self._compile(fn)(t)
        self.assertEqual(out, t + 1)
        self.assertIs(cls, inst_cls)

    @torch._dynamo.config.patch(enable_trace_load_build_class=True)
    def test___class___set_by_metaclass_new(self):
        def fn(t):
            seen = None

            class Meta(type):
                def __new__(cls, name, bases, namespace):
                    nonlocal seen
                    self = super().__new__(cls, name, bases, namespace)
                    seen = self.f()
                    return self

            class A(metaclass=Meta):
                @staticmethod
                def f():
                    return __class__

            return t + 1, seen is A

        t = torch.ones(2)
        out, ok = self._compile(fn)(t)
        self.assertEqual(out, t + 1)
        self.assertTrue(ok)

    @torch._dynamo.config.patch(enable_trace_load_build_class=True)
    def test___class___filled_when_type_called_later(self):
        def fn(t):
            namespace = None

            class Meta(type):
                def __new__(cls, name, bases, ns):
                    nonlocal namespace
                    namespace = ns
                    return None

            class A(metaclass=Meta):
                @staticmethod
                def f():
                    return __class__

            B = type("B", (), namespace)
            return t + 1, A is None, B.f() is B

        t = torch.ones(2)
        out, a_is_none, b_matches = self._compile(fn)(t)
        self.assertEqual(out, t + 1)
        self.assertTrue(a_is_none)
        self.assertTrue(b_matches)

    @torch._dynamo.config.patch(enable_trace_load_build_class=True)
    def test___class___visible_during_mro(self):
        def fn(t):
            seen = None

            class Meta(type):
                def mro(self):
                    self.__dict__["f"]()
                    return super().mro()

            class A(metaclass=Meta):
                def f():
                    nonlocal seen
                    seen = __class__

            return t + 1, seen is A

        t = torch.ones(2)
        out, ok = self._compile(fn)(t)
        self.assertEqual(out, t + 1)
        self.assertTrue(ok)

    @torch._dynamo.config.patch(enable_trace_load_build_class=True)
    def test_enclosing___class___in_class_body(self):
        compiled = self._compile(_EnclosingHost.read_enclosing)
        self.assertIs(compiled(_EnclosingHost()), _EnclosingHost)

    @torch._dynamo.config.patch(enable_trace_load_build_class=True)
    def test___class___pathologies(self):
        compiled = self._compile(_PathologyHost.pathologies)
        shadowed, via_super, enclosing, saw_name_error, stored, leaked, rebound = (
            compiled(_PathologyHost())
        )
        self.assertEqual(shadowed, 413)
        self.assertEqual(via_super, "A")
        self.assertIs(enclosing, _PathologyHost)
        self.assertTrue(saw_name_error)
        self.assertEqual(stored, 42)
        self.assertFalse(leaked)
        self.assertEqual(rebound, 42)

    @torch._dynamo.config.patch(enable_trace_load_build_class=True)
    def test_successful_exec_still_graph_breaks(self):
        def fn(t):
            ns = {}
            exec("x = 1", ns)
            return t + ns["x"]

        t = torch.ones(2)
        eager = fn(t)
        compiled = torch.compile(fn, backend="eager", fullgraph=False)
        self.assertEqual(compiled(t), eager)
        torch._dynamo.reset()

        def fn_full(t):
            ns = {}
            exec("x = 1", ns)
            return t + ns["x"]

        with self.assertRaises(torch._dynamo.exc.Unsupported):
            torch.compile(fn_full, backend="eager", fullgraph=True)(t)


if __name__ == "__main__":
    run_tests()
