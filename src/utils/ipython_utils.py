##^# ipython or ignore if not available ########################################
class DummyIPython:
    def magic(self, *args):
        return None


try:
    from IPython import get_ipython

    ip = get_ipython()
    assert ip is not None
except Exception:
    ip = DummyIPython()


def run_ipython(*args):
    for arg in args:
        ip.magic(arg)


##$#############################################################################
