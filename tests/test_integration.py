"""Integration tests for Voicesynth."""
from src.core import Voicesynth

class TestVoicesynth:
    def setup_method(self):
        self.c = Voicesynth()
    def test_10_ops(self):
        for i in range(10): self.c.synthesize(i=i)
        assert self.c.get_stats()["ops"] == 10
    def test_service_name(self):
        assert self.c.synthesize()["service"] == "voicesynth"
    def test_different_inputs(self):
        self.c.synthesize(type="a"); self.c.synthesize(type="b")
        assert self.c.get_stats()["ops"] == 2
    def test_config(self):
        c = Voicesynth(config={"debug": True})
        assert c.config["debug"] is True
    def test_empty_call(self):
        assert self.c.synthesize()["ok"] is True
    def test_large_batch(self):
        for _ in range(100): self.c.synthesize()
        assert self.c.get_stats()["ops"] == 100
