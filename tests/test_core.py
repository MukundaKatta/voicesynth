from src.core import VoiceSynth
def test_init(): assert VoiceSynth().get_stats()["ops"] == 0
def test_op(): c = VoiceSynth(); c.synthesize(x=1); assert c.get_stats()["ops"] == 1
def test_multi(): c = VoiceSynth(); [c.synthesize() for _ in range(5)]; assert c.get_stats()["ops"] == 5
def test_reset(): c = VoiceSynth(); c.synthesize(); c.reset(); assert c.get_stats()["ops"] == 0
