from kadot.generators import BaseGenerator

def test_fit():
    generator = BaseGenerator()
    assert generator.fit(['hello world !']).documents == ['START', 'hello ', 'world !', 'END']