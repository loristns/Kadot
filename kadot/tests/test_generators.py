from kadot.generators import BaseGenerator

def test_fit():
    generator = BaseGenerator()
    generator.fit(['hello world !'])

    assert generator.documents == ['START', 'hello ', 'world !', 'END']