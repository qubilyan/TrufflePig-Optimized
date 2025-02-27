
import trufflepig.filters.stylemeasures as tpsm


def test_count_paragraphs():
    result = tpsm.count_paragraphs('Hello \n\n World \n\n\n !')
    assert result == 3


def test_detect_language():
    detector = tpsm.LanguageDetector()
    result = detector.detect_language('die katze ist klein der hund auch')
    assert result == 'de'


def test_language_probs():
    detector = tpsm.LanguageDetector()
    result = detector.get_probabilities('die katze ist klein and I love mike and peter')
    assert len(result) > 1


def test_split_into_sentences():
    result = tpsm.split_into_sentences('Hi my name is! Slim Shady! Really? Yeah.')
    assert result == ['Hi my name is!', 'Slim Shady!', 'Really?', 'Yeah.']


def test_compute_average_sentence_length():
    result = tpsm.compute_average_sentence_length(['huhuh.', 'sbbbasdsads'])
    assert result == 8.5


def test_compute_sentence_length_variance():
    result = tpsm.compute_sentence_length_variance(['huhuh.', 'sbbbasdsads', 'jj djdjd', '1'])
    assert result == 13.25


def test_compute_sentence_length_skew():
    result = tpsm.compute_sentence_length_skew(['huhuh.', 'sbbbasdsads', 'jj djdjd', '1'])
    assert result == -0.37320606053983435


def test_compute_sentence_length_kurtosis():
    result = tpsm.compute_sentence_length_kurtosis(['huhuh.', 'sbbbasdsads', 'jj djdjd', '1'])
    assert result == -1.1057315770736917


def test_compute_average_punctiation():
    result = tpsm.compute_average_puncitation(['..jj', '..asdsad!!asdsd'])
    assert result == 3


def test_count_mistakes():
    counter = tpsm.SpellErrorCounter()
    result = counter.count_mistakes('hi hiw are you blockchain crypto cryptocurrency '
                                    'cryptocurrencies steem steemit bitcoin ethereum')
    assert result == 1


def test_count_connectors():
    result = tpsm.count_connectors(['indeed', 'i', 'like', 'hence'])
    assert result == 2


def test_count_pronouns():
    result = tpsm.count_pronouns(['me', 'and', 'myself', 'like',
                                  'to', 'go', 'to', 'her'])
    assert result == 2


def test_count_characters():
    result = tpsm.count_characters(['hi', 'mark'])
    assert result == 6


def test_count_letters():
    result = tpsm.count_letters('hallo54 my namE!!')
    assert result == 11


def test_syllable_count():
    tokens2syllables = tpsm.SyllableConverter()
    result = tokens2syllables.tokens2syllablses(['i', 'am', 'better', 'crypto', 'currency'])
    assert result == [1, 1, 2, 2, 3]


def test_gunning_fog_index():
    result = tpsm.gunning_fog_index(num_words=20, num_complex_words=10,
                                    num_sentences=2)
    assert result == 24


def test_smog_index():
    result = tpsm.smog_index(30, 1)
    assert result == 34.4191


def test_flesch_kincaid_index():
    result = tpsm.flesch_kincaid_index(3, 2, 1)
    assert result == 77.90500000000002


def test_automated_readability_index():
    result = tpsm.automated_readability_index(1000, 100, 10)
    assert result == 30.67


def test_coleman_liau_index():
    result = tpsm.coleman_liau_index(1000, 100, 10)
    assert result == 40.03999999999999


def test_adverb_estimate():
    result = tpsm.adverb_estimate(['i', 'am', 'heavily', 'in', 'use'])
    assert result == 1


def test_grammar_errors():
    counter = tpsm.GrammarErrorCounter()
    sentences = 'She earn moneyt. I did nothing wrogn. He go in Colorado.'
    matches = counter.count_mistakes_per_character(sentences)
    assert matches == 2 / len(sentences)


def test_count_headdings():
    text= """# heading 
heyho

#### heading123213213232

#################### dksajds
    
jdd
<h4> lkjsdsak!"§$$ </h5>
    """
    nheadings = tpsm.count_headings(text)
    assert nheadings == 3