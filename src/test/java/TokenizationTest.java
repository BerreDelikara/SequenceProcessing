import SequenceProcessing.Tokenization.BPETokenizer;
import SequenceProcessing.Tokenization.PaperTurkishTokenizer;
import SequenceProcessing.Tokenization.TurkishTokenizer;
import SequenceProcessing.Tokenization.UnigramTokenizer;
import SequenceProcessing.Tokenization.WordPieceTokenizer;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.*;

public class TokenizationTest {

    private static final List<String> CORPUS = Arrays.asList(
            "kitap kitaplar kitaplardan kitaplara kitabı kitaba",
            "araba arabalar arabaları arabadan arabaya",
            "ev evler evlere evden evde evin",
            "adam adamlar adamdan adama adamı",
            "okul okullar okuldan okula okulu"
    );

    private BPETokenizer bpe;
    private WordPieceTokenizer wordPiece;
    private TurkishTokenizer turkish;
    private UnigramTokenizer unigram;

    @Before
    public void setUp() {
        bpe = new BPETokenizer(200);
        bpe.train(CORPUS);

        wordPiece = new WordPieceTokenizer(200);
        wordPiece.train(CORPUS);

        turkish = new TurkishTokenizer(200);
        turkish.train(CORPUS);

        unigram = new UnigramTokenizer(200);
        unigram.train(CORPUS);
    }

    // ── BPE ──────────────────────────────────────────────────────────────────

    @Test
    public void bpeVocabularyNotEmpty() {
        assertFalse(bpe.getVocabulary().isEmpty());
    }

    @Test
    public void bpeTokenizeReturnsNonEmptyList() {
        assertFalse(bpe.tokenize("kitap").isEmpty());
    }

    @Test
    public void bpeTokenizeJoinsBackToOriginal() {
        String word = "kitaplardan";
        assertEquals(word, String.join("", bpe.tokenize(word)));
    }

    @Test
    public void bpeEncodeMultipleWords() {
        assertFalse(bpe.encode("kitap ev").isEmpty());
    }

    @Test
    public void bpeUntrainedFallsBackToChars() {
        BPETokenizer untrained = new BPETokenizer(100);
        List<String> tokens = untrained.tokenize("xyz");
        assertEquals(3, tokens.size());
    }

    @Test
    public void bpeMergeRulesGrow() {
        assertFalse(bpe.getMergeRules().isEmpty());
    }

    // ── WordPiece ─────────────────────────────────────────────────────────────

    @Test
    public void wordPieceVocabularyNotEmpty() {
        assertFalse(wordPiece.getVocabulary().isEmpty());
    }

    @Test
    public void wordPieceTokenizeSeenWord() {
        List<String> tokens = wordPiece.tokenize("kitap");
        assertFalse(tokens.isEmpty());
        assertNotEquals(WordPieceTokenizer.UNK, tokens.get(0));
    }

    @Test
    public void wordPieceUnseenWordReturnsUnk() {
        WordPieceTokenizer small = new WordPieceTokenizer(50);
        small.train(Arrays.asList("merhaba dünya"));
        assertEquals(WordPieceTokenizer.UNK, wordPiece.tokenize("xyzxyzxyz").get(0));
    }

    @Test
    public void wordPieceContinuationPrefixPresent() {
        assertTrue(wordPiece.getVocabulary().stream()
                .anyMatch(t -> t.startsWith(WordPieceTokenizer.CONTINUATION_PREFIX)));
    }

    @Test
    public void wordPieceEncodeMultipleWords() {
        assertFalse(wordPiece.encode("kitap araba").isEmpty());
    }

    // ── TurkishTokenizer ──────────────────────────────────────────────────────

    @Test
    public void turkishTokenizeKnownWordHasRootFirst() {
        List<String> tokens = turkish.tokenize("kitaplardan");
        assertFalse(tokens.isEmpty());
        assertEquals(TurkishTokenizer.ROOT_PREFIX + "kitap", tokens.get(0));
    }

    @Test
    public void turkishTokenizeRootOnly() {
        List<String> tokens = turkish.tokenize("kitap");
        assertFalse(tokens.isEmpty());
        assertEquals(TurkishTokenizer.ROOT_PREFIX + "kitap", tokens.get(0));
    }

    @Test
    public void turkishCapitalizedWordEmitsUppercaseToken() {
        List<String> tokens = turkish.tokenize("Kitap");
        assertFalse(tokens.isEmpty());
        assertEquals(TurkishTokenizer.UPPERCASE_TOKEN, tokens.get(0));
        // Second token should be the root of the lowercased word
        assertTrue(tokens.size() > 1);
        assertEquals(TurkishTokenizer.ROOT_PREFIX + "kitap", tokens.get(1));
    }

    @Test
    public void turkishPunctuationReturnedAsIs() {
        List<String> tokens = turkish.tokenize(".");
        assertEquals(1, tokens.size());
        assertEquals(".", tokens.get(0));
    }

    @Test
    public void turkishOovFallsBackToBpe() {
        List<String> tokens = turkish.tokenize("xkzqfoo");
        assertFalse(tokens.isEmpty());
    }

    @Test
    public void turkishEncodeFullSentence() {
        assertFalse(turkish.encode("kitap evden araba").isEmpty());
    }

    @Test
    public void turkishPluralProducesExtraTag() {
        List<String> singular = turkish.tokenize("kitap");
        List<String> plural = turkish.tokenize("kitaplar");
        assertTrue(plural.size() > singular.size());
    }

    @Test
    public void turkishDecodeRoundtripPlural() {
        assertEquals("kitaplar", turkish.decode(turkish.tokenize("kitaplar")).trim());
    }

    @Test
    public void turkishDecodeRoundtripAblative() {
        assertEquals("evden", turkish.decode(turkish.tokenize("evden")).trim());
    }

    @Test
    public void turkishDecodeRoundtripCapitalized() {
        assertEquals("Kitap", turkish.decode(turkish.tokenize("Kitap")).trim());
    }

    @Test
    public void turkishDecodeRoundtripSentence() {
        assertEquals("kitap evden araba",
                turkish.decode(turkish.encode("kitap evden araba")).trim());
    }

    // ── UnigramTokenizer ──────────────────────────────────────────────────────

    @Test
    public void unigramVocabularyNotEmpty() {
        assertFalse(unigram.getVocabulary().isEmpty());
    }

    @Test
    public void unigramVocabRespectsSizeLimit() {
        assertTrue(unigram.getVocabulary().size() <= 200);
    }

    @Test
    public void unigramTokenizeReturnsNonEmptyList() {
        assertFalse(unigram.tokenize("kitap").isEmpty());
    }

    @Test
    public void unigramTokenizeJoinsBackToOriginal() {
        String word = "kitaplardan";
        assertEquals(word, String.join("", unigram.tokenize(word)));
    }

    @Test
    public void unigramEncodeMultipleWords() {
        assertFalse(unigram.encode("kitap ev").isEmpty());
    }

    @Test
    public void unigramLogProbsAreNegative() {
        for (double lp : unigram.getVocabulary().values()) {
            assertTrue(lp <= 0.0);
        }
    }

    // ── PaperTurkishTokenizer ─────────────────────────────────────────────────
    // These tests are skipped automatically if the paper's lexical resources
    // are not present at src/main/resources/turkish-mft/ — see the README there.

    private PaperTurkishTokenizer loadPaperTokenizerOrSkip() {
        try {
            return new PaperTurkishTokenizer();
        } catch (IllegalStateException e) {
            org.junit.Assume.assumeNoException(
                    "paper-faithful tokenizer resources not installed", e);
            return null;
        }
    }

    @Test
    public void paperVocabularySize() {
        PaperTurkishTokenizer paper = loadPaperTokenizerOrSkip();
        assertTrue(paper.getVocabularySize() > 20000);
    }

    @Test
    public void paperTokenizeKnownWordNonEmpty() {
        PaperTurkishTokenizer paper = loadPaperTokenizerOrSkip();
        assertFalse(paper.tokenize("kitap").isEmpty());
    }

    @Test
    public void paperEncodeIdsNonEmpty() {
        PaperTurkishTokenizer paper = loadPaperTokenizerOrSkip();
        assertFalse(paper.encodeIds("kitap evden araba").isEmpty());
    }

    @Test
    public void paperCapitalizedEmitsUppercaseMarker() {
        PaperTurkishTokenizer paper = loadPaperTokenizerOrSkip();
        List<String> tokens = paper.tokenize("Kitap");
        assertEquals(PaperTurkishTokenizer.UPPERCASE_TOKEN, tokens.get(0));
    }

    @Test
    public void paperReverseDictHasAllIds() {
        PaperTurkishTokenizer paper = loadPaperTokenizerOrSkip();
        List<Integer> ids = paper.encodeIds("kitap");
        for (int id : ids) {
            assertTrue("id " + id + " missing from reverse dict",
                    paper.getReverseDict().containsKey(id));
        }
    }

    @Test
    public void paperRoundtripKitap() {
        PaperTurkishTokenizer paper = loadPaperTokenizerOrSkip();
        assertEquals("kitap", paper.decode(paper.encodeIds("kitap")).trim());
    }

    @Test
    public void paperRoundtripPlural() {
        PaperTurkishTokenizer paper = loadPaperTokenizerOrSkip();
        assertEquals("kitaplar", paper.decode(paper.encodeIds("kitaplar")).trim());
    }

    @Test
    public void paperRoundtripWithCase() {
        PaperTurkishTokenizer paper = loadPaperTokenizerOrSkip();
        assertEquals("Kitap", paper.decode(paper.encodeIds("Kitap")).trim());
    }

    @Test
    public void paperRoundtripAblative() {
        PaperTurkishTokenizer paper = loadPaperTokenizerOrSkip();
        assertEquals("evden", paper.decode(paper.encodeIds("evden")).trim());
    }

    @Test
    public void paperMatchesPaperQualitativeExample() {
        PaperTurkishTokenizer paper = loadPaperTokenizerOrSkip();
        // Sentence from Bayram et al. 2026, p.8 ("Proposed TurkishTokenizer" block).
        String input = "Atasözleri geçmişten günümüze kadar ulaşan "
                + "anlamı bakımından mecazlı bir mana kazanan "
                + "kalıplaşmış sözlerdir.";
        List<String> expected = Arrays.asList(
                "<uppercase>", " atasöz", "leri", " geçmiş", "ten",
                " gün", "üm", "üz", "e", " kadar",
                " ulaş", "an", " anlam", "ı", " bakım", "ın", "dan",
                " mecaz", "lı", " bir", " mana", " kazan", "an",
                " kalıp", "laş", "mış", " söz", "ler", "dir", ".");
        assertEquals(expected, paper.encode(input));
        assertEquals(input, paper.decode(paper.encodeIds(input)).trim());
    }
}
