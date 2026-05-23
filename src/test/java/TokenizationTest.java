import SequenceProcessing.Tokenization.ByteLevelBPETokenizer;
import SequenceProcessing.Tokenization.CosmosGPT2Tokenizer;
import SequenceProcessing.Tokenization.MursitTokenizer;
import SequenceProcessing.Tokenization.TabiBERTTokenizer;
import SequenceProcessing.Tokenization.Tokenizer;
import SequenceProcessing.Tokenization.TurkishTokenizer;
import org.junit.BeforeClass;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/**
 * Unit tests for the four Turkish tokenizers compared in Bayram et al. 2026:
 * {@link TurkishTokenizer}, {@link CosmosGPT2Tokenizer},
 * {@link TabiBERTTokenizer}, {@link MursitTokenizer}.
 */
public class TokenizationTest {

    private static CosmosGPT2Tokenizer cosmos;
    private static TabiBERTTokenizer tabi;
    private static MursitTokenizer mursit;
    private static TurkishTokenizer turkish;

    private static final List<String> TRAINING_CORPUS = Arrays.asList(
            "kitap kitaplar kitaplardan kitaplara kitabı",
            "araba arabalar arabaları",
            "ev evler evden evde evin",
            "okul okullar okuldan okula");

    @BeforeClass
    public static void loadAll() {
        cosmos = new CosmosGPT2Tokenizer();
        tabi = new TabiBERTTokenizer();
        mursit = new MursitTokenizer();
        turkish = new TurkishTokenizer(200);
        turkish.train(TRAINING_CORPUS);
    }

    // ─── Shared tests ──────────────────────────────────────────────────────

    @Test
    public void byteEncoderIsBijection() {
        // Round-trip all 256 byte values through BYTE_TO_CHAR + CHAR_TO_BYTE.
        java.util.Set<Character> seen = new java.util.HashSet<>();
        for (int b = 0; b < 256; b++) {
            char c = ByteLevelBPETokenizer.BYTE_TO_CHAR[b];
            assertTrue("duplicate codepoint in BYTE_TO_CHAR", seen.add(c));
            Integer back = ByteLevelBPETokenizer.CHAR_TO_BYTE.get(c);
            assertNotNull("missing reverse mapping for " + (int) c, back);
            assertEquals("inverse mismatch at byte " + b, b, (int) back);
        }
        assertEquals(256, seen.size());
    }

    @Test
    public void allTokenizersImplementBaseInterface() {
        assertTrue(cosmos instanceof Tokenizer);
        assertTrue(tabi instanceof Tokenizer);
        assertTrue(mursit instanceof Tokenizer);
        assertTrue(turkish instanceof Tokenizer);
    }

    @Test
    public void vocabsLoadFromClasspath() {
        assertFalse(cosmos.getVocabulary().isEmpty());
        assertFalse(tabi.getVocabulary().isEmpty());
        assertFalse(mursit.getVocabulary().isEmpty());
    }

    @Test
    public void byteEncodedSpaceIsGCircumflex() {
        // GPT-2 family maps byte 0x20 (space) to U+0120 = Ġ.
        assertEquals('Ġ', ByteLevelBPETokenizer.BYTE_TO_CHAR[0x20]);
    }

    // ─── CosmosGPT2 ────────────────────────────────────────────────────────

    @Test
    public void cosmosVocabularySize() {
        assertEquals(50257, cosmos.getVocabularySize());
    }

    @Test
    public void cosmosTokenizesAscii() {
        List<String> tokens = cosmos.encode("hello");
        assertFalse(tokens.isEmpty());
    }

    @Test
    public void cosmosTokenizesTurkishDiacritics() {
        List<String> tokens = cosmos.encode("şimdi çığlık");
        assertFalse(tokens.isEmpty());
    }

    @Test
    public void cosmosLeadingSpaceProducesGToken() {
        // " kitap" should start with a Ġ-prefixed token (byte-level encoding).
        List<String> tokens = cosmos.encode(" kitap");
        assertFalse(tokens.isEmpty());
        assertTrue("expected leading Ġ in first token: " + tokens.get(0),
                tokens.get(0).startsWith("Ġ"));
    }

    @Test
    public void cosmosEncodeIdsNonEmpty() {
        List<Integer> ids = cosmos.encodeIds("Merhaba dünya");
        assertFalse(ids.isEmpty());
    }

    @Test
    public void cosmosRoundtripsAsciiSentence() {
        // Pure ASCII without special punctuation should round-trip exactly.
        String input = "hello world";
        String back = cosmos.decode(cosmos.encodeIds(input));
        assertEquals(input, back);
    }

    // ─── TabiBERT ──────────────────────────────────────────────────────────

    @Test
    public void tabiBertVocabularySize() {
        assertEquals(50176, tabi.getVocabularySize());
    }

    @Test
    public void tabiBertTokenizesAscii() {
        List<String> tokens = tabi.encode("hello");
        assertFalse(tokens.isEmpty());
    }

    @Test
    public void tabiBertTokenizesTurkishDiacritics() {
        List<String> tokens = tabi.encode("şimdi çığlık");
        assertFalse(tokens.isEmpty());
    }

    @Test
    public void tabiBertEncodeFullSentence() {
        List<String> tokens = tabi.encode("Atasözleri sözlerdir.");
        assertFalse(tokens.isEmpty());
    }

    @Test
    public void tabiBertSpecialTokensInVocab() {
        assertNotNull(tabi.getVocabulary().get("[PAD]"));
        assertNotNull(tabi.getVocabulary().get("[UNK]"));
        assertNotNull(tabi.getVocabulary().get("[CLS]"));
        assertNotNull(tabi.getVocabulary().get("[SEP]"));
        assertNotNull(tabi.getVocabulary().get("[MASK]"));
    }

    @Test
    public void tabiBertSpecialTokenIdsFixed() {
        assertEquals(Integer.valueOf(0), tabi.getVocabulary().get("[PAD]"));
        assertEquals(Integer.valueOf(1), tabi.getVocabulary().get("[UNK]"));
        assertEquals(Integer.valueOf(4), tabi.getVocabulary().get("[CLS]"));
        assertEquals(Integer.valueOf(5), tabi.getVocabulary().get("[SEP]"));
        assertEquals(Integer.valueOf(6), tabi.getVocabulary().get("[MASK]"));
    }

    // ─── Mursit ────────────────────────────────────────────────────────────

    @Test
    public void mursitVocabularySize() {
        assertEquals(59008, mursit.getVocabularySize());
    }

    @Test
    public void mursitTokenizesAscii() {
        List<String> tokens = mursit.encode("hello");
        assertFalse(tokens.isEmpty());
    }

    @Test
    public void mursitTokenizesTurkishDiacritics() {
        List<String> tokens = mursit.encode("şimdi çığlık");
        assertFalse(tokens.isEmpty());
    }

    @Test
    public void mursitEncodeFullSentence() {
        List<String> tokens = mursit.encode("hukuk davası açıldı.");
        assertFalse(tokens.isEmpty());
    }

    @Test
    public void mursitSpecialTokensInVocab() {
        // From the released special_tokens.txt: <pad>, <s>, </s>, <unk>, <mask>
        assertNotNull(mursit.getVocabulary().get("<unk>"));
        assertNotNull(mursit.getVocabulary().get("<s>"));
        assertNotNull(mursit.getVocabulary().get("</s>"));
    }

    @Test
    public void mursitTokenizationDiffersFromTabi() {
        // Sanity: different vocab → at least some inputs tokenize differently.
        List<Integer> mursitIds = mursit.encodeIds("Türkiye Cumhuriyeti");
        List<Integer> tabiIds = tabi.encodeIds("Türkiye Cumhuriyeti");
        // They might have a few overlapping IDs but the lists shouldn't be identical.
        assertNotEquals(mursitIds, tabiIds);
    }

    // ─── TurkishTokenizer ──────────────────────────────────────────────────

    @Test
    public void turkishPunctuationPassthrough() {
        assertEquals(Arrays.asList("."), turkish.tokenize("."));
        assertEquals(Arrays.asList("!"), turkish.tokenize("!"));
    }

    @Test
    public void turkishCapitalizedEmitsUppercaseToken() {
        List<String> tokens = turkish.tokenize("Kitap");
        assertEquals(TurkishTokenizer.UPPERCASE_TOKEN, tokens.get(0));
        assertTrue("expected at least 2 tokens", tokens.size() >= 2);
        assertEquals(TurkishTokenizer.ROOT_PREFIX + "kitap", tokens.get(1));
    }

    @Test
    public void turkishRootHasLeadingSpace() {
        List<String> tokens = turkish.tokenize("kitap");
        assertFalse(tokens.isEmpty());
        assertEquals(TurkishTokenizer.ROOT_PREFIX + "kitap", tokens.get(0));
    }

    @Test
    public void turkishPluralProducesExtraTag() {
        List<String> singular = turkish.tokenize("kitap");
        List<String> plural = turkish.tokenize("kitaplar");
        assertTrue("plural should produce more tokens than singular",
                plural.size() > singular.size());
    }

    @Test
    public void turkishKnownWordTokenizes() {
        List<String> tokens = turkish.tokenize("kitaplardan");
        assertFalse(tokens.isEmpty());
        // Root present as first token (with leading space)
        assertEquals(TurkishTokenizer.ROOT_PREFIX + "kitap", tokens.get(0));
    }

    @Test
    public void turkishLenitionPreservesSurface() {
        // kitabı's lemma is kitap, but surface starts with "kitab" (lenition p->b).
        // The encoder should emit the softened root so encode→decode-ish is lossless.
        List<String> tokens = turkish.tokenize("kitabı");
        assertFalse(tokens.isEmpty());
        assertEquals(TurkishTokenizer.ROOT_PREFIX + "kitab", tokens.get(0));
    }

    @Test
    public void turkishEncodeSentence() {
        List<String> tokens = turkish.encode("kitap evden araba");
        assertFalse(tokens.isEmpty());
    }

    @Test
    public void turkishOOVFallsBackToBpe() {
        // "xkzqfoo" is not Turkish — morph analyzer fails, falls through to BPE.
        List<String> tokens = turkish.tokenize("xkzqfoo");
        assertFalse(tokens.isEmpty());
    }

    @Test
    public void turkishDropsDefaultAffixes() {
        // "kitap" parsed as kitap+NOUN+A3SG+PNON+NOM should emit ONLY the root —
        // NOUN, A3SG, PNON, NOM are POS labels and defaults that we drop.
        List<String> tokens = turkish.tokenize("kitap");
        assertEquals(1, tokens.size());
    }

    @Test
    public void turkishFallbackIsTrainable() {
        // Train on a tiny corpus containing a synthetic OOV; then tokenize one
        // of its words and confirm we got something non-empty.
        TurkishTokenizer fresh = new TurkishTokenizer(100);
        fresh.train(Arrays.asList("xkzfoo xkzbar xkzbaz xkzqux xkzquux"));
        List<String> tokens = fresh.tokenize("xkzwhatever");
        assertFalse(tokens.isEmpty());
    }

    @Test
    public void turkishCapitalizedRoundtripsCase() {
        List<String> kitapTokens = turkish.tokenize("kitap");
        List<String> kitapTokensCap = turkish.tokenize("Kitap");
        // Capitalized version should have exactly one more token (<uppercase>).
        assertEquals(kitapTokens.size() + 1, kitapTokensCap.size());
    }

    @Test
    public void turkishDerivationalMorphemesPreservedAcrossDbBoundary() {
        // çalıştırıldı parses as çalış+VERB^DB+VERB+CAUS^DB+VERB+PASS+POS+PAST+A3SG.
        // The CAUS and PASS derivational suffixes live in groups after the
        // first ^DB+. They must not be silently dropped.
        List<String> tokens = turkish.tokenize("çalıştırıldı");
        assertEquals(Arrays.asList(
                TurkishTokenizer.ROOT_PREFIX + "çalış", "CAUS", "PASS", "PAST"),
                tokens);
    }

    @Test
    public void turkishUnmappedDerivationalTagsPassThrough() {
        // kitaplaşmış has a NOUN→VERB derivation with the BECOME marker, which
        // is NOT in AFFIX_MAP. The token stream should preserve it as-is
        // rather than swallow it (the prior policy dropped any unmapped tag).
        List<String> tokens = turkish.tokenize("kitaplaşmış");
        assertTrue("BECOME morpheme should be preserved: " + tokens,
                tokens.contains("BECOME"));
    }

    @Test
    public void turkishBareInfPreserved() {
        // okutturulamamak ends in +NOUN+INF+A3SG+PNON+NOM. The infinitive marker
        // appears here as the bare "INF" tag (not INF1/INF2/INF3), which means
        // AFFIX_MAP doesn't rename it. It should still be emitted.
        List<String> tokens = turkish.tokenize("okutturulamamak");
        assertTrue("INF morpheme should be preserved: " + tokens,
                tokens.contains("INF"));
    }
}
