package SequenceProcessing.Tokenization;

import java.util.Map;
import java.util.regex.Pattern;

/**
 * Byte-level BPE tokenizer for the TabiBERT model
 * ({@code boun-tabilab/TabiBERT}) — Türker et al. 2026,
 * "TabiBERT: A Large-Scale ModernBERT Foundation Model and A Unified Benchmark
 * for Turkish" (arXiv:2512.23065). Loads the released vocabulary and merge
 * rules from {@code /tabi-bert/} on the classpath.
 *
 * <p>Vocabulary size: 50,176. Unknown token: {@code [UNK]}.
 * Pre-tokenizer: ModernBERT-style regex shared with {@link MursitTokenizer}.
 * The regex shipped in TabiBERT's own {@code tokenizer.json} contains
 * over-escaped backslashes that would not behave as intended in any standard
 * regex engine; this Java port uses the cleaner equivalent from the
 * ModernBERT-family tokenizer family (same morphological behavior).
 *
 * <p>License of bundled data: Apache 2.0.
 */
public class TabiBERTTokenizer extends ByteLevelBPETokenizer {

    private static final String UNKNOWN_TOKEN = "[UNK]";

    /** ModernBERT-style pre-tokenizer regex (shared with Mursit). */
    private static final Pattern PRE_TOKENIZER = Pattern.compile(
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+"
                    + "|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*"
                    + "|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
            Pattern.UNICODE_CHARACTER_CLASS);

    public TabiBERTTokenizer() {
        super();
    }

    @Override
    protected String vocabResourcePath() {
        return "/tabi-bert/vocab.json";
    }

    @Override
    protected String mergesResourcePath() {
        return "/tabi-bert/merges.txt";
    }

    @Override
    protected Pattern preTokenizerPattern() {
        return PRE_TOKENIZER;
    }

    @Override
    protected int resolveUnknownTokenId(Map<String, Integer> vocab) {
        Integer id = vocab.get(UNKNOWN_TOKEN);
        return id != null ? id : -1;
    }
}
