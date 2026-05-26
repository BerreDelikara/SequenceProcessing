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
 * Pre-tokenizer: the regex shipped in TabiBERT's own {@code tokenizer.json}
 * is malformed — it contains over-escaped backslashes (e.g. {@code \\p{N}}
 * instead of {@code \p{N}}) that no standard regex engine treats as the
 * intended Unicode property classes. This Java port substitutes the
 * well-formed equivalent that ships in the sibling ModernBERT-family Turkish
 * tokenizer ({@link MursitTokenizer}'s {@code tokenizer.json}, verified
 * byte-for-byte). The two regex literals in this file and {@link MursitTokenizer}
 * are intentionally identical.
 *
 * <p>License of bundled data: Apache 2.0.
 */
public class TabiBERTTokenizer extends ByteLevelBPETokenizer {

    private static final String UNKNOWN_TOKEN = "[UNK]";
    private static final String VOCAB_RESOURCE = "/tabi-bert/vocab.json";
    private static final String MERGES_RESOURCE = "/tabi-bert/merges.txt";


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
        return VOCAB_RESOURCE;
    }

    @Override
    protected String mergesResourcePath() {
        return MERGES_RESOURCE;
    }

    @Override
    protected Pattern preTokenizerPattern() {
        return PRE_TOKENIZER;
    }

    @Override
    protected int resolveUnknownTokenId(Map<String, Integer> vocab) {
        Integer id = vocab.get(UNKNOWN_TOKEN);
            if (id == null) {
                throw new IllegalStateException("Vocabulary does not contain required unknown token: " + UNKNOWN_TOKEN);
            }
        return id;
    }
}
