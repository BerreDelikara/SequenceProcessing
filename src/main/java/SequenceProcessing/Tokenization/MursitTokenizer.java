package SequenceProcessing.Tokenization;

import java.util.Map;
import java.util.regex.Pattern;

/**
 * Byte-level BPE tokenizer for the Mursit (Mecellem) model family
 * ({@code newmindai/Mursit-Base}) — Uğur et al. 2026, "Mecellem Models:
 * Turkish Models Trained from Scratch and Continually Pre-trained for the
 * Legal Domain" (arXiv:2601.16018). Loads the released vocabulary and merge
 * rules from {@code /mursit/} on the classpath.
 *
 * <p>Vocabulary size: 59,008. Unknown token: {@code <unk>}.
 * Pre-tokenizer: ModernBERT-style regex extracted byte-for-byte from
 * Mursit's published {@code tokenizer.json}.
 * License of bundled data: Apache 2.0.
 */
public class MursitTokenizer extends ByteLevelBPETokenizer {

    private static final String UNKNOWN_TOKEN = "<unk>";
    private static final String VOCAB_RESOURCE = "/mursit/vocab.json";
    private static final String MERGES_RESOURCE = "/mursit/merges.txt";

    /** ModernBERT-style pre-tokenizer regex extracted from Mursit's tokenizer.json. */
    private static final Pattern PRE_TOKENIZER = Pattern.compile(
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+"
                    + "|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*"
                    + "|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
            Pattern.UNICODE_CHARACTER_CLASS);

    public MursitTokenizer() {
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
