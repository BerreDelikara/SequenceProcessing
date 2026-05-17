package SequenceProcessing.Tokenization;

import java.util.Map;
import java.util.regex.Pattern;

/**
 * Byte-level BPE tokenizer for the CosmosGPT2 model
 * ({@code ytu-ce-cosmos/turkish-gpt2-large}) — Kesgin et al. 2024,
 * "Introducing cosmosGPT: Monolingual Training for Turkish Language Models"
 * (arXiv:2404.17336). Loads the released vocabulary and merge rules from
 * {@code /cosmos-gpt2/} on the classpath.
 *
 * <p>Vocabulary size: 50,257. Unknown token: {@code <|endoftext|>}.
 * Pre-tokenizer: standard GPT-2 regex.
 * License of bundled data: MIT.
 */
public class CosmosGPT2Tokenizer extends ByteLevelBPETokenizer {

    private static final String UNKNOWN_TOKEN = "<|endoftext|>";

    /** Standard GPT-2 pre-tokenizer regex (Radford et al. 2019). */
    private static final Pattern PRE_TOKENIZER = Pattern.compile(
            "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+",
            Pattern.UNICODE_CHARACTER_CLASS);

    public CosmosGPT2Tokenizer() {
        super();
    }

    @Override
    protected String vocabResourcePath() {
        return "/cosmos-gpt2/vocab.json";
    }

    @Override
    protected String mergesResourcePath() {
        return "/cosmos-gpt2/merges.txt";
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
