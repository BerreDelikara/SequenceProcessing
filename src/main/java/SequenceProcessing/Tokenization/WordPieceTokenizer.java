package SequenceProcessing.Tokenization;

import java.util.*;

/**
 * WordPiece tokenizer (Schuster &amp; Nakajima, 2012). Like BPE but selects the
 * pair to merge by likelihood score {@code freq(AB) / (freq(A) * freq(B))}
 * rather than raw co-occurrence count. Continuation pieces inside a word are
 * marked with the prefix {@code ##}.
 */
public class WordPieceTokenizer extends Tokenizer {

    /** Prefix attached to subword tokens that do not start a word. */
    public static final String CONTINUATION_PREFIX = "##";

    /** Token returned when a word cannot be segmented from the vocabulary. */
    public static final String UNK = "[UNK]";

    private final Set<String> vocabulary;

    /**
     * Constructs a WordPiece tokenizer with the given target vocabulary size.
     * @param vocabSize Maximum number of tokens to learn.
     */
    public WordPieceTokenizer(int vocabSize) {
        super(vocabSize);
        this.vocabulary = new LinkedHashSet<>();
    }

    /**
     * Builds the WordPiece vocabulary by repeatedly merging the adjacent
     * token pair with the highest likelihood score until {@link #getVocabSize()}
     * is reached.
     * @param corpus List of whitespace-separated sentences.
     */
    @Override
    public void train(List<String> corpus) {
        vocabulary.clear();

        Map<String, Integer> wordFreqs = new LinkedHashMap<>();
        for (String sentence : corpus) {
            for (String word : sentence.trim().split("\\s+")) {
                if (!word.isEmpty()) {
                    wordFreqs.merge(word, 1, Integer::sum);
                }
            }
        }

        Map<List<String>, Integer> wordTokens = new LinkedHashMap<>();
        for (Map.Entry<String, Integer> entry : wordFreqs.entrySet()) {
            wordTokens.put(wordToInitialTokens(entry.getKey()), entry.getValue());
        }

        for (List<String> tokens : wordTokens.keySet()) {
            vocabulary.addAll(tokens);
        }

        while (vocabulary.size() < vocabSize) {
            Map<String, Integer> tokenFreqs = computeTokenFreqs(wordTokens);

            Map<String, Double> pairScores = new LinkedHashMap<>();
            for (Map.Entry<List<String>, Integer> entry : wordTokens.entrySet()) {
                List<String> tokens = entry.getKey();
                int wordFreq = entry.getValue();
                for (int i = 0; i < tokens.size() - 1; i++) {
                    String a = tokens.get(i);
                    String b = tokens.get(i + 1);
                    String pair = a + "\t" + b;
                    double score = (double) wordFreq
                            / ((double) tokenFreqs.get(a) * tokenFreqs.get(b));
                    pairScores.merge(pair, score, Double::sum);
                }
            }

            if (pairScores.isEmpty()) break;

            String bestPair = Collections.max(pairScores.entrySet(),
                    Map.Entry.comparingByValue()).getKey();
            String[] parts = bestPair.split("\t");
            String merged = merge(parts[0], parts[1]);
            vocabulary.add(merged);

            wordTokens = applyMerge(wordTokens, parts[0], parts[1], merged);
        }
    }

    /**
     * Segments a word using greedy longest-match against the vocabulary.
     * Pieces after the first carry the {@link #CONTINUATION_PREFIX}. Returns
     * a single {@link #UNK} token if no valid segmentation exists.
     * @param word The word to tokenize.
     * @return Ordered list of WordPiece tokens, or {@code [UNK]} on failure.
     */
    @Override
    public List<String> tokenize(String word) {
        if (vocabulary.isEmpty()) {
            return wordToInitialTokens(word);
        }

        List<String> result = new ArrayList<>();
        int start = 0;
        while (start < word.length()) {
            int end = word.length();
            String found = null;
            while (start < end) {
                String substr = word.substring(start, end);
                String candidate = (start == 0) ? substr : CONTINUATION_PREFIX + substr;
                if (vocabulary.contains(candidate)) {
                    found = candidate;
                    break;
                }
                end--;
            }
            if (found == null) {
                return Collections.singletonList(UNK);
            }
            result.add(found);
            start = end;
        }
        return result;
    }

    private List<String> wordToInitialTokens(String word) {
        List<String> tokens = new ArrayList<>();
        for (int i = 0; i < word.length(); i++) {
            String ch = String.valueOf(word.charAt(i));
            tokens.add(i == 0 ? ch : CONTINUATION_PREFIX + ch);
        }
        return tokens;
    }

    private Map<String, Integer> computeTokenFreqs(Map<List<String>, Integer> wordTokens) {
        Map<String, Integer> freqs = new LinkedHashMap<>();
        for (Map.Entry<List<String>, Integer> entry : wordTokens.entrySet()) {
            for (String tok : entry.getKey()) {
                freqs.merge(tok, entry.getValue(), Integer::sum);
            }
        }
        return freqs;
    }

    private Map<List<String>, Integer> applyMerge(Map<List<String>, Integer> wordTokens,
                                                   String left, String right, String merged) {
        Map<List<String>, Integer> result = new LinkedHashMap<>();
        for (Map.Entry<List<String>, Integer> entry : wordTokens.entrySet()) {
            List<String> tokens = entry.getKey();
            List<String> newTokens = new ArrayList<>();
            int i = 0;
            while (i < tokens.size()) {
                if (i < tokens.size() - 1
                        && tokens.get(i).equals(left)
                        && tokens.get(i + 1).equals(right)) {
                    newTokens.add(merged);
                    i += 2;
                } else {
                    newTokens.add(tokens.get(i));
                    i++;
                }
            }
            result.merge(newTokens, entry.getValue(), Integer::sum);
        }
        return result;
    }

    private String merge(String left, String right) {
        String rightPart = right.startsWith(CONTINUATION_PREFIX)
                ? right.substring(CONTINUATION_PREFIX.length())
                : right;
        return left + rightPart;
    }

    /**
     * Returns the learned WordPiece vocabulary.
     * @return Unmodifiable set of all tokens (including {@code ##}-prefixed continuations).
     */
    public Set<String> getVocabulary() {
        return Collections.unmodifiableSet(vocabulary);
    }
}
