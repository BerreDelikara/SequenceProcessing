package SequenceProcessing.Tokenization;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Trainable BPE used as the OOV fallback for {@link TurkishTokenizer}.
 * Reproduces the algorithmic core of {@code rsennrich/subword-nmt}'s
 * {@code learn_bpe.py} and {@code apply_bpe.py}: end-of-word marker appended
 * to the last character, frequency-based merges with lexicographic
 * tie-breaking, minimum frequency 2, lowest-rank-wins encoding.
 */
public class TurkishBPEFallback extends Tokenizer {
    private static final String EOW = "</w>";
    private static final int MIN_FREQUENCY = 2;

    private final List<String[]> mergeRules = new ArrayList<>();
    private final Map<String, Integer> mergeRank = new LinkedHashMap<>();
    private final Set<String> vocabulary = new LinkedHashSet<>();

    public TurkishBPEFallback(int vocabSize) {
        super(vocabSize);
    }

    @Override
    public void train(List<String> corpus) {
        // Reset state — train() is idempotent w.r.t. previous calls.
        mergeRules.clear();
        mergeRank.clear();
        vocabulary.clear();

        // Step 1: collect word-frequency counts across the corpus.
        Map<String, Integer> wordCounts = new LinkedHashMap<>();
        for (String sentence : corpus) {
            for (String word : sentence.trim().split("\\s+")) {
                if (!word.isEmpty()) wordCounts.merge(word, 1, Integer::sum);
            }
        }

        // Step 2: explode each unique word into its character-level
        // symbol sequence with the end-of-word marker on the last char
        // (subword-nmt convention). Wrap with its frequency for later
        // statistics passes.
        List<BPEWordEntry> sortedVocab = new ArrayList<>(wordCounts.size());
        for (Map.Entry<String, Integer> e : wordCounts.entrySet()) {
            String word = e.getKey();
            List<String> symbols = new ArrayList<>(word.length());
            for (int i = 0; i < word.length() - 1; i++) {
                symbols.add(String.valueOf(word.charAt(i)));
            }
            symbols.add(word.charAt(word.length() - 1) + EOW);
            sortedVocab.add(new BPEWordEntry(symbols, e.getValue()));
        }
        // Frequency-sort so the initial vocabulary additions are
        // deterministic and the most-common chars are inserted first.
        sortedVocab.sort((a, b) -> Integer.compare(b.freq, a.freq));
        // Seed the vocabulary with every initial symbol (all single chars).
        for (BPEWordEntry we : sortedVocab) vocabulary.addAll(we.symbols);

        // Step 3: BPE merge loop. Each iteration finds the most frequent
        // adjacent symbol pair across the whole corpus, records it as a
        // merge rule, and rewrites every word so future iterations see
        // the merged symbol. Stops when budget exhausted, no pairs left,
        // or the best remaining pair falls below MIN_FREQUENCY.
        int numSymbols = Math.max(0, vocabSize - vocabulary.size());
        for (int iter = 0; iter < numSymbols; iter++) {
            Map<BPESymbolPair, Integer> stats = pairStatistics(sortedVocab);
            if (stats.isEmpty()) break;
            BPESymbolPair best = pickBestPair(stats);
            if (best == null || stats.get(best) < MIN_FREQUENCY) break;
            // Record the merge: rule order = its rank (lowest wins at apply time).
            mergeRules.add(new String[]{best.left, best.right});
            mergeRank.put(best.left + " " + best.right, mergeRules.size() - 1);
            vocabulary.add(best.left + best.right);
            // Rewrite every word with the new merge applied.
            for (BPEWordEntry we : sortedVocab) {
                we.symbols = mergeAdjacent(we.symbols, best.left, best.right);
            }
        }
    }

    @Override
    public List<String> tokenize(String word) {
        // Untrained fallback: return raw character split so the caller
        // still gets a usable segmentation.
        if (mergeRules.isEmpty()) {
            List<String> chars = new ArrayList<>(word.length());
            for (int i = 0; i < word.length(); i++) chars.add(String.valueOf(word.charAt(i)));
            return chars;
        }
        if (word.isEmpty()) return Collections.emptyList();
        if (word.length() == 1) return Collections.singletonList(word);

        // Initial state: one symbol per character, EOW glued to the last.
        List<String> symbols = new ArrayList<>(word.length());
        for (int i = 0; i < word.length() - 1; i++) symbols.add(String.valueOf(word.charAt(i)));
        symbols.add(word.charAt(word.length() - 1) + EOW);

        // Greedy lowest-rank-wins encoding: each pass scans all adjacent
        // pairs, finds the one whose merge rule has the lowest rank
        // (i.e. earliest learned), and applies it. Repeat until no
        // adjacent pair matches any merge rule.
        while (symbols.size() > 1) {
            int bestRank = Integer.MAX_VALUE;
            String bestLeft = null, bestRight = null;
            for (int i = 0; i < symbols.size() - 1; i++) {
                Integer rank = mergeRank.get(symbols.get(i) + " " + symbols.get(i + 1));
                if (rank != null && rank < bestRank) {
                    bestRank = rank;
                    bestLeft = symbols.get(i);
                    bestRight = symbols.get(i + 1);
                }
            }
            if (bestLeft == null) break;
            symbols = mergeAdjacent(symbols, bestLeft, bestRight);
        }
        // Strip EOW from the last symbol before returning — it's an
        // internal training marker, not part of the output token set.
        if (!symbols.isEmpty()) {
            String last = symbols.get(symbols.size() - 1);
            if (last.equals(EOW)) {
                symbols.remove(symbols.size() - 1);
            } else if (last.endsWith(EOW)) {
                symbols.set(symbols.size() - 1, last.substring(0, last.length() - EOW.length()));
            }
        }
        return symbols;
    }

    private static Map<BPESymbolPair, Integer> pairStatistics(List<BPEWordEntry> vocab) {
        Map<BPESymbolPair, Integer> stats = new LinkedHashMap<>();
        for (BPEWordEntry we : vocab) {
            List<String> s = we.symbols;
            for (int i = 0; i < s.size() - 1; i++) {
                stats.merge(new BPESymbolPair(s.get(i), s.get(i + 1)), we.freq, Integer::sum);
            }
        }
        return stats;
    }

    private static BPESymbolPair pickBestPair(Map<BPESymbolPair, Integer> stats) {
        BPESymbolPair best = null;
        int bestFreq = Integer.MIN_VALUE;
        for (Map.Entry<BPESymbolPair, Integer> e : stats.entrySet()) {
            int f = e.getValue();
            BPESymbolPair p = e.getKey();
            if (best == null
                    || f > bestFreq
                    || (f == bestFreq && p.compareTo(best) > 0)) {
                bestFreq = f;
                best = p;
            }
        }
        return best;
    }

    private static List<String> mergeAdjacent(List<String> symbols, String left, String right) {
        List<String> out = new ArrayList<>(symbols.size());
        int i = 0;
        while (i < symbols.size()) {
            if (i < symbols.size() - 1
                    && symbols.get(i).equals(left)
                    && symbols.get(i + 1).equals(right)) {
                out.add(left + right);
                i += 2;
            } else {
                out.add(symbols.get(i));
                i++;
            }
        }
        return out;
    }
}
