package SequenceProcessing.Tokenization;

import java.util.*;

public class UnigramTokenizer extends Tokenizer {

    private static final int MAX_SUBWORD_LENGTH = 20;
    private static final double PRUNE_RATIO = 0.20;

    private Map<String, Double> logProbs;

    public UnigramTokenizer(int vocabSize) {
        super(vocabSize);
        this.logProbs = new LinkedHashMap<>();
    }

    @Override
    public void train(List<String> corpus) {
        logProbs.clear();

        // Count word frequencies
        Map<String, Integer> wordFreqs = new LinkedHashMap<>();
        for (String sentence : corpus) {
            for (String word : sentence.trim().split("\\s+")) {
                if (!word.isEmpty()) {
                    wordFreqs.merge(word, 1, Integer::sum);
                }
            }
        }

        // Initialise vocabulary: all substrings up to MAX_SUBWORD_LENGTH
        Map<String, Double> counts = new LinkedHashMap<>();
        for (Map.Entry<String, Integer> entry : wordFreqs.entrySet()) {
            String word = entry.getKey();
            int freq = entry.getValue();
            for (int start = 0; start < word.length(); start++) {
                for (int end = start + 1; end <= Math.min(word.length(), start + MAX_SUBWORD_LENGTH); end++) {
                    String sub = word.substring(start, end);
                    counts.merge(sub, (double) freq, Double::sum);
                }
            }
        }

        // Convert counts to log-probabilities
        double total = counts.values().stream().mapToDouble(Double::doubleValue).sum();
        for (Map.Entry<String, Double> e : counts.entrySet()) {
            logProbs.put(e.getKey(), Math.log(e.getValue() / total));
        }

        // EM-based pruning loop: prune until vocab size reached
        while (logProbs.size() > vocabSize) {
            // E-step: compute expected counts via Viterbi for each word
            Map<String, Double> expectedCounts = new LinkedHashMap<>();
            for (Map.Entry<String, Integer> entry : wordFreqs.entrySet()) {
                String word = entry.getKey();
                int freq = entry.getValue();
                List<String> segmentation = viterbi(word);
                for (String token : segmentation) {
                    expectedCounts.merge(token, (double) freq, Double::sum);
                }
            }

            // Remove tokens not used in any segmentation
            logProbs.keySet().retainAll(expectedCounts.keySet());

            // If already within size, stop
            if (logProbs.size() <= vocabSize) break;

            // M-step: recompute log-probs from expected counts
            double totalExpected = expectedCounts.values().stream().mapToDouble(Double::doubleValue).sum();
            for (String token : new ArrayList<>(logProbs.keySet())) {
                double count = expectedCounts.getOrDefault(token, 0.0);
                logProbs.put(token, Math.log(count / totalExpected));
            }

            // Prune bottom PRUNE_RATIO fraction by log-prob, but always keep single characters
            int targetPrune = (int) (logProbs.size() * PRUNE_RATIO);
            if (targetPrune == 0) break;

            List<Map.Entry<String, Double>> sorted = new ArrayList<>(logProbs.entrySet());
            sorted.sort(Map.Entry.comparingByValue());

            int pruned = 0;
            for (Map.Entry<String, Double> e : sorted) {
                if (pruned >= targetPrune) break;
                if (e.getKey().length() > 1) {
                    logProbs.remove(e.getKey());
                    pruned++;
                }
            }
        }
    }

    @Override
    public List<String> tokenize(String word) {
        if (logProbs.isEmpty()) {
            List<String> chars = new ArrayList<>();
            for (char c : word.toCharArray()) chars.add(String.valueOf(c));
            return chars;
        }
        return viterbi(word);
    }

    private List<String> viterbi(String word) {
        int n = word.length();
        double[] dp = new double[n + 1];
        int[] backtrack = new int[n + 1];
        Arrays.fill(dp, Double.NEGATIVE_INFINITY);
        Arrays.fill(backtrack, -1);
        dp[0] = 0.0;

        for (int i = 1; i <= n; i++) {
            for (int j = Math.max(0, i - MAX_SUBWORD_LENGTH); j < i; j++) {
                String sub = word.substring(j, i);
                Double lp = logProbs.get(sub);
                if (lp != null && dp[j] + lp > dp[i]) {
                    dp[i] = dp[j] + lp;
                    backtrack[i] = j;
                }
            }
            // Fallback: if no token covers position i, use the single character
            if (dp[i] == Double.NEGATIVE_INFINITY) {
                String ch = String.valueOf(word.charAt(i - 1));
                double lp = logProbs.getOrDefault(ch, Math.log(1e-10));
                if (dp[i - 1] + lp > dp[i]) {
                    dp[i] = dp[i - 1] + lp;
                    backtrack[i] = i - 1;
                }
            }
        }

        // Reconstruct path
        List<String> tokens = new ArrayList<>();
        int pos = n;
        while (pos > 0) {
            int prev = backtrack[pos];
            tokens.add(0, word.substring(prev, pos));
            pos = prev;
        }
        return tokens;
    }

    public Map<String, Double> getVocabulary() {
        return Collections.unmodifiableMap(logProbs);
    }
}
