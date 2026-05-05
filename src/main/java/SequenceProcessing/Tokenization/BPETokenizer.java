package SequenceProcessing.Tokenization;

import java.util.*;

public class BPETokenizer extends Tokenizer {

    private final List<String[]> mergeRules;
    private final Set<String> vocabulary;

    public BPETokenizer(int vocabSize) {
        super(vocabSize);
        this.mergeRules = new ArrayList<>();
        this.vocabulary = new LinkedHashSet<>();
    }

    @Override
    public void train(List<String> corpus) {
        mergeRules.clear();
        vocabulary.clear();

        Map<String, Integer> wordFreqs = new LinkedHashMap<>();
        for (String sentence : corpus) {
            for (String word : sentence.trim().split("\\s+")) {
                if (!word.isEmpty()) {
                    wordFreqs.merge(word, 1, Integer::sum);
                }
            }
        }

        Map<String, Integer> vocab = new LinkedHashMap<>();
        for (Map.Entry<String, Integer> entry : wordFreqs.entrySet()) {
            String word = entry.getKey();
            StringBuilder spaced = new StringBuilder();
            for (int i = 0; i < word.length(); i++) {
                if (i > 0) spaced.append(' ');
                spaced.append(word.charAt(i));
            }
            spaced.append("</w>");
            vocab.put(spaced.toString(), entry.getValue());
        }

        for (String wordRepr : vocab.keySet()) {
            for (String sym : wordRepr.split(" ")) {
                vocabulary.add(sym);
            }
        }

        while (vocabulary.size() < vocabSize) {
            Map<String, Integer> pairFreqs = getPairFrequencies(vocab);
            if (pairFreqs.isEmpty()) break;

            String bestPair = Collections.max(pairFreqs.entrySet(),
                    Map.Entry.comparingByValue()).getKey();

            String[] parts = bestPair.split("\t");
            mergeRules.add(parts);
            vocabulary.add(parts[0] + parts[1]);

            vocab = applyMerge(vocab, parts[0], parts[1]);
        }
    }

    @Override
    public List<String> tokenize(String word) {
        if (mergeRules.isEmpty()) {
            List<String> chars = new ArrayList<>();
            for (char c : word.toCharArray()) chars.add(String.valueOf(c));
            return chars;
        }

        List<String> symbols = new ArrayList<>();
        for (int i = 0; i < word.length(); i++) {
            symbols.add(String.valueOf(word.charAt(i)));
        }
        symbols.add("</w>");

        for (String[] rule : mergeRules) {
            symbols = applyMergeToList(symbols, rule[0], rule[1]);
        }

        if (!symbols.isEmpty()) {
            String last = symbols.get(symbols.size() - 1);
            symbols.set(symbols.size() - 1, last.replace("</w>", ""));
            if (symbols.get(symbols.size() - 1).isEmpty()) {
                symbols.remove(symbols.size() - 1);
            }
        }
        return symbols;
    }

    private Map<String, Integer> getPairFrequencies(Map<String, Integer> vocab) {
        Map<String, Integer> pairs = new LinkedHashMap<>();
        for (Map.Entry<String, Integer> entry : vocab.entrySet()) {
            String[] symbols = entry.getKey().split(" ");
            int freq = entry.getValue();
            for (int i = 0; i < symbols.length - 1; i++) {
                String pair = symbols[i] + "\t" + symbols[i + 1];
                pairs.merge(pair, freq, Integer::sum);
            }
        }
        return pairs;
    }

    private Map<String, Integer> applyMerge(Map<String, Integer> vocab, String left, String right) {
        Map<String, Integer> newVocab = new LinkedHashMap<>();
        String pattern = left + " " + right;
        String replacement = left + right;
        for (Map.Entry<String, Integer> entry : vocab.entrySet()) {
            String newKey = entry.getKey().replace(pattern, replacement);
            newVocab.merge(newKey, entry.getValue(), Integer::sum);
        }
        return newVocab;
    }

    private List<String> applyMergeToList(List<String> symbols, String left, String right) {
        List<String> result = new ArrayList<>();
        int i = 0;
        while (i < symbols.size()) {
            if (i < symbols.size() - 1
                    && symbols.get(i).equals(left)
                    && symbols.get(i + 1).equals(right)) {
                result.add(left + right);
                i += 2;
            } else {
                result.add(symbols.get(i));
                i++;
            }
        }
        return result;
    }

    public Set<String> getVocabulary() {
        return Collections.unmodifiableSet(vocabulary);
    }

    public List<String[]> getMergeRules() {
        return Collections.unmodifiableList(mergeRules);
    }
}
