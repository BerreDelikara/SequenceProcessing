package SequenceProcessing.Tokenization;

import java.util.List;

public abstract class Tokenizer {

    protected int vocabSize;

    public Tokenizer(int vocabSize) {
        this.vocabSize = vocabSize;
    }

    public abstract void train(List<String> corpus);

    public abstract List<String> tokenize(String word);

    public List<String> encode(String sentence) {
        List<String> tokens = new java.util.ArrayList<>();
        for (String word : sentence.trim().split("\\s+")) {
            if (!word.isEmpty()) {
                tokens.addAll(tokenize(word));
            }
        }
        return tokens;
    }

    public int getVocabSize() {
        return vocabSize;
    }
}
