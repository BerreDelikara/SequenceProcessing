package SequenceProcessing.Tokenization;

import java.util.List;

/**
 * Base class for subword tokenizers. Subclasses learn a vocabulary from a
 * corpus and segment individual words into tokens.
 */
public abstract class Tokenizer {

    protected int vocabSize;

    /**
     * Constructs a tokenizer with the given target vocabulary size.
     * @param vocabSize Maximum number of tokens to learn during training.
     */
    public Tokenizer(int vocabSize) {
        this.vocabSize = vocabSize;
    }

    /**
     * Trains the tokenizer on the given corpus.
     * @param corpus List of sentences used to build the vocabulary.
     */
    public abstract void train(List<String> corpus);

    /**
     * Splits a single word into its learned subword tokens.
     * @param word The word to tokenize.
     * @return Ordered list of tokens whose concatenation reconstructs the word.
     */
    public abstract List<String> tokenize(String word);

    /**
     * Tokenizes a whitespace-separated sentence by tokenizing each word in turn.
     * @param sentence Sentence to encode.
     * @return Concatenated token list across all words in the sentence.
     */
    public List<String> encode(String sentence) {
        List<String> tokens = new java.util.ArrayList<>();
        for (String word : sentence.trim().split("\\s+")) {
            if (!word.isEmpty()) {
                tokens.addAll(tokenize(word));
            }
        }
        return tokens;
    }

    /**
     * Returns the target vocabulary size set at construction time.
     * @return The configured vocabulary size.
     */
    public int getVocabSize() {
        return vocabSize;
    }
}
