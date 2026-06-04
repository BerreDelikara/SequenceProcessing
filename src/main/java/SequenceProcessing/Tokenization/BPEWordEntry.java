package SequenceProcessing.Tokenization;

import java.util.List;

/**
 * Word + frequency record used during BPE training in {@link TurkishBPEFallback}.
 * Symbols are mutated in place as merges are applied; the frequency is fixed
 * once collected from the corpus.
 */
final class BPEWordEntry {
    List<String> symbols;
    final int freq;

    BPEWordEntry(List<String> symbols, int freq) {
        this.symbols = symbols;
        this.freq = freq;
    }
}
