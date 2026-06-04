package SequenceProcessing.Tokenization;

/**
 * Ordered pair of adjacent BPE symbols, used as a map key during merge
 * statistics collection in {@link TurkishBPEFallback}. Natural ordering is
 * lexicographic on (left, right) for deterministic tie-breaking when two
 * pairs share the same corpus frequency.
 */
final class BPESymbolPair implements Comparable<BPESymbolPair> {
    final String left;
    final String right;

    BPESymbolPair(String left, String right) {
        this.left = left;
        this.right = right;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof BPESymbolPair)) return false;
        BPESymbolPair p = (BPESymbolPair) o;
        return left.equals(p.left) && right.equals(p.right);
    }

    @Override
    public int hashCode() {
        return 31 * left.hashCode() + right.hashCode();
    }

    @Override
    public int compareTo(BPESymbolPair o) {
        int c = left.compareTo(o.left);
        return c != 0 ? c : right.compareTo(o.right);
    }
}
