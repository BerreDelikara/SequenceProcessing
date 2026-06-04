package SequenceProcessing.Tokenization;

import MorphologicalAnalysis.FsmMorphologicalAnalyzer;
import MorphologicalAnalysis.FsmParse;
import MorphologicalAnalysis.FsmParseList;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;
import java.util.regex.Matcher;

/**
 * Hybrid morphology-aware Turkish tokenizer following Algorithm 1 of Bayram
 * et al. 2026 ("Tokens with Meaning: A Hybrid Tokenization Approach for
 * Turkish", arXiv:2508.14292), implemented on top of the
 * {@link FsmMorphologicalAnalyzer} from the StarlangSoftware
 * {@code MorphologicalAnalysis} library.
 *
 * <p>Pipeline per word: punctuation passes through as-is; capitalized words
 * emit a leading {@code <uppercase>} marker and are then lowercased; the
 * morphological analyzer is queried, and the parse with the shortest root and
 * most suffixes is selected. The root is emitted with a leading space (the
 * paper's word-boundary convention) followed by canonical affix identifiers
 * (allomorphic variants collapsed: e.g. {@code A3PL} -> {@code PL},
 * {@code ABL} -> {@code ABL}, default markers like {@code A3SG}/{@code PNON}/
 * {@code NOM} dropped). Words the analyzer cannot parse fall through to a
 * trainable BPE learned during {@link #train(List)} on the OOV subset of the
 * corpus.
 *
 * <p>No external data files are required: morphological knowledge comes from
 * the {@code MorphologicalAnalysis} dependency.
 */
public class TurkishTokenizer extends Tokenizer {

    public static final String UPPERCASE_TOKEN = "<uppercase>";
    public static final String ROOT_PREFIX = " ";

    private static final Locale TURKISH_LOCALE = new Locale("tr", "TR");

    private static final Map<String, String> AFFIX_MAP = createAffixMap();

    /**
     * FSM tags that carry no morphological content on their own and should not
     * appear in the token stream: part-of-speech category labels, the default
     * agreement/case/possessive/polarity markers, and the silent-derivation
     * marker. Tags not in this set and not in {@link #AFFIX_MAP} pass through
     * as-is so derivational morphemes like {@code BECOME}, {@code ACQUIRE},
     * {@code AGT} are preserved.
     */
    private static final Set<String> DROP_TAGS = Collections.unmodifiableSet(new HashSet<>(Arrays.asList(
            "NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "CONJ", "INTERJ",
            "POSTP", "NUM", "PROP", "CARD", "ORD", "REAL", "RANGE", "PUNC",
            "A3SG", "PNON", "NOM", "POS", "ZERO")));

    private static final java.util.regex.Pattern WORD_OR_PUNCTUATION = 
            Pattern.compile("\\p{L}[\\p{L}\\p{M}\\p{N}']*|\\p{N}+|[^\\s]");

    private static Map<String, String> createAffixMap() {
        Map<String, String> m = new HashMap<>();
        m.put("A1SG", "1SG"); m.put("A2SG", "2SG");
        m.put("A1PL", "1PL"); m.put("A2PL", "2PL"); m.put("A3PL", "PL");
        m.put("P1SG", "POSS_1SG"); m.put("P2SG", "POSS_2SG"); m.put("P3SG", "POSS_3SG");
        m.put("P1PL", "POSS_1PL"); m.put("P2PL", "POSS_2PL"); m.put("P3PL", "POSS_3PL");
        m.put("ACC", "ACC"); m.put("DAT", "DAT"); m.put("LOC", "LOC");
        m.put("ABL", "ABL"); m.put("GEN", "GEN"); m.put("INS", "INS");
        m.put("EQU", "EQU");
        m.put("NEG", "NEG"); m.put("ABLE", "ABLE");
        m.put("PAST", "PAST"); m.put("NARR", "NARR"); m.put("PRES", "PRES");
        m.put("FUT", "FUT"); m.put("AOR", "AOR");
        m.put("PROG1", "PROG"); m.put("PROG2", "PROG");
        m.put("COND", "COND"); m.put("IMP", "IMP"); m.put("OPT", "OPT");
        m.put("DESR", "DESR"); m.put("NECES", "NECES");
        m.put("PASS", "PASS"); m.put("CAUS", "CAUS");
        m.put("RECIP", "RECIP"); m.put("REFLEX", "REFLEX");
        m.put("INF1", "INF"); m.put("INF2", "INF"); m.put("INF3", "INF");
        m.put("PASTPART", "PASTPART"); m.put("PRESPART", "PRESPART"); m.put("FUTPART", "FUTPART");
        m.put("AORPART", "AORPART");
        return Collections.unmodifiableMap(m);
    }

    private final FsmMorphologicalAnalyzer analyzer;
    private final InnerBPE fallback;

    /**
     * @param vocabSize Target vocabulary size for the trainable BPE fallback.
     *                  The morphological side does not depend on this.
     */
    public TurkishTokenizer(int vocabSize) {
        super(vocabSize);
        this.analyzer = new FsmMorphologicalAnalyzer();
        this.fallback = new InnerBPE(vocabSize);
    }

    /**
     * Trains the BPE fallback on the subset of {@code corpus} whose words the
     * morphological analyzer cannot parse. Words it can parse are excluded so
     * the BPE vocabulary focuses on residual stems and out-of-dictionary forms
     * (per Bayram et al. 2026, Section 3).
     * @param corpus Sentences used for training; only the OOV subset is forwarded
     *               to the BPE fallback.
     */
    @Override
    public void train(List<String> corpus) {
        // Build an OOV-only sub-corpus: keep only words the FSM analyzer can't parse,
        // so the BPE fallback specializes on residual forms rather than re-learning
        // morphology the analyzer already covers.
        List<String> oovSentences = new ArrayList<>();
        if (corpus == null || corpus.isEmpty()) {
            fallback.train(oovSentences);
            return;
        }

        for (String sentence : corpus) {
            if (sentence == null || sentence.trim().isEmpty()) {
                continue;
            }
            StringBuilder oov = new StringBuilder();
            for (String word : sentence.trim().split("\\s+")) {
                // Skip punctuation — it passes through the main tokenizer unchanged.
                if (word.isEmpty() || isPunctuation(word)) continue;
                // Lowercase capitalized words before querying the analyzer; the
                // <uppercase> marker is the runtime concern, not training's.
                String processed = Character.isUpperCase(word.charAt(0))
                        ? word.toLowerCase(TURKISH_LOCALE) : word;
                // No FSM parse => OOV: send the *original* (case-preserved) word
                // to the BPE so it sees real-world surface forms.
                if (analyzer.morphologicalAnalysis(processed).size() == 0) {
                    if (oov.length() > 0) oov.append(' ');
                    oov.append(word);
                }
            }
            if (oov.length() > 0) oovSentences.add(oov.toString());
        }
        fallback.train(oovSentences);
    }

    /**
     * Tokenizes a sentence by separating Turkish words and punctuation first.
     * This prevents words like "kitaplar," from being sent to the morphological
     * analyzer together with the comma.
     */
    @Override
    public List<String> encode(String sentence) {
        List<String> tokens = new ArrayList<>();
        if (sentence == null || sentence.trim().isEmpty()) {
            return tokens;
        }

        Matcher matcher = WORD_OR_PUNCTUATION.matcher(sentence);

        while (matcher.find()) {
            tokens.addAll(tokenize(matcher.group()));
        }

        return tokens;
    }

    /**
     * Tokenizes a single word per Algorithm 1.
     * @param word The input word (may include leading uppercase or punctuation).
     * @return Ordered list of tokens: optionally an {@code <uppercase>} marker,
     *         then the space-prefixed root followed by canonical affix tokens,
     *         or BPE pieces if the word is OOV.
     */
    @Override
    public List<String> tokenize(String word) {
        if (word == null || word.isEmpty()) return Collections.emptyList();

        List<String> result = new ArrayList<>();

        // Pass 1: punctuation is a leaf — emit verbatim, no morphology applies.
        if (isPunctuation(word)) {
            result.add(word);
            return result;
        }

        // Pass 2: capitalization is lifted into a dedicated <uppercase> marker
        // (Bayram 2026 convention) so the morphological analyzer sees a
        // case-normalized form and decoders can reconstruct casing later.
        String processed = word;
        if (Character.isUpperCase(word.charAt(0))) {
            result.add(UPPERCASE_TOKEN);
            processed = word.toLowerCase(TURKISH_LOCALE);
        }

        // Pass 3: query the FSM. If no parse exists the word is OOV; route it
        // to the BPE fallback (adding the leading-space root marker to the
        // first piece so byte-level decoders preserve word boundaries).
        FsmParseList parses = analyzer.morphologicalAnalysis(processed);
        if (parses.size() == 0) {
            result.addAll(prefixFirstPiece(fallback.tokenize(processed)));
            return result;
        }

        // Pass 4: among the FSM's competing parses pick the one with the
        // shortest root + most suffixes (paper's heuristic) and emit its
        // root + canonical-affix sequence.
        FsmParse best = selectMostSuffixedParse(parses);
        result.addAll(extractTokens(best, processed));
        return result;
    }

    /**
     * Returns the trainable BPE fallback.
     * @return The inner BPE tokenizer used for OOV words.
     */
    public Tokenizer getFallbackTokenizer() {
        return fallback;
    }

    // ─── Internals ────────────────────────────────────────────────────────

    private FsmParse selectMostSuffixedParse(FsmParseList parses) {
        FsmParse best = parses.getFsmParse(0);
        int bestRootLen = rootLength(best);
        int bestSuffixCount = countSuffixes(best);
        for (int i = 1; i < parses.size(); i++) {
            FsmParse candidate = parses.getFsmParse(i);
            int rootLen = rootLength(candidate);
            int suffixCount = countSuffixes(candidate);
            if (rootLen < bestRootLen
                    || (rootLen == bestRootLen && suffixCount > bestSuffixCount)) {
                best = candidate;
                bestRootLen = rootLen;
                bestSuffixCount = suffixCount;
            }
        }
        return best;
    }

    private int rootLength(FsmParse parse) {
        String t = parse.transitionList();
        if (t == null || t.isEmpty()) return parse.getLastLemma().length();
        int plus = t.indexOf('+');
        return plus < 0 ? t.length() : plus;
    }

    private int countSuffixes(FsmParse parse) {
        String t = parse.transitionList();
        if (t == null || t.isEmpty()) return 0;
        int count = 0;
        for (int i = 0; i < t.length(); i++) if (t.charAt(i) == '+') count++;
        return count;
    }

    private List<String> extractTokens(FsmParse parse, String inputWord) {
        List<String> tokens = new ArrayList<>();
        String transition = parse.transitionList();

        // Degenerate case: parse has no tags, just emit the root.
        if (transition == null || transition.isEmpty()) {
            tokens.add(ROOT_PREFIX + matchRootToInput(parse.getLastLemma(), inputWord));
            return tokens;
        }

        // FSM transitions are grouped by derivational boundaries (^DB+).
        // Each group is a sequence of '+'-separated tags; the first tag of
        // the first group is the root lemma, the rest are inflectional /
        // derivational affixes.
        String[] groups = transition.split("\\^DB\\+");
        boolean rootAdded = false;
        for (String group : groups) {
            String[] parts = group.split("\\+");
            int startIndex = 0;

            // Only the very first group contributes the root. Reconcile the
            // FSM lemma against the input surface (handles lenition like
            // kitap -> kitab) so the emitted root prefixes the original word.
            if (!rootAdded) {
                tokens.add(ROOT_PREFIX + matchRootToInput(parts[0], inputWord));
                rootAdded = true;
                startIndex = 1;
            }

            // Emit each tag in the group: drop POS/default markers, map
            // allomorphic FSM tags to canonical names (A3PL->PL, etc.),
            // and pass through derivational morphemes unchanged.
            for (int i = startIndex; i < parts.length; i++) {
                String tag = parts[i];
                if (tag.isEmpty() || DROP_TAGS.contains(tag)) continue;
                String mapped = AFFIX_MAP.get(tag);
                tokens.add(mapped != null ? mapped : tag);
            }
        }
        return tokens;
    }

    /**
     * If the FSM lemma does not literally prefix the input word, try common
     * Turkish consonant softening patterns (p->b, t->d, k->ğ, ç->c) and emit the
     * softened form whose surface actually prefixes the input. This helps recover
     * surface roots for forms like {@code kitabi} / {@code kitabı}
     * where lemma {@code kitap} appears as surface {@code kitab}.
     * @param lemma     The lemma returned by the FSM analyzer.
     * @param inputWord The original surface form being tokenized.
     * @return Either the lemma itself, or its lenited variant when that variant
     *         prefixes {@code inputWord}.
     */
    private String matchRootToInput(String lemma, String inputWord) {
        if (lemma == null || lemma.isEmpty()) return lemma == null ? "" : lemma;
        if (inputWord == null || inputWord.startsWith(lemma)) return lemma;
        char last = lemma.charAt(lemma.length() - 1);
        char softened;
        switch (last) {
            case 'p': softened = 'b'; break;
            case 't': softened = 'd'; break;
            case 'k': softened = 'ğ'; break;
            case 'ç': softened = 'c'; break;
            default: return lemma;
        }
        String candidate = lemma.substring(0, lemma.length() - 1) + softened;
        return inputWord.startsWith(candidate) ? candidate : lemma;
    }

    private boolean isPunctuation(String word) {
        if (word.length() == 1) return !Character.isLetterOrDigit(word.charAt(0));
        for (int i = 0; i < word.length(); i++) {
            if (Character.isLetterOrDigit(word.charAt(i))) return false;
        }
        return true;
    }

    private List<String> prefixFirstPiece(List<String> pieces) {
        if (pieces.isEmpty()) return pieces;
        List<String> out = new ArrayList<>(pieces.size());
        out.add(ROOT_PREFIX + pieces.get(0));
        for (int i = 1; i < pieces.size(); i++) out.add(pieces.get(i));
        return out;
    }

    // ─── Inner BPE fallback (subword-nmt style, trained on OOV subset) ────

    /**
     * Trainable BPE used as the OOV fallback. Reproduces the algorithmic core
     * of {@code rsennrich/subword-nmt}'s {@code learn_bpe.py} and
     * {@code apply_bpe.py}: end-of-word marker appended to the last character,
     * frequency-based merges with lexicographic tie-breaking, minimum
     * frequency 2, lowest-rank-wins encoding.
     */
    private static final class InnerBPE extends Tokenizer {
        private static final String EOW = "</w>";
        private static final int MIN_FREQUENCY = 2;

        private final List<String[]> mergeRules = new ArrayList<>();
        private final Map<String, Integer> mergeRank = new LinkedHashMap<>();
        private final Set<String> vocabulary = new LinkedHashSet<>();

        InnerBPE(int vocabSize) {
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
            List<WordEntry> sortedVocab = new ArrayList<>(wordCounts.size());
            for (Map.Entry<String, Integer> e : wordCounts.entrySet()) {
                String word = e.getKey();
                List<String> symbols = new ArrayList<>(word.length());
                for (int i = 0; i < word.length() - 1; i++) {
                    symbols.add(String.valueOf(word.charAt(i)));
                }
                symbols.add(word.charAt(word.length() - 1) + EOW);
                sortedVocab.add(new WordEntry(symbols, e.getValue()));
            }
            // Frequency-sort so the initial vocabulary additions are
            // deterministic and the most-common chars are inserted first.
            sortedVocab.sort((a, b) -> Integer.compare(b.freq, a.freq));
            // Seed the vocabulary with every initial symbol (all single chars).
            for (WordEntry we : sortedVocab) vocabulary.addAll(we.symbols);

            // Step 3: BPE merge loop. Each iteration finds the most frequent
            // adjacent symbol pair across the whole corpus, records it as a
            // merge rule, and rewrites every word so future iterations see
            // the merged symbol. Stops when budget exhausted, no pairs left,
            // or the best remaining pair falls below MIN_FREQUENCY.
            int numSymbols = Math.max(0, vocabSize - vocabulary.size());
            for (int iter = 0; iter < numSymbols; iter++) {
                Map<Pair, Integer> stats = pairStatistics(sortedVocab);
                if (stats.isEmpty()) break;
                Pair best = pickBestPair(stats);
                if (best == null || stats.get(best) < MIN_FREQUENCY) break;
                // Record the merge: rule order = its rank (lowest wins at apply time).
                mergeRules.add(new String[]{best.left, best.right});
                mergeRank.put(best.left + " " + best.right, mergeRules.size() - 1);
                vocabulary.add(best.left + best.right);
                // Rewrite every word with the new merge applied.
                for (WordEntry we : sortedVocab) {
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

        private static Map<Pair, Integer> pairStatistics(List<WordEntry> vocab) {
            Map<Pair, Integer> stats = new LinkedHashMap<>();
            for (WordEntry we : vocab) {
                List<String> s = we.symbols;
                for (int i = 0; i < s.size() - 1; i++) {
                    stats.merge(new Pair(s.get(i), s.get(i + 1)), we.freq, Integer::sum);
                }
            }
            return stats;
        }

        private static Pair pickBestPair(Map<Pair, Integer> stats) {
            Pair best = null;
            int bestFreq = Integer.MIN_VALUE;
            for (Map.Entry<Pair, Integer> e : stats.entrySet()) {
                int f = e.getValue();
                Pair p = e.getKey();
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

        private static final class WordEntry {
            List<String> symbols;
            final int freq;
            WordEntry(List<String> symbols, int freq) { this.symbols = symbols; this.freq = freq; }
        }

        private static final class Pair implements Comparable<Pair> {
            final String left, right;
            Pair(String left, String right) { this.left = left; this.right = right; }
            @Override public boolean equals(Object o) {
                if (this == o) return true;
                if (!(o instanceof Pair)) return false;
                Pair p = (Pair) o;
                return left.equals(p.left) && right.equals(p.right);
            }
            @Override public int hashCode() { return 31 * left.hashCode() + right.hashCode(); }
            @Override public int compareTo(Pair o) {
                int c = left.compareTo(o.left);
                return c != 0 ? c : right.compareTo(o.right);
            }
        }
    }

}
