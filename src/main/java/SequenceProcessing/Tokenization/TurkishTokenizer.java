package SequenceProcessing.Tokenization;

import MorphologicalAnalysis.FsmMorphologicalAnalyzer;
import MorphologicalAnalysis.FsmParse;
import MorphologicalAnalysis.FsmParseList;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
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
    private final TurkishBPEFallback fallback;

    /**
     * @param vocabSize Target vocabulary size for the trainable BPE fallback.
     *                  The morphological side does not depend on this.
     */
    public TurkishTokenizer(int vocabSize) {
        super(vocabSize);
        this.analyzer = new FsmMorphologicalAnalyzer();
        this.fallback = new TurkishBPEFallback(vocabSize);
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
        String surfaceRoot = null; //Ünsüz benzeşmesi checker
        boolean hasMeaningfulSuffixAfterRoot = false;
        for (String group : groups) {
            String[] parts = group.split("\\+");
            int startIndex = 0;

            // Only the very first group contributes the root. Reconcile the
            // FSM lemma against the input surface (handles lenition like
            // kitap -> kitab) so the emitted root prefixes the original word.
            if (!rootAdded) {
                surfaceRoot = matchRootToInput(parts[0], inputWord);
                tokens.add(ROOT_PREFIX + surfaceRoot);
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
                String token = mapped != null ? mapped : tag;

                // Added: connect ünsüz benzeşmesi to suffix-side changes.
                // Example: kitap + LOC => kitapta, so LOC is marked as hardened.
                if (surfaceRoot != null && hasConsonantAssimilation(surfaceRoot, token)&& !hasMeaningfulSuffixAfterRoot) {
                    token = token + "_HARD";
                }


                tokens.add(token);
                hasMeaningfulSuffixAfterRoot = true;
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
        switch (last) { // ünsüz yumuşaması
            case 'p': softened = 'b'; break;
            case 't': softened = 'd'; break;
            case 'k': softened = 'ğ'; break;
            case 'ç': softened = 'c'; break;
            default: return lemma;
        }
        String candidate = lemma.substring(0, lemma.length() - 1) + softened;
        return inputWord.startsWith(candidate) ? candidate : lemma;
    }

     /**
     * Checks whether a suffix tag is affected by Turkish consonant assimilation
     * after a hard consonant.
     *
     * Ünsüz benzeşmesi:
     * If the root ends with f, s, t, k, ç, ş, h, p,
     * suffixes that normally begin with c, d, g surface with ç, t, k.
     *
     * Examples:
     * kitap + LOC -> kitapta
     * ağaç + ABL -> ağaçtan
     * bak + PAST -> baktı
     */
    private boolean hasConsonantAssimilation(String root, String tag) {
        if (root == null || root.isEmpty() || tag == null || tag.isEmpty()) {
            return false;
        }

        char lastRootChar = root.charAt(root.length() - 1);

        if (!isHardConsonant(lastRootChar)) {
            return false;
        }

        return tag.equals("LOC")      // -da/-de -> -ta/-te
                || tag.equals("ABL")  // -dan/-den -> -tan/-ten
                || tag.equals("PAST") // -dı/-di -> -tı/-ti
                || tag.equals("AGT"); // -cı/-ci -> -çı/-çi
    }

    /**
     * Turkish hard consonants: f, s, t, k, ç, ş, h, p.
     */
    private boolean isHardConsonant(char c) {
        return c == 'f' || c == 's' || c == 't' || c == 'k'
                || c == 'ç' || c == 'ş' || c == 'h' || c == 'p';
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


}
