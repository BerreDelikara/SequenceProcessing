package SequenceProcessing.Tokenization;

import MorphologicalAnalysis.FsmMorphologicalAnalyzer;
import MorphologicalAnalysis.FsmParse;
import MorphologicalAnalysis.FsmParseList;

import java.util.*;

/**
 * Hybrid Turkish tokenizer following Algorithm 1 of Ulusoy &amp; Ate&#x015F;
 * "Tokens with Meaning: A Hybrid Tokenization Approach for Turkish"
 * (arXiv:2508.14292). Punctuation passes through as-is, capitalisation is
 * captured by an explicit {@code <uppercase>} marker, words are decomposed
 * into root + morphological tags via the FSM morphological analyzer, and
 * out-of-vocabulary words fall back to BPE.
 */
public class TurkishTokenizer extends Tokenizer {

    /** Marker emitted before the lowercased form of a capitalised word. */
    public static final String UPPERCASE_TOKEN = "<uppercase>";

    /** Leading-space marker prepended to root tokens for lossless word-boundary recovery. */
    public static final String ROOT_PREFIX = " ";

    private static final Locale TURKISH_LOCALE = new Locale("tr", "TR");

    /**
     * Maps FSM grammatical tags to canonical affix identifiers per the paper's
     * normalization layer. Allomorphic variants collapse onto shared IDs
     * (e.g., {@code A3pl} → {@code PL} covers both -lAr surface forms;
     * {@code Abl} → {@code ABL} covers -dAn/-tAn). Unmapped tags (POS labels
     * like {@code Noun}, defaults like {@code Nom}/{@code Pos}/{@code Pnon},
     * and {@code A3sg}) are dropped to keep the affix inventory bounded.
     */
    private static final Map<String, String> AFFIX_MAP = createAffixMap();

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
    private final BPETokenizer fallback;

    /**
     * Constructs a Turkish tokenizer. The BPE fallback is sized with the same
     * target vocabulary size and is used when the morphological analyzer cannot
     * parse a word.
     * @param vocabSize Target vocabulary size for the BPE fallback.
     */
    public TurkishTokenizer(int vocabSize) {
        super(vocabSize);
        this.analyzer = new FsmMorphologicalAnalyzer();
        this.fallback = new BPETokenizer(vocabSize);
    }

    /**
     * Trains the BPE fallback on a masked version of the corpus. Words that
     * the morphological analyzer can already parse are removed before
     * training, so the fallback vocabulary focuses on residual stems and
     * out-of-dictionary forms (per Bayram et al. 2026, Section 3).
     * @param corpus List of whitespace-separated sentences.
     */
    @Override
    public void train(List<String> corpus) {
        List<String> maskedCorpus = new ArrayList<>();
        for (String sentence : corpus) {
            StringBuilder masked = new StringBuilder();
            for (String word : sentence.trim().split("\\s+")) {
                if (word.isEmpty() || isPunctuation(word)) continue;
                String processed = Character.isUpperCase(word.charAt(0))
                        ? word.toLowerCase(TURKISH_LOCALE)
                        : word;
                if (analyzer.morphologicalAnalysis(processed).size() == 0) {
                    if (masked.length() > 0) masked.append(' ');
                    masked.append(word);
                }
            }
            if (masked.length() > 0) {
                maskedCorpus.add(masked.toString());
            }
        }
        fallback.train(maskedCorpus);
    }

    /**
     * Tokenizes a single word per Algorithm 1: punctuation is returned as-is;
     * a leading uppercase letter triggers an {@link #UPPERCASE_TOKEN} followed
     * by the lowercased form; the word is then morphologically analysed and
     * emitted as {@code [root, tag1, tag2, ...]}, falling back to BPE when no
     * parse is available.
     * @param word The word to tokenize.
     * @return Ordered list of tokens.
     */
    @Override
    public List<String> tokenize(String word) {
        if (word.isEmpty()) {
            return Collections.emptyList();
        }

        List<String> result = new ArrayList<>();

        // Step 1: handle punctuation — emit as-is
        if (isPunctuation(word)) {
            result.add(word);
            return result;
        }

        // Step 2: handle capitalization — emit <uppercase> token, then lowercase
        String processedWord = word;
        if (Character.isUpperCase(word.charAt(0))) {
            result.add(UPPERCASE_TOKEN);
            processedWord = word.toLowerCase(TURKISH_LOCALE);
        }

        // Step 3: morphological analysis
        FsmParseList parseList = analyzer.morphologicalAnalysis(processedWord);
        if (parseList.size() == 0) {
            // Step 4: BPE fallback for OOV words; first piece carries the root prefix
            result.addAll(prefixFirstPiece(fallback.tokenize(processedWord)));
            return result;
        }

        // Paper's "longest chain of valid suffixes": prefer the parse with the most
        // suffix transitions, breaking ties by shorter root surface form.
        FsmParse best = selectMostSuffixedParse(parseList);
        result.addAll(extractTokens(best, processedWord));
        return result;
    }

    private FsmParse selectMostSuffixedParse(FsmParseList parseList) {
        FsmParse best = parseList.getFsmParse(0);
        int bestRootLen = rootLength(best);
        int bestSuffixCount = countSuffixes(best);
        for (int i = 1; i < parseList.size(); i++) {
            FsmParse candidate = parseList.getFsmParse(i);
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
        String transition = parse.transitionList();
        if (transition == null || transition.isEmpty()) {
            return parse.getLastLemma().length();
        }
        int plus = transition.indexOf('+');
        return plus < 0 ? transition.length() : plus;
    }

    private int countSuffixes(FsmParse parse) {
        String transition = parse.transitionList();
        if (transition == null || transition.isEmpty()) return 0;
        int plusCount = 0;
        for (int i = 0; i < transition.length(); i++) {
            if (transition.charAt(i) == '+') plusCount++;
        }
        return plusCount;
    }

    private List<String> extractTokens(FsmParse parse, String inputWord) {
        List<String> tokens = new ArrayList<>();
        String transition = parse.transitionList();
        if (transition == null || transition.isEmpty()) {
            tokens.add(ROOT_PREFIX + matchRootToInput(parse.getLastLemma(), inputWord));
            return tokens;
        }

        // transitionList format: "root+Tag1+Tag2+...^DB+Tag3+..."
        // Split on derivational boundary marker first
        String[] groups = transition.split("\\^DB\\+");

        boolean rootAdded = false;
        for (String group : groups) {
            String[] parts = group.split("\\+");
            int startIndex = 0;

            if (!rootAdded) {
                tokens.add(ROOT_PREFIX + matchRootToInput(parts[0], inputWord));
                rootAdded = true;
                startIndex = 1;
            }

            for (int i = startIndex; i < parts.length; i++) {
                String mapped = AFFIX_MAP.get(parts[i]);
                if (mapped != null) {
                    tokens.add(mapped);
                }
            }
        }
        return tokens;
    }

    /**
     * If the canonical lemma does not literally prefix the input word, try the
     * standard Turkish lenition softenings (p→b, t→d, k→ğ, ç→c) and emit the
     * softened form whose surface form actually prefixes the input. This makes
     * encode → decode lossless for forms like {@code araba} (parsed as
     * {@code arap+Dat}, surface starts with "arab").
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

    private List<String> prefixFirstPiece(List<String> pieces) {
        if (pieces.isEmpty()) return pieces;
        List<String> prefixed = new ArrayList<>(pieces.size());
        prefixed.add(ROOT_PREFIX + pieces.get(0));
        prefixed.addAll(pieces.subList(1, pieces.size()));
        return prefixed;
    }

    private boolean isPunctuation(String word) {
        if (word.length() == 1) {
            char c = word.charAt(0);
            return !Character.isLetterOrDigit(c);
        }
        for (char c : word.toCharArray()) {
            if (Character.isLetterOrDigit(c)) {
                return false;
            }
        }
        return true;
    }

    /**
     * Returns the underlying BPE fallback tokenizer used for OOV words.
     * @return The trained BPE fallback.
     */
    public BPETokenizer getFallbackTokenizer() {
        return fallback;
    }

    // ───────────────────────────────────────────────────────────────────────
    // Decoder: reconstructs surface text from a token list emitted by this
    // tokenizer. Applies Turkish phonology (vowel harmony, consonant
    // assimilation) so canonical affix tags like {@code PL} → "lar"/"ler"
    // become the correct surface form for the preceding root.
    // ───────────────────────────────────────────────────────────────────────

    private static final String ALL_VOWELS = "aeıioöuüâ";
    private static final String INCE_VOWELS = "eiöü";
    private static final String AI_VOWELS = "aıâ";
    private static final String EI_VOWELS = "ei";
    private static final String OU_VOWELS = "ou";
    private static final String HARD_CONSONANTS = "fstkçşhp";

    private enum HarmonyRule {
        NONE,
        TWO_BACK,           // [back, front]
        TWO_BACK_BUFFER,    // [back, front, y-back, y-front]
        FOUR_WAY,           // [back-unr, front-unr, back-r, front-r]
        FOUR_WAY_BUFFER,    // 8 variants: first 4 after consonant, last 4 after vowel
        DA_DE,              // [da, de, ta, te]
        DI_DU               // [dı, di, du, dü, tı, ti, tu, tü]
    }

    private static final class AffixSpec {
        final String[] variants;
        final HarmonyRule rule;
        AffixSpec(String[] variants, HarmonyRule rule) {
            this.variants = variants;
            this.rule = rule;
        }
        String select(String prev) {
            switch (rule) {
                case NONE:
                    return variants[0];
                case TWO_BACK:
                    return endsWithInce(prev) ? variants[1] : variants[0];
                case TWO_BACK_BUFFER: {
                    int base = endsWithInce(prev) ? 1 : 0;
                    return endsWithVowel(prev) ? variants[base + 2] : variants[base];
                }
                case FOUR_WAY:
                    return variants[vowelSuffixIndex(prev)];
                case FOUR_WAY_BUFFER: {
                    int base = vowelSuffixIndex(prev);
                    return endsWithVowel(prev) ? variants[base + 4] : variants[base];
                }
                case DA_DE: {
                    if (endsWithHardConsonant(prev)) {
                        return endsWithInce(prev) ? variants[3] : variants[2];
                    }
                    return endsWithInce(prev) ? variants[1] : variants[0];
                }
                case DI_DU: {
                    int base = vowelSuffixIndex(prev);
                    return endsWithHardConsonant(prev) ? variants[base + 4] : variants[base];
                }
                default:
                    return variants[0];
            }
        }
    }

    private static final Map<String, AffixSpec> AFFIX_SURFACE_FORMS = createSurfaceForms();

    private static Map<String, AffixSpec> createSurfaceForms() {
        Map<String, AffixSpec> m = new LinkedHashMap<>();

        m.put("PL", new AffixSpec(new String[]{"lar", "ler"}, HarmonyRule.TWO_BACK));

        m.put("ACC", new AffixSpec(new String[]{"ı", "i", "u", "ü", "yı", "yi", "yu", "yü"}, HarmonyRule.FOUR_WAY_BUFFER));
        m.put("DAT", new AffixSpec(new String[]{"a", "e", "ya", "ye"}, HarmonyRule.TWO_BACK_BUFFER));
        m.put("LOC", new AffixSpec(new String[]{"da", "de", "ta", "te"}, HarmonyRule.DA_DE));
        m.put("ABL", new AffixSpec(new String[]{"dan", "den", "tan", "ten"}, HarmonyRule.DA_DE));
        m.put("GEN", new AffixSpec(new String[]{"ın", "in", "un", "ün", "nın", "nin", "nun", "nün"}, HarmonyRule.FOUR_WAY_BUFFER));
        m.put("INS", new AffixSpec(new String[]{"la", "le", "yla", "yle"}, HarmonyRule.TWO_BACK_BUFFER));
        m.put("EQU", new AffixSpec(new String[]{"ca", "ce", "ça", "çe"}, HarmonyRule.DA_DE));

        m.put("POSS_1SG", new AffixSpec(new String[]{"ım", "im", "um", "üm", "m", "m", "m", "m"}, HarmonyRule.FOUR_WAY_BUFFER));
        m.put("POSS_2SG", new AffixSpec(new String[]{"ın", "in", "un", "ün", "n", "n", "n", "n"}, HarmonyRule.FOUR_WAY_BUFFER));
        m.put("POSS_3SG", new AffixSpec(new String[]{"ı", "i", "u", "ü", "sı", "si", "su", "sü"}, HarmonyRule.FOUR_WAY_BUFFER));
        m.put("POSS_1PL", new AffixSpec(new String[]{"ımız", "imiz", "umuz", "ümüz", "mız", "miz", "muz", "müz"}, HarmonyRule.FOUR_WAY_BUFFER));
        m.put("POSS_2PL", new AffixSpec(new String[]{"ınız", "iniz", "unuz", "ünüz", "nız", "niz", "nuz", "nüz"}, HarmonyRule.FOUR_WAY_BUFFER));
        m.put("POSS_3PL", new AffixSpec(new String[]{"ları", "leri"}, HarmonyRule.TWO_BACK));

        m.put("1SG", new AffixSpec(new String[]{"ım", "im", "um", "üm"}, HarmonyRule.FOUR_WAY));
        m.put("2SG", new AffixSpec(new String[]{"sın", "sin", "sun", "sün"}, HarmonyRule.FOUR_WAY));
        m.put("1PL", new AffixSpec(new String[]{"ız", "iz", "uz", "üz"}, HarmonyRule.FOUR_WAY));
        m.put("2PL", new AffixSpec(new String[]{"sınız", "siniz", "sunuz", "sünüz"}, HarmonyRule.FOUR_WAY));

        m.put("PAST", new AffixSpec(new String[]{"dı", "di", "du", "dü", "tı", "ti", "tu", "tü"}, HarmonyRule.DI_DU));
        m.put("NARR", new AffixSpec(new String[]{"mış", "miş", "muş", "müş"}, HarmonyRule.FOUR_WAY));
        m.put("FUT", new AffixSpec(new String[]{"acak", "ecek", "yacak", "yecek"}, HarmonyRule.TWO_BACK_BUFFER));
        m.put("COND", new AffixSpec(new String[]{"sa", "se"}, HarmonyRule.TWO_BACK));
        m.put("NECES", new AffixSpec(new String[]{"malı", "meli"}, HarmonyRule.TWO_BACK));
        m.put("PROG", new AffixSpec(new String[]{"ıyor", "iyor", "uyor", "üyor"}, HarmonyRule.FOUR_WAY));
        m.put("AOR", new AffixSpec(new String[]{"ır", "ir", "ur", "ür"}, HarmonyRule.FOUR_WAY));

        m.put("NEG", new AffixSpec(new String[]{"ma", "me"}, HarmonyRule.TWO_BACK));
        m.put("ABLE", new AffixSpec(new String[]{"abil", "ebil", "yabil", "yebil"}, HarmonyRule.TWO_BACK_BUFFER));

        m.put("OPT", new AffixSpec(new String[]{"a", "e"}, HarmonyRule.TWO_BACK));
        m.put("DESR", new AffixSpec(new String[]{"sa", "se"}, HarmonyRule.TWO_BACK));
        m.put("IMP", new AffixSpec(new String[]{""}, HarmonyRule.NONE));
        m.put("PRES", new AffixSpec(new String[]{""}, HarmonyRule.NONE));

        m.put("PASS", new AffixSpec(new String[]{"ıl", "il", "ul", "ül"}, HarmonyRule.FOUR_WAY));
        m.put("CAUS", new AffixSpec(new String[]{"t"}, HarmonyRule.NONE));
        m.put("RECIP", new AffixSpec(new String[]{"ış", "iş", "uş", "üş"}, HarmonyRule.FOUR_WAY));
        m.put("REFLEX", new AffixSpec(new String[]{"ın", "in", "un", "ün"}, HarmonyRule.FOUR_WAY));

        m.put("INF", new AffixSpec(new String[]{"mak", "mek"}, HarmonyRule.TWO_BACK));
        m.put("PASTPART", new AffixSpec(new String[]{"dık", "dik", "duk", "dük", "tık", "tik", "tuk", "tük"}, HarmonyRule.DI_DU));
        m.put("PRESPART", new AffixSpec(new String[]{"an", "en"}, HarmonyRule.TWO_BACK));
        m.put("FUTPART", new AffixSpec(new String[]{"acak", "ecek", "yacak", "yecek"}, HarmonyRule.TWO_BACK_BUFFER));
        m.put("AORPART", new AffixSpec(new String[]{"ır", "ir", "ur", "ür"}, HarmonyRule.FOUR_WAY));

        return Collections.unmodifiableMap(m);
    }

    /**
     * Reconstructs surface text from a token list produced by this tokenizer.
     * Capitalisation, leading-space roots, canonical affixes, and punctuation
     * are all handled; phonologically-conditioned allomorphs are resolved via
     * Turkish vowel harmony and consonant assimilation rules.
     * @param tokens List of tokens previously emitted by {@link #tokenize} or {@link #encode}.
     * @return Reconstructed Turkish text.
     */
    public String decode(List<String> tokens) {
        StringBuilder out = new StringBuilder();
        boolean capitalizeNext = false;

        for (String tok : tokens) {
            if (UPPERCASE_TOKEN.equals(tok)) {
                capitalizeNext = true;
                continue;
            }

            if (tok.startsWith(ROOT_PREFIX) && tok.length() > ROOT_PREFIX.length()) {
                String root = tok.substring(ROOT_PREFIX.length());
                if (capitalizeNext) {
                    root = trCapitalize(root);
                    capitalizeNext = false;
                }
                if (out.length() > 0) out.append(' ');
                out.append(root);
                continue;
            }

            AffixSpec spec = AFFIX_SURFACE_FORMS.get(tok);
            if (spec != null) {
                String context = currentWordContext(out);
                // POSS_3PL after PL: surface merges to just the possessive vowel
                // (araba+lAr+I, not araba+lAr+lArI).
                if ("POSS_3PL".equals(tok)
                        && (context.endsWith("lar") || context.endsWith("ler"))) {
                    out.append(endsWithInce(context) ? "i" : "ı");
                    continue;
                }
                out.append(spec.select(context));
                continue;
            }

            // Unknown tag or punctuation: emit as-is.
            out.append(tok);
        }
        return out.toString();
    }

    private static String currentWordContext(StringBuilder out) {
        int i = out.length() - 1;
        while (i >= 0 && out.charAt(i) != ' ') i--;
        return out.substring(i + 1);
    }

    private static String trCapitalize(String word) {
        if (word.isEmpty()) return word;
        if (word.startsWith("i")) return "İ" + word.substring(1);
        return word.substring(0, 1).toUpperCase(TURKISH_LOCALE) + word.substring(1);
    }

    // ─── Turkish phonology predicates ──────────────────────────────────────

    private static boolean endsWithVowel(String word) {
        return !word.isEmpty() && ALL_VOWELS.indexOf(word.charAt(word.length() - 1)) >= 0;
    }

    private static boolean endsWithHardConsonant(String word) {
        return !word.isEmpty() && HARD_CONSONANTS.indexOf(word.charAt(word.length() - 1)) >= 0;
    }

    private static boolean endsWithAny(String word, String charset) {
        for (int idx = word.length() - 1; idx >= 0; idx--) {
            char c = word.charAt(idx);
            if (charset.indexOf(c) >= 0) return true;
            if (ALL_VOWELS.indexOf(c) >= 0) return false;
        }
        return false;
    }

    private static boolean endsWithInce(String word) {
        return endsWithAny(word, INCE_VOWELS);
    }

    private static int vowelSuffixIndex(String word) {
        if (endsWithAny(word, AI_VOWELS)) return 0;
        if (endsWithAny(word, EI_VOWELS)) return 1;
        if (endsWithAny(word, OU_VOWELS)) return 2;
        return 3;
    }
}
