package SequenceProcessing.Tokenization;

import MorphologicalAnalysis.FsmMorphologicalAnalyzer;
import MorphologicalAnalysis.FsmParse;
import MorphologicalAnalysis.FsmParseList;

import java.util.*;

public class TurkishTokenizer extends Tokenizer {

    public static final String UPPERCASE_TOKEN = "<uppercase>";

    private static final Locale TURKISH_LOCALE = new Locale("tr", "TR");

    private static final Set<String> SKIP_TAGS = new HashSet<>(Arrays.asList(
            "Noun", "Verb", "Adj", "Adv", "Pron", "Det", "Conj", "Postp",
            "Ques", "Interj", "Num", "Dup", "Code", "Punc",
            "A3sg", "Pnon", "Nom", "Pos"
    ));

    private final FsmMorphologicalAnalyzer analyzer;
    private final BPETokenizer fallback;

    public TurkishTokenizer(int vocabSize) {
        super(vocabSize);
        this.analyzer = new FsmMorphologicalAnalyzer();
        this.fallback = new BPETokenizer(vocabSize);
    }

    @Override
    public void train(List<String> corpus) {
        fallback.train(corpus);
    }

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
            // Step 4: BPE fallback for OOV words
            result.addAll(fallback.tokenize(processedWord));
            return result;
        }

        // Greedy longest-root-word match (closest to paper's longest-prefix strategy)
        FsmParse best = parseList.getParseWithLongestRootWord();
        result.addAll(extractTokens(best));
        return result;
    }

    private List<String> extractTokens(FsmParse parse) {
        List<String> tokens = new ArrayList<>();
        String transition = parse.transitionList();
        if (transition == null || transition.isEmpty()) {
            tokens.add(parse.getLastLemma());
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
                // First token in first group is the root word
                tokens.add(parts[0]);
                rootAdded = true;
                startIndex = 1;
            }

            for (int i = startIndex; i < parts.length; i++) {
                String tag = parts[i];
                if (!SKIP_TAGS.contains(tag) && !tag.isEmpty()) {
                    tokens.add(tag);
                }
            }
        }
        return tokens;
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

    public BPETokenizer getFallbackTokenizer() {
        return fallback;
    }
}
