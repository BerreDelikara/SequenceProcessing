package SequenceProcessing.Tokenization;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Abstract base for GPT-2-family byte-level BPE tokenizers. Implements the
 * standard byte-to-char bijection of Radford et al. 2019, the BPE encoding
 * loop from {@code apply_bpe.py}, and resource loading from {@code vocab.json}
 * + {@code merges.txt}. Subclasses provide the pre-tokenizer regex and the
 * resource paths.
 *
 * <p>The byte mapping converts each of the 256 byte values to a unique BMP
 * character (printable chars map to themselves; the 68 non-printable bytes
 * are remapped to {@code U+0100}..{@code U+0143}). This lets BPE merge rules
 * stay purely textual while losslessly representing arbitrary UTF-8 input.
 */
public abstract class ByteLevelBPETokenizer extends Tokenizer {

    /** Byte value -> mapped BMP character. */
    public static final char[] BYTE_TO_CHAR = new char[256];
    /** Mapped BMP character -> byte value. */
    public static final Map<Character, Integer> CHAR_TO_BYTE;

    static {
        // Build the GPT-2 byte-to-unicode bijection.
        int[] codepoints = new int[256];
        boolean[] selfMapped = new boolean[256];
        // Three ranges of printable bytes that already display as themselves:
        // ASCII printable, Latin-1 supplement above NBSP (minus a few control-
        // adjacent codepoints), and the rest of Latin-1.
        for (int b = '!'; b <= '~'; b++) selfMapped[b] = true;
        for (int b = 0xA1; b <= 0xAC; b++) selfMapped[b] = true;
        for (int b = 0xAE; b <= 0xFF; b++) selfMapped[b] = true;
        // Self-mapped bytes use their own value as the codepoint.
        for (int b = 0; b < 256; b++) {
            if (selfMapped[b]) codepoints[b] = b;
        }
        // The remaining 68 non-printable bytes get reassigned to the
        // contiguous block U+0100..U+0143 so every byte ends up mapping to
        // a unique, visible BMP character.
        int n = 0;
        for (int b = 0; b < 256; b++) {
            if (!selfMapped[b]) {
                codepoints[b] = 256 + n;
                n++;
            }
        }
        // Materialize both directions of the mapping.
        Map<Character, Integer> inverse = new HashMap<>(256);
        for (int b = 0; b < 256; b++) {
            BYTE_TO_CHAR[b] = (char) codepoints[b];
            inverse.put(BYTE_TO_CHAR[b], b);
        }
        CHAR_TO_BYTE = Collections.unmodifiableMap(inverse);
    }

    protected final Map<String, Integer> vocab;
    protected final Map<Integer, String> idToToken;
    protected final Map<String, Integer> mergeRank;
    protected final int unknownTokenId;

    private final Map<String, List<String>> bpeCache = new ConcurrentHashMap<>();

    /**
     * Constructs the tokenizer by loading {@code vocab.json} and
     * {@code merges.txt} from the classpath. Subclass {@link #vocabResourcePath()}
     * and {@link #mergesResourcePath()} must return constant paths.
     */
    protected ByteLevelBPETokenizer() {
        super(0);
        Map<String, Integer> v = loadVocab(vocabResourcePath());
        this.vocab = Collections.unmodifiableMap(v);
        Map<Integer, String> inv = new HashMap<>(v.size());
        for (Map.Entry<String, Integer> e : v.entrySet()) inv.put(e.getValue(), e.getKey());
        this.idToToken = Collections.unmodifiableMap(inv);
        this.mergeRank = buildMergeRank(loadMerges(mergesResourcePath()));
        this.unknownTokenId = resolveUnknownTokenId(v);
    }

    /**
     * Classpath resource path for the vocab.json (e.g. {@code "/cosmos-gpt2/vocab.json"}).
     * @return Absolute classpath resource path to the model's vocab.json.
     */
    protected abstract String vocabResourcePath();

    /**
     * Classpath resource path for the merges.txt.
     * @return Absolute classpath resource path to the model's merges.txt.
     */
    protected abstract String mergesResourcePath();

    /**
     * Pre-tokenizer regex applied before BPE. Must use the same semantics as the HF tokenizer.
     * @return Compiled regex used to split input into pre-BPE chunks.
     */
    protected abstract Pattern preTokenizerPattern();

    /**
     * Returns the integer ID of this tokenizer's unknown-token, or {@code -1}
     * if there is none and OOV bytes should be silently dropped.
     * @param vocab Loaded token-to-id vocabulary, used to look up the model's
     *              unknown-token symbol.
     * @return The unknown-token ID, or {@code -1} when no UNK is defined.
     */
    protected abstract int resolveUnknownTokenId(Map<String, Integer> vocab);

    /**
     * No-op: byte-level BPE tokenizers ship with pre-trained vocab and merges.
     * @param corpus Ignored.
     */
    @Override
    public final void train(List<String> corpus) {
        // Pre-trained; no-op.
    }

    /**
     * Tokenizes a single pre-tokenized chunk through byte mapping + BPE.
     * @param chunk One pre-tokenized substring (output of the regex split).
     * @return BPE token pieces produced from {@code chunk} after byte mapping.
     */
    @Override
    public List<String> tokenize(String chunk) {
        if (chunk == null || chunk.isEmpty()) return Collections.emptyList();
        String mapped = byteEncode(chunk);
        return bpe(mapped);
    }

    /**
     * Encodes a sentence to BPE tokens, applying the pre-tokenizer regex first.
     * Overrides the base class which splits on whitespace (that would strip
     * the leading-space cues byte-level BPE depends on).
     * @param sentence Full input sentence.
     * @return Concatenated BPE tokens across every regex-matched chunk.
     */
    @Override
    public List<String> encode(String sentence) {
        List<String> out = new ArrayList<>();
        if (sentence == null || sentence.isEmpty()) return out;
        Matcher m = preTokenizerPattern().matcher(sentence);
        while (m.find()) {
            out.addAll(tokenize(m.group()));
        }
        return out;
    }

    /**
     * Encodes a sentence to integer token IDs.
     * @param sentence Full input sentence.
     * @return Token IDs for each piece, substituting the UNK ID for OOV pieces
     *         when one is defined (otherwise OOV pieces are dropped).
     */
    public List<Integer> encodeIds(String sentence) {
        List<String> tokens = encode(sentence);
        List<Integer> ids = new ArrayList<>(tokens.size());
        for (String t : tokens) {
            Integer id = vocab.get(t);
            if (id != null) {
                ids.add(id);
            } else if (unknownTokenId >= 0) {
                ids.add(unknownTokenId);
            }
        }
        return ids;
    }

    /**
     * Decodes a token-ID sequence back to a UTF-8 string.
     * @param ids Token IDs produced by {@link #encodeIds(String)}.
     * @return The reconstructed string. Unknown IDs are skipped.
     */
    public String decode(List<Integer> ids) {
        StringBuilder mapped = new StringBuilder();
        for (Integer id : ids) {
            String tok = idToToken.get(id);
            if (tok != null) mapped.append(tok);
        }
        return byteDecode(mapped.toString());
    }

    /**
     * Number of entries in the vocabulary.
     * @return Vocabulary size as loaded from vocab.json.
     */
    public int getVocabularySize() {
        return vocab.size();
    }

    /**
     * Unmodifiable view of the underlying vocabulary (token -> id).
     * @return Read-only token-to-id map preserving insertion order.
     */
    public Map<String, Integer> getVocabulary() {
        return vocab;
    }

    // ─── Byte-level mapping ────────────────────────────────────────────────

    /**
     * UTF-8 encodes the chunk, then maps each byte through {@link #BYTE_TO_CHAR}.
     * @param chunk Pre-tokenized substring to byte-map.
     * @return A string where each char represents one source UTF-8 byte under
     *         the GPT-2 byte-to-unicode mapping.
     */
    protected static String byteEncode(String chunk) {
        byte[] bytes = chunk.getBytes(StandardCharsets.UTF_8);
        StringBuilder out = new StringBuilder(bytes.length);
        for (byte b : bytes) out.append(BYTE_TO_CHAR[b & 0xFF]);
        return out.toString();
    }

    /**
     * Reverses {@link #byteEncode}: each char back to its byte, then UTF-8 decode.
     * @param mapped Byte-mapped string produced by {@link #byteEncode(String)}.
     * @return The original UTF-8 string.
     */
    protected static String byteDecode(String mapped) {
        byte[] bytes = new byte[mapped.length()];
        int n = 0;
        for (int i = 0; i < mapped.length(); i++) {
            Integer b = CHAR_TO_BYTE.get(mapped.charAt(i));
            if (b != null) bytes[n++] = (byte) b.intValue();
        }
        return new String(bytes, 0, n, StandardCharsets.UTF_8);
    }

    // ─── BPE merge loop (port of apply_bpe.py) ─────────────────────────────

    private List<String> bpe(String word) {
        // Cache hit: this byte-mapped chunk has been encoded before.
        // Hand back a defensive copy so callers can mutate freely.
        List<String> cached = bpeCache.get(word);
        if (cached != null) return new ArrayList<>(cached);

        // Trivial-length fast paths — nothing to merge.
        if (word.length() <= 1) {
            List<String> single = Collections.singletonList(word);
            bpeCache.put(word, single);
            return new ArrayList<>(single);
        }

        // Initial state: one symbol per character of the byte-mapped chunk.
        List<String> symbols = new ArrayList<>(word.length());
        for (int i = 0; i < word.length(); i++) {
            symbols.add(String.valueOf(word.charAt(i)));
        }

        // Apply merges lowest-rank-first until no adjacent pair matches a
        // rule. This is the exact greedy strategy of HF's apply_bpe.py.
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

        // Memoize before returning; store an immutable snapshot, hand out
        // a copy so the cached entry stays clean.
        bpeCache.put(word, new ArrayList<>(symbols));
        return symbols;
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

    // ─── Resource loading ─────────────────────────────────────────────────

    private static Map<String, Integer> loadVocab(String resourcePath) {
        try (InputStream in = ByteLevelBPETokenizer.class.getResourceAsStream(resourcePath)) {
            if (in == null) {
                throw new IllegalStateException("Missing resource: " + resourcePath);
            }
            return JsonObjectParser.parse(slurp(in));
        } catch (IOException e) {
            throw new RuntimeException("Failed to load " + resourcePath, e);
        }
    }

    private static List<String[]> loadMerges(String resourcePath) {
        try (InputStream in = ByteLevelBPETokenizer.class.getResourceAsStream(resourcePath)) {
            if (in == null) {
                throw new IllegalStateException("Missing resource: " + resourcePath);
            }
            List<String[]> rules = new ArrayList<>();
            try (BufferedReader r = new BufferedReader(
                    new InputStreamReader(in, StandardCharsets.UTF_8))) {
                String line;
                while ((line = r.readLine()) != null) {
                    if (line.isEmpty() || line.startsWith("#")) continue;
                    int sp = line.indexOf(' ');
                    if (sp < 0) continue;
                    rules.add(new String[]{
                            line.substring(0, sp),
                            line.substring(sp + 1)
                    });
                }
            }
            return rules;
        } catch (IOException e) {
            throw new RuntimeException("Failed to load " + resourcePath, e);
        }
    }

    private static Map<String, Integer> buildMergeRank(List<String[]> merges) {
        Map<String, Integer> rank = new LinkedHashMap<>(merges.size() * 2);
        for (int i = 0; i < merges.size(); i++) {
            rank.put(merges.get(i)[0] + " " + merges.get(i)[1], i);
        }
        return Collections.unmodifiableMap(rank);
    }

    private static String slurp(InputStream in) throws IOException {
        StringBuilder sb = new StringBuilder();
        try (BufferedReader r = new BufferedReader(
                new InputStreamReader(in, StandardCharsets.UTF_8))) {
            char[] buf = new char[8192];
            int n;
            while ((n = r.read(buf)) >= 0) sb.append(buf, 0, n);
        }
        return sb.toString();
    }
}
