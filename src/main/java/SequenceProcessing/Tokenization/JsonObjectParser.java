package SequenceProcessing.Tokenization;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Minimal parser for flat JSON objects of the form {@code {"key": int, ...}}.
 * Supports standard string escapes and integer values (including negative).
 * Does not support nested objects, arrays, floats, or unicode surrogate pairs
 * beyond a single \\uXXXX code unit.
 */
final class JsonObjectParser {

    private JsonObjectParser() {}

    /**
     * Parses a flat JSON object string into a {@link LinkedHashMap} preserving
     * insertion order.
     * @param text JSON text, expected to be a single object with string keys
     *             and integer values.
     * @return Map of keys to integer values.
     * @throws RuntimeException if the input is not well-formed JSON of the
     *                          supported subset.
     */
    static Map<String, Integer> parse(String text) {
        State s = new State(text);
        // Consume the opening brace of the top-level object.
        skipWs(s);
        expect(s, '{');
        Map<String, Integer> result = new LinkedHashMap<>();
        // Empty-object fast path: '{}' returns an empty map.
        skipWs(s);
        if (peek(s) == '}') { s.i++; return result; }

        // Read "key": value pairs separated by commas until we hit '}'.
        // LinkedHashMap preserves vocab.json's original ordering, which
        // matters because token IDs are positional.
        while (true) {
            skipWs(s);
            String key = readString(s);
            skipWs(s);
            expect(s, ':');
            skipWs(s);
            int value = readInt(s);
            result.put(key, value);
            // After each pair: ',' means another pair follows, '}' ends
            // the object, anything else is malformed.
            skipWs(s);
            char c = peek(s);
            if (c == ',') { s.i++; continue; }
            if (c == '}') { s.i++; break; }
            throw err(s, "expected ',' or '}', got '" + c + "'");
        }
        return result;
    }

    private static String readString(State s) {
        // Consume the opening quote.
        if (peek(s) != '"') throw err(s, "expected '\"' to start string");
        s.i++;
        StringBuilder sb = new StringBuilder();
        // Walk char-by-char until the matching close quote. Backslash
        // introduces an escape sequence handled below; everything else
        // is appended verbatim.
        while (s.i < s.text.length()) {
            char c = s.text.charAt(s.i);
            if (c == '"') { s.i++; return sb.toString(); }
            if (c == '\\') {
                s.i++;
                if (s.i >= s.text.length()) throw err(s, "unterminated escape");
                char esc = s.text.charAt(s.i++);
                // Standard JSON escapes plus the 4-hex-digit unicode escape
                // (single BMP code unit only — no surrogate pair handling).
                switch (esc) {
                    case '"': sb.append('"'); break;
                    case '\\': sb.append('\\'); break;
                    case '/': sb.append('/'); break;
                    case 'b': sb.append('\b'); break;
                    case 'f': sb.append('\f'); break;
                    case 'n': sb.append('\n'); break;
                    case 'r': sb.append('\r'); break;
                    case 't': sb.append('\t'); break;
                    case 'u':
                        if (s.i + 4 > s.text.length()) throw err(s, "bad u escape");
                        sb.append((char) Integer.parseInt(
                                s.text.substring(s.i, s.i + 4), 16));
                        s.i += 4;
                        break;
                    default: throw err(s, "unknown escape \\" + esc);
                }
            } else {
                sb.append(c);
                s.i++;
            }
        }
        throw err(s, "unterminated string");
    }

    private static int readInt(State s) {
        int start = s.i;
        if (peek(s) == '-') s.i++;
        while (s.i < s.text.length() && Character.isDigit(s.text.charAt(s.i))) s.i++;
        if (s.i == start) throw err(s, "expected integer");
        return Integer.parseInt(s.text.substring(start, s.i));
    }

    private static void skipWs(State s) {
        while (s.i < s.text.length() && Character.isWhitespace(s.text.charAt(s.i))) s.i++;
    }

    private static char peek(State s) {
        if (s.i >= s.text.length()) throw err(s, "unexpected end of input");
        return s.text.charAt(s.i);
    }

    private static void expect(State s, char c) {
        if (peek(s) != c) throw err(s, "expected '" + c + "', got '" + peek(s) + "'");
        s.i++;
    }

    private static RuntimeException err(State s, String msg) {
        return new RuntimeException("JSON parse error at offset " + s.i + ": " + msg);
    }

    private static final class State {
        final String text;
        int i;
        State(String text) { this.text = text; }
    }
}
