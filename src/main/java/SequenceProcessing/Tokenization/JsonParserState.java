package SequenceProcessing.Tokenization;

/**
 * Mutable cursor over the JSON source text consumed by {@link JsonObjectParser}.
 * Package-private; not intended for external use.
 */
final class JsonParserState {
    final String text;
    int i;

    JsonParserState(String text) {
        this.text = text;
    }
}
