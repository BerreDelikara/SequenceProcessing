import SequenceProcessing.Tokenization.*;
import java.util.*;

String sentence = "Kitaptan kütüphane oluştu";
System.out.println("Input: " + sentence);
System.out.println();

TurkishTokenizer tt = new TurkishTokenizer(200);
tt.train(Arrays.asList("kitap kitaplar", "kütüphane", "ol"));
System.out.println("TurkishTokenizer (FSM):  " + tt.encode(sentence));
System.out.println("CosmosGPT2 (BPE 50257):  " + new CosmosGPT2Tokenizer().encode(sentence));
System.out.println("TabiBERT  (BPE 50176):   " + new TabiBERTTokenizer().encode(sentence));
System.out.println("Mursit    (BPE 59008):   " + new MursitTokenizer().encode(sentence));
/exit
