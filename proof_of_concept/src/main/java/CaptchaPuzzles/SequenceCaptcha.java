import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.stage.Modality;
import javafx.stage.Stage;
import javafx.stage.Window;
import java.util.Collections;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Demo Sequence Captcha I had in mind for hard captcha 
 */
public class SequenceCaptcha {

    public static boolean show(Window owner) {
        
        // stage set up
        Stage dialog = new Stage();
        dialog.initOwner(owner);
        dialog.initModality(Modality.WINDOW_MODAL);
        dialog.setTitle("Hard Captcha");

        // sample sequence
        List<String> order = new ArrayList<>(Arrays.asList("A","B","C"));
        Collections.shuffle(order);             // randomly ordered

        Label prompt = new Label("Click the buttons in order: " + String.join(" → ", order));

        // buttons
        Button a = new Button("A");
        Button b = new Button("B");
        Button c = new Button("C");

        // when clicked add to the arraylist in order clicked
        final List<String> clicked = new ArrayList<>();
        a.setOnAction(e -> clicked.add("A"));
        b.setOnAction(e -> clicked.add("B"));
        c.setOnAction(e -> clicked.add("C"));

        HBox row = new HBox(12);
        row.setAlignment(Pos.CENTER);

        // shuffle buttons
        List<Button> btns = new ArrayList<>(Arrays.asList(a, b, c));
        Collections.shuffle(btns);
        row.getChildren().setAll(btns);

        Button submit = new Button("Submit");
        final boolean[] passed = {false};
        
        // pass only if the clicks match the randomized target sequence
        submit.setOnAction(e -> {
            passed[0] = clicked.size() == order.size();
            if (passed[0]) 
            {
                for (int i = 0; i < order.size(); i++) 
                {
                    if (!order.get(i).equals(clicked.get(i))) 
                    { 
                        passed[0] = false; 
                        break; 
                    }
                }
            }
            dialog.close();
        });

        // show prompt
        VBox root = new VBox(16, prompt, row, submit);
        root.setPadding(new Insets(16));
        root.setAlignment(Pos.CENTER);

        dialog.setScene(new Scene(root, 340, 180));
        dialog.showAndWait();
        return passed[0];
    }
}
