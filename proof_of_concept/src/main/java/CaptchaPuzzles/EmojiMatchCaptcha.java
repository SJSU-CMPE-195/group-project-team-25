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
import javafx.scene.text.Font;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Easy Captcha example (where they get image must select corresponding emoji)
 */
public class EmojiMatchCaptcha {

    public static boolean show(Window owner)
    {
        
        // set up the stage
        Stage dialog = new Stage ();
        dialog.initOwner(owner);
        dialog.initModality(Modality.WINDOW_MODAL);
        dialog.setTitle("Easy captcha");

        // prompt for emoji 
        Label prompt = new Label("Select the emoji that matches: Apple");
        prompt.setWrapText(true);

        // button of options
        List<String> options = Arrays.asList("Apple","Banana","Grape");
        Collections.shuffle(options);

        final boolean[] passed = {false};

        HBox buttons = new HBox(12);
        buttons.setAlignment(Pos.CENTER);

        // select correct emoji
        for (String o : options) {
            Button b = new Button(o);
            b.setOnAction(e -> {
                passed[0] = "Apple".equals(o);
                dialog.close();
            });
            buttons.getChildren().add(b);
        }

        // show dialog + root
        VBox root = new VBox(16, prompt, buttons);
        root.setAlignment(Pos.CENTER);
        root.setPadding(new Insets(16));

        dialog.setScene(new Scene(root, 320, 160));
        dialog.showAndWait();
        return passed[0];
    }
}