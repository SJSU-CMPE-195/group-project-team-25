import javafx.animation.PauseTransition;
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
import javafx.util.Duration;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Memory Flash Captcha I had in mind for hard captcha 
 */
public class MemoryFlashCaptcha {

    public static boolean show(Window owner) {
        
        // set up for memory flash
        Stage dialog = new Stage();
        dialog.initOwner(owner);
        dialog.initModality(Modality.WINDOW_MODAL);
        dialog.setTitle("Memory Flash");

        // prompt for user
        Label prompt = new Label("Emoji will flash briefly.");
        Label flash = new Label("Bee");
        flash.setStyle("-fx-font-size: 36px;");

        // alignment + padding
        VBox stage1 = new VBox(10, prompt, flash);
        stage1.setAlignment(Pos.CENTER);

        // centering root and padding
        VBox root = new VBox(16, stage1);
        root.setPadding(new Insets(16));
        root.setAlignment(Pos.CENTER);

        Scene scene = new Scene(root, 360, 220);
        dialog.setScene(scene);

        final boolean[] passed = {false};

        // transition to question
        PauseTransition pt = new PauseTransition(Duration.millis(800));
        
        // show the question prompt
        pt.setOnFinished(ev -> {
            stage1.getChildren().clear();
            Label q = new Label("Which emoji did you see?");
            List<String> picks = Arrays.asList("Butterfly","Bee","LadyBug");
            Collections.shuffle(picks);

            // buttons they must click bee
            HBox row = new HBox(12);
            row.setAlignment(Pos.CENTER);
            for (String p : picks) {
                Button btn = new Button(p);
                btn.setOnAction(e -> {
                    passed[0] = "Bee".equals(p);
                    dialog.close();
                });
                row.getChildren().add(btn);
            }

            VBox second = new VBox(16, q, row);
            second.setAlignment(Pos.CENTER);
            root.getChildren().setAll(second);
        });

        // run the dialog
        dialog.setOnShown(e -> pt.playFromStart());
        dialog.showAndWait();

        return passed[0];
    }
}

