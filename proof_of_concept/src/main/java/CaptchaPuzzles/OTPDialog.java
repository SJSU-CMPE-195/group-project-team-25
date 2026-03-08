import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.PasswordField;
import javafx.scene.layout.VBox;
import javafx.stage.Modality;
import javafx.stage.Stage;
import javafx.stage.Window;

/**
 * OTP dialog for when user is assumed to be bot with high risk score
 * 000000 - for demo
 */
public class OTPDialog {

    public static boolean show(Window owner) {
        
        // set up stage
        Stage dialog = new Stage();
        dialog.initOwner(owner);
        dialog.initModality(Modality.WINDOW_MODAL);
        dialog.setTitle("Phone Verification");

        // prompt for 6 digit code (000000)
        Label label = new Label("Enter the 6-digit code we sent to your phone.");
        PasswordField code = new PasswordField();
        code.setPromptText("6-digit code");

        // check for equality
        final boolean[] ok = {false};
        Button verify = new Button("Verify");
        verify.setOnAction(e -> {
            ok[0] = "000000".equals(code.getText());
            dialog.close();
        });

        // print result
        VBox root = new VBox(12, label, code, verify);
        root.setPadding(new Insets(16));

        dialog.setScene(new Scene(root, 360, 160));
        dialog.showAndWait();
        return ok[0];
    }
}
