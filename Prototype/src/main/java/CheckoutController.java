import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import java.net.URL;
import java.util.ResourceBundle;
import javafx.scene.control.ChoiceBox;
import javafx.scene.control.Button;
import javafx.event.ActionEvent;
import java.util.Random;

public class CheckoutController implements Initializable {

    @FXML
    private ChoiceBox<String> stateChoiceBox;
    @FXML
    private ChoiceBox<String> countryChoiceBox;
    @FXML
    private Button honeypotButton;
    @FXML
    private Button checkoutButton;

    //Four sample states to populate the dropdown
    private final String[] states = { "California", "Ohio", "New York", "Florida" };

    //Sample countries
    private final String[] countries = { "United States", "Canada", "Mexico" };

    private final Random random = new Random();
    private int userScore; // Start with base human score
    private boolean isBot = false;

    @Override
    public void initialize(URL location, ResourceBundle resources) {
        // Populate states ChoiceBox
        stateChoiceBox.getItems().addAll(states);
        stateChoiceBox.setValue("Select State");

        // Populate countries ChoiceBox
        countryChoiceBox.getItems().addAll(countries);
        countryChoiceBox.setValue("Select Country");

        // Set up honeypot button detection
        honeypotButton.setOnAction(this::honeypotClicked);

        // Set up checkout button handler
        checkoutButton.setOnAction(this::handleCheckout);
    }

    @FXML
    public void honeypotClicked(ActionEvent event) {
        isBot = true;
        userScore = 0;
        System.out.println("BOT DETECTED: Honeypot clicked!");
    }

    private void calculateFinalScore() {
        // Add random element to make bot detection harder
        int randomFactor = random.nextInt(20) - 10; // -10 to +10
        userScore = random.nextInt(80);
        System.out.println("Initial user score: " + userScore);
        userScore += randomFactor;

        System.out.println("Final user score: " + userScore);
    }

    @FXML
    public void handleCheckout(ActionEvent event) {
        calculateFinalScore();

        double normalized = Math.max(0, Math.min(1, userScore / 100.0)); // map to [0,1]

        boolean ok = CaptchaManager.run(normalized, isBot, checkoutButton.getScene().getWindow()); // Call captcha manager

        if (ok) {
            System.out.println("ACCESS GRANTED - Human verified (Score: " + userScore + ")");
        } else {
            System.out.println("ACCESS DENIED - Verification failed (Score: " + userScore + ")");
        }
    }

}
