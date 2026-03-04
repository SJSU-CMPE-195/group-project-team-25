import javafx.scene.control.Alert;
import javafx.scene.control.ButtonType;
import javafx.stage.Window;

/**
 * Manage the captchas for each bucket
 */
public class CaptchaManager {

    public static boolean run(double score, boolean honeyPotHit, Window owner)
    {
        // Get the decision 
        CaptchaOrganization.Decision decision = CaptchaOrganization.decide(score, honeyPotHit);
        System.out.println("Captcha Decision:" + decision);

        // switch based on each one (true if the person passes, false otherwise)
        switch (decision.bucket) 
        {
            case ZERO_FRICTION:
                return true;

            case EASY_CAPTCHA:
                boolean emojiCAPTCHA = EmojiMatchCaptcha.show(owner);
                return emojiCAPTCHA;

            case HARD_CAPTCHA:

                boolean sequence_puzzle = SequenceCaptcha.show(owner);
                if (!sequence_puzzle)
                {
                    return false;
                }

                boolean memoryCaptcha = MemoryFlashCaptcha.show(owner);
                if (!memoryCaptcha)
                {
                    return false;
                }
                return true;

            case LOCKED_OUT:
                boolean otpValidation = OTPDialog.show(owner);
                if (!otpValidation)
                {
                    Alert a = new Alert(Alert.AlertType.ERROR, "locked out - cant buy", ButtonType.OK);
                    a.initOwner(owner);
                    a.showAndWait();
                    return false;
                }
                return true;
        }
        return false;
    }
}