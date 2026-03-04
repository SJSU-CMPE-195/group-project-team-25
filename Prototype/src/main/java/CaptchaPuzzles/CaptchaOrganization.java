public class CaptchaOrganization {

    // buckets of CAPTCHA
    public enum Bucket {
        ZERO_FRICTION,      // when score is less or equal than 0.2
        EASY_CAPTCHA,       // when score is > 0.2 and less than 0.5
        HARD_CAPTCHA,       // when score is > 0.5 and less than 0.9
        LOCKED_OUT          // when score is greater than 0.9
    }

    /**
     * Inner Class - Descion made based on score
     */
    public static class Decision {
        public final Bucket bucket;                 // which bucket
        public final boolean explained_telemetry;   // explainable results + more logging
        public final int maxRetries;                // limit on failures
        public final boolean requireOtp;            // what happens when locked out / are we locked out

        public Decision (Bucket bucket, boolean explained_telemetry, int maxRetries, boolean requireOtp)
        {
            this.bucket = bucket;
            this.explained_telemetry = explained_telemetry;
            this.maxRetries = maxRetries;
            this.requireOtp = requireOtp;
        }

        public String toString()
        {
            return "bucket chosen:" + this.bucket;
        }

    }

    public static Decision decide(double score, boolean honeyPotHit)
    {
        // if they hit honey pot (for now just auto send them to hard captcha)
        if (honeyPotHit)
        {
            score = Math.max(0.5, score);
        }

        if (score <= 0.2)
        {
            return new Decision(Bucket.ZERO_FRICTION, false, 0, false);
        }
        else if (score <= 0.5)
        {
            return new Decision(Bucket.EASY_CAPTCHA, false, 1, false);
        }
        else if (score <= 0.9)
        {
            return new Decision(Bucket.HARD_CAPTCHA, true, 2, false);
        }
        else
        {
            return new Decision(Bucket.LOCKED_OUT, true, 0, true);
        }
    }
}