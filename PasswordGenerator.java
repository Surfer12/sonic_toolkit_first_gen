import java.security.SecureRandom;
import java.util.Scanner;

public class PasswordGenerator {
    private static final String LOWERCASE = "abcdefghijklmnopqrstuvwxyz";
    private static final String UPPERCASE = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    private static final String DIGITS = "0123456789";
    private static final String SPECIAL = "!@#$%^&*()_+-=[]{}|;:,.<>?/~";
    private static SecureRandom random = new SecureRandom();

    /**
     * Generate a secure random password with default settings.
     *
     * @return A 12-character password
     */
    public static String generatePassword() {
        return generatePassword(12, true, true, true, true);
    }

    /**
     * Generate a secure random password with custom length.
     *
     * @param length Password length
     * @return Generated password
     */
    public static String generatePassword(int length) {
        return generatePassword(length, true, true, true, true);
    }

    /**
     * Generate a secure random password with customizable character sets.
     *
     * @param length       Password length (must be >= 4 for security)
     * @param useLower     Include lowercase letters
     * @param useUpper     Include uppercase letters
     * @param useDigits    Include digits
     * @param useSpecial   Include special characters
     * @return Generated password
     */
    public static String generatePassword(int length, boolean useLower, boolean useUpper,
                                        boolean useDigits, boolean useSpecial) {
        if (length < 1) {
            throw new IllegalArgumentException("Password length must be at least 1");
        }

        // Build character set based on options
        StringBuilder charSet = new StringBuilder();
        if (useLower) charSet.append(LOWERCASE);
        if (useUpper) charSet.append(UPPERCASE);
        if (useDigits) charSet.append(DIGITS);
        if (useSpecial) charSet.append(SPECIAL);

        if (charSet.length() == 0) {
            throw new IllegalArgumentException("At least one character type must be selected");
        }

        String characters = charSet.toString();

        // Ensure password contains at least one of each selected type for better security
        StringBuilder password = new StringBuilder();

        // Add at least one character from each selected type
        if (useLower && LOWERCASE.length() > 0) {
            password.append(LOWERCASE.charAt(random.nextInt(LOWERCASE.length())));
        }
        if (useUpper && UPPERCASE.length() > 0) {
            password.append(UPPERCASE.charAt(random.nextInt(UPPERCASE.length())));
        }
        if (useDigits && DIGITS.length() > 0) {
            password.append(DIGITS.charAt(random.nextInt(DIGITS.length())));
        }
        if (useSpecial && SPECIAL.length() > 0) {
            password.append(SPECIAL.charAt(random.nextInt(SPECIAL.length())));
        }

        // Fill the rest randomly
        int remainingLength = length - password.length();
        for (int i = 0; i < remainingLength; i++) {
            int index = random.nextInt(characters.length());
            password.append(characters.charAt(index));
        }

        // Shuffle to avoid predictable patterns
        char[] passwordArray = password.toString().toCharArray();
        for (int i = passwordArray.length - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            char temp = passwordArray[i];
            passwordArray[i] = passwordArray[j];
            passwordArray[j] = temp;
        }

        return new String(passwordArray);
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        try {
            System.out.print("How many passwords do you want to generate? ");
            int numPasswords = scanner.nextInt();

            if (numPasswords < 1) {
                throw new IllegalArgumentException("Number of passwords must be at least 1");
            }

            System.out.print("Enter the desired password length (minimum 4): ");
            int passwordLength = scanner.nextInt();

            if (passwordLength < 4) {
                System.out.println("Warning: Passwords shorter than 4 characters are less secure.");
            }

            // Get user preferences
            System.out.print("Include special characters? (y/n): ");
            boolean useSpecial = scanner.next().toLowerCase().startsWith("y");

            System.out.print("Include digits? (y/n): ");
            boolean useDigits = scanner.next().toLowerCase().startsWith("y");

            System.out.print("Include uppercase letters? (y/n): ");
            boolean useUppercase = scanner.next().toLowerCase().startsWith("y");

            System.out.print("Include lowercase letters? (y/n): ");
            boolean useLowercase = scanner.next().toLowerCase().startsWith("y");

            System.out.println("\nGenerated Passwords:");
            System.out.println("-".repeat(50));

            for (int i = 0; i < numPasswords; i++) {
                String password = generatePassword(passwordLength, useLowercase, useUppercase,
                                                 useDigits, useSpecial);
                System.out.println("Password " + (i + 1) + ": " + password);
            }

            System.out.println("-".repeat(50));
            System.out.println("Note: These passwords are generated using cryptographically secure methods.");
            System.out.println("Store them securely and don't share them.");

        } catch (Exception e) {
            System.out.println("Error: " + e.getMessage());
            System.out.println("Please enter valid inputs.");
        } finally {
            scanner.close();
        }
    }
}
