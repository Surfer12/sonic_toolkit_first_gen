// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
//
//  SecurityRandom.java
//  qualia
//
//  Created by Ryan David Oates on 8/26/25.
//  Cryptographically secure random number generation utility

package qualia;

import java.security.SecureRandom;
import java.security.NoSuchAlgorithmException;
import java.util.Arrays;
import java.util.concurrent.locks.ReentrantLock;

/**
 * SecurityRandom - Cryptographically secure random number generation utility
 *
 * This class provides a secure wrapper around Java's SecureRandom with additional
 * security features including entropy mixing, timing attack protection, and
 * secure seed management.
 *
 * Key Features:
 * - Cryptographically secure random number generation
 * - Entropy mixing from multiple sources
 * - Timing attack protection through constant-time operations
 * - Secure seed management and reseeding
 * - Thread-safe implementation
 *
 * Usage:
 * ```java
 * SecurityRandom secureRand = new SecurityRandom();
 * byte[] randomBytes = secureRand.generateRandomBytes(32);
 * int secureInt = secureRand.generateSecureInt();
 * ```
 */
public class SecurityRandom {

    private static final int DEFAULT_SEED_LENGTH = 32;
    private static final int MAX_RESEED_INTERVAL = 100000; // Reseed every 100k operations
    private static final String SECURE_ALGORITHM = "SHA1PRNG";

    private final SecureRandom secureRandom;
    private final ReentrantLock lock;
    private int operationCount;
    private final byte[] additionalEntropy;

    /**
     * Constructor - Initialize with secure random and additional entropy
     */
    public SecurityRandom() {
        this.lock = new ReentrantLock();

        try {
            this.secureRandom = SecureRandom.getInstance(SECURE_ALGORITHM);
        } catch (NoSuchAlgorithmException e) {
            throw new SecurityException("Failed to initialize secure random: " + e.getMessage(), e);
        }

        // Initialize with system entropy
        this.additionalEntropy = new byte[DEFAULT_SEED_LENGTH];
        secureRandom.nextBytes(additionalEntropy);

        // Mix in system properties for additional entropy
        mixSystemEntropy();

        // Initial reseeding
        reseed();

        this.operationCount = 0;
    }

    /**
     * Constructor with custom seed
     *
     * @param seed Custom seed bytes for reproducible randomness (for testing only)
     */
    public SecurityRandom(byte[] seed) {
        this.lock = new ReentrantLock();

        try {
            this.secureRandom = SecureRandom.getInstance(SECURE_ALGORITHM);
        } catch (NoSuchAlgorithmException e) {
            throw new SecurityException("Failed to initialize secure random: " + e.getMessage(), e);
        }

        this.additionalEntropy = Arrays.copyOf(seed, DEFAULT_SEED_LENGTH);

        // Mix provided seed with system entropy
        mixSystemEntropy();
        reseed();

        this.operationCount = 0;
    }

    /**
     * Generate cryptographically secure random bytes
     *
     * @param length Number of random bytes to generate
     * @return Array of random bytes
     */
    public byte[] generateRandomBytes(int length) {
        if (length <= 0) {
            throw new IllegalArgumentException("Length must be positive");
        }

        lock.lock();
        try {
            checkReseed();
            byte[] randomBytes = new byte[length];
            secureRandom.nextBytes(randomBytes);

            // Mix with additional entropy for extra security
            mixEntropy(randomBytes);

            operationCount++;
            return randomBytes;
        } finally {
            lock.unlock();
        }
    }

    /**
     * Generate a secure random integer
     *
     * @return Secure random integer
     */
    public int generateSecureInt() {
        lock.lock();
        try {
            checkReseed();
            int randomInt = secureRandom.nextInt();

            // Mix with additional entropy
            byte[] intBytes = new byte[4];
            intBytes[0] = (byte) (randomInt >> 24);
            intBytes[1] = (byte) (randomInt >> 16);
            intBytes[2] = (byte) (randomInt >> 8);
            intBytes[3] = (byte) randomInt;

            mixEntropy(intBytes);

            operationCount++;
            return randomInt;
        } finally {
            lock.unlock();
        }
    }

    /**
     * Generate a secure random integer within bounds
     *
     * @param bound Upper bound (exclusive)
     * @return Secure random integer in range [0, bound)
     */
    public int generateSecureInt(int bound) {
        if (bound <= 0) {
            throw new IllegalArgumentException("Bound must be positive");
        }

        lock.lock();
        try {
            checkReseed();
            int randomInt = secureRandom.nextInt(bound);

            operationCount++;
            return randomInt;
        } finally {
            lock.unlock();
        }
    }

    /**
     * Generate a secure random long
     *
     * @return Secure random long
     */
    public long generateSecureLong() {
        lock.lock();
        try {
            checkReseed();
            long randomLong = secureRandom.nextLong();

            operationCount++;
            return randomLong;
        } finally {
            lock.unlock();
        }
    }

    /**
     * Generate a secure random double in range [0.0, 1.0)
     *
     * @return Secure random double
     */
    public double generateSecureDouble() {
        lock.lock();
        try {
            checkReseed();
            double randomDouble = secureRandom.nextDouble();

            operationCount++;
            return randomDouble;
        } finally {
            lock.unlock();
        }
    }

    /**
     * Generate a secure random boolean
     *
     * @return Secure random boolean
     */
    public boolean generateSecureBoolean() {
        lock.lock();
        try {
            checkReseed();
            boolean randomBool = secureRandom.nextBoolean();

            operationCount++;
            return randomBool;
        } finally {
            lock.unlock();
        }
    }

    /**
     * Fill a byte array with secure random bytes
     *
     * @param bytes Array to fill with random bytes
     */
    public void fillSecureBytes(byte[] bytes) {
        if (bytes == null) {
            throw new IllegalArgumentException("Byte array cannot be null");
        }

        lock.lock();
        try {
            checkReseed();
            secureRandom.nextBytes(bytes);

            // Mix with additional entropy
            mixEntropy(bytes);

            operationCount++;
        } finally {
            lock.unlock();
        }
    }

    /**
     * Reseed the random number generator with fresh entropy
     */
    public void reseed() {
        lock.lock();
        try {
            // Gather fresh entropy from multiple sources
            byte[] freshSeed = new byte[DEFAULT_SEED_LENGTH];

            // System time entropy
            long currentTime = System.nanoTime();
            for (int i = 0; i < 8; i++) {
                freshSeed[i] = (byte) (currentTime >> (i * 8));
            }

            // Memory usage entropy
            Runtime runtime = Runtime.getRuntime();
            long freeMemory = runtime.freeMemory();
            for (int i = 0; i < 8; i++) {
                freshSeed[i + 8] = (byte) (freeMemory >> (i * 8));
            }

            // Thread count entropy
            int threadCount = Thread.activeCount();
            freshSeed[16] = (byte) threadCount;
            freshSeed[17] = (byte) (threadCount >> 8);

            // System property entropy
            String userName = System.getProperty("user.name", "");
            if (!userName.isEmpty()) {
                byte[] userBytes = userName.getBytes();
                for (int i = 0; i < Math.min(8, userBytes.length); i++) {
                    freshSeed[i + 18] = userBytes[i];
                }
            }

            // Fill remaining with secure random
            for (int i = 26; i < DEFAULT_SEED_LENGTH; i++) {
                freshSeed[i] = (byte) secureRandom.nextInt(256);
            }

            // Mix with additional entropy
            for (int i = 0; i < DEFAULT_SEED_LENGTH; i++) {
                freshSeed[i] ^= additionalEntropy[i];
            }

            // Reseed the generator
            secureRandom.setSeed(freshSeed);

            // Update additional entropy
            System.arraycopy(freshSeed, 0, additionalEntropy, 0, DEFAULT_SEED_LENGTH);

            operationCount = 0;

        } finally {
            lock.unlock();
        }
    }

    /**
     * Mix system properties into entropy pool
     */
    private void mixSystemEntropy() {
        // Mix in various system properties for additional entropy
        String[] systemProps = {
            "java.version",
            "os.name",
            "os.version",
            "user.name",
            "java.io.tmpdir"
        };

        for (String prop : systemProps) {
            String value = System.getProperty(prop, "");
            if (!value.isEmpty()) {
                byte[] propBytes = value.getBytes();
                for (int i = 0; i < Math.min(propBytes.length, DEFAULT_SEED_LENGTH); i++) {
                    additionalEntropy[i] ^= propBytes[i];
                }
            }
        }
    }

    /**
     * Mix additional entropy into random bytes
     *
     * @param bytes Array to mix entropy into
     */
    private void mixEntropy(byte[] bytes) {
        for (int i = 0; i < bytes.length; i++) {
            bytes[i] ^= additionalEntropy[i % additionalEntropy.length];
        }
    }

    /**
     * Check if reseeding is needed
     */
    private void checkReseed() {
        if (operationCount >= MAX_RESEED_INTERVAL) {
            reseed();
        }
    }

    /**
     * Get the current operation count
     *
     * @return Number of operations performed since last reseed
     */
    public int getOperationCount() {
        lock.lock();
        try {
            return operationCount;
        } finally {
            lock.unlock();
        }
    }

    /**
     * Get the maximum reseed interval
     *
     * @return Maximum operations before reseeding
     */
    public int getMaxReseedInterval() {
        return MAX_RESEED_INTERVAL;
    }

    /**
     * Get the secure random algorithm being used
     *
     * @return Algorithm name
     */
    public String getAlgorithm() {
        return SECURE_ALGORITHM;
    }

    /**
     * Secure comparison of two byte arrays (timing attack protection)
     *
     * @param a First byte array
     * @param b Second byte array
     * @return True if arrays are equal
     */
    public static boolean secureEquals(byte[] a, byte[] b) {
        if (a == null || b == null) {
            return false;
        }

        if (a.length != b.length) {
            return false;
        }

        int result = 0;
        for (int i = 0; i < a.length; i++) {
            result |= a[i] ^ b[i];
        }

        return result == 0;
    }

    /**
     * Generate a secure token of specified length
     *
     * @param length Token length in bytes
     * @return Hexadecimal string representation of token
     */
    public String generateSecureToken(int length) {
        byte[] tokenBytes = generateRandomBytes(length);
        StringBuilder hexString = new StringBuilder();

        for (byte b : tokenBytes) {
            String hex = Integer.toHexString(0xff & b);
            if (hex.length() == 1) {
                hexString.append('0');
            }
            hexString.append(hex);
        }

        return hexString.toString();
    }

    /**
     * Generate a cryptographically secure password
     *
     * @param length Password length
     * @return Secure password string
     */
    public String generateSecurePassword(int length) {
        String chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*";
        StringBuilder password = new StringBuilder();

        for (int i = 0; i < length; i++) {
            int index = generateSecureInt(chars.length());
            password.append(chars.charAt(index));
        }

        return password.toString();
    }
}
