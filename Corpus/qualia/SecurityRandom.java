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

/**
 * SecurityRandom - Cryptographically secure random number generation utility
 */
public class SecurityRandom {

    private final SecureRandom secureRandom;
    private int operationCount;

    public SecurityRandom() {
        this.secureRandom = new SecureRandom();
        this.operationCount = 0;
    }

    public byte[] generateRandomBytes(int length) {
        if (length <= 0) {
            throw new IllegalArgumentException("Length must be positive");
        }
        operationCount++;
        byte[] bytes = new byte[length];
        secureRandom.nextBytes(bytes);
        return bytes;
    }

    public int generateSecureInt() {
        operationCount++;
        return secureRandom.nextInt();
    }

    public int generateSecureInt(int bound) {
        if (bound <= 0) {
            throw new IllegalArgumentException("Bound must be positive");
        }
        operationCount++;
        return secureRandom.nextInt(bound);
    }

    public long generateSecureLong() {
        operationCount++;
        return secureRandom.nextLong();
    }

    public double generateSecureDouble() {
        operationCount++;
        return secureRandom.nextDouble();
    }

    public boolean generateSecureBoolean() {
        operationCount++;
        return secureRandom.nextBoolean();
    }

    public void fillSecureBytes(byte[] bytes) {
        if (bytes == null) {
            throw new IllegalArgumentException("Byte array cannot be null");
        }
        operationCount++;
        secureRandom.nextBytes(bytes);
    }

    public int getOperationCount() {
        return operationCount;
    }

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
