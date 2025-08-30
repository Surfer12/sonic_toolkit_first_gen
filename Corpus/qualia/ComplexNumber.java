// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
//
//  ComplexNumber.java
//  qualia
//
//  Created by Ryan David Oates on 8/26/25.
//  Complex number implementation for reverse koopman operators
package qualia;

import java.lang.Math;

/**
 * Complex number representation for eigenvalue computations
 */
public class ComplexNumber {
    private final double real;
    private final double imaginary;

    public ComplexNumber(double real, double imaginary) {
        this.real = real;
        this.imaginary = imaginary;
    }

    public double getReal() { return real; }
    public double getImaginary() { return imaginary; }

    public double magnitude() {
        return Math.sqrt(real * real + imaginary * imaginary);
    }

    public double phase() {
        return Math.atan2(imaginary, real);
    }

    public ComplexNumber add(ComplexNumber other) {
        return new ComplexNumber(real + other.real, imaginary + other.imaginary);
    }

    public ComplexNumber multiply(ComplexNumber other) {
        return new ComplexNumber(
            real * other.real - imaginary * other.imaginary,
            real * other.imaginary + imaginary * other.real
        );
    }

    public ComplexNumber conjugate() {
        return new ComplexNumber(real, -imaginary);
    }

    @Override
    public String toString() {
        return String.format("%.6f + %.6fi", real, imaginary);
    }
}
