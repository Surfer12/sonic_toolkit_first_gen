import secrets
import string

def generate_password(length=12, use_special=True, use_digits=True, use_uppercase=True, use_lowercase=True):
    """
    Generate a secure random password.

    Args:
        length (int): Length of the password
        use_special (bool): Include special characters
        use_digits (bool): Include digits
        use_uppercase (bool): Include uppercase letters
        use_lowercase (bool): Include lowercase letters

    Returns:
        str: Generated password
    """
    if length < 1:
        raise ValueError("Password length must be at least 1")

    # Build character set based on options
    characters = ""
    if use_lowercase:
        characters += string.ascii_lowercase
    if use_uppercase:
        characters += string.ascii_uppercase
    if use_digits:
        characters += string.digits
    if use_special:
        characters += "!@#$%^&*()_+-=[]{}|;:,.<>?/~"

    if not characters:
        raise ValueError("At least one character type must be selected")

    # Ensure password contains at least one of each selected type for better security
    password = []

    # Add at least one character from each selected type
    if use_lowercase:
        password.append(secrets.choice(string.ascii_lowercase))
    if use_uppercase:
        password.append(secrets.choice(string.ascii_uppercase))
    if use_digits:
        password.append(secrets.choice(string.digits))
    if use_special:
        password.append(secrets.choice("!@#$%^&*()_+-=[]{}|;:,.<>?/~"))

    # Fill the rest randomly
    remaining_length = length - len(password)
    for _ in range(remaining_length):
        password.append(secrets.choice(characters))

    # Shuffle to avoid predictable patterns
    secrets.SystemRandom().shuffle(password)

    return ''.join(password)

def main():
    try:
        num_passwords = int(input("How many passwords do you want to generate? "))
        if num_passwords < 1:
            raise ValueError("Number of passwords must be at least 1")

        password_length = int(input("Enter the desired password length (minimum 4): "))
        if password_length < 4:
            print("Warning: Passwords shorter than 4 characters are less secure.")

        # Get user preferences
        use_special = input("Include special characters? (y/n): ").lower().startswith('y')
        use_digits = input("Include digits? (y/n): ").lower().startswith('y')
        use_uppercase = input("Include uppercase letters? (y/n): ").lower().startswith('y')
        use_lowercase = input("Include lowercase letters? (y/n): ").lower().startswith('y')

        print("\nGenerated Passwords:")
        print("-" * 50)

        for i in range(num_passwords):
            password = generate_password(
                length=password_length,
                use_special=use_special,
                use_digits=use_digits,
                use_uppercase=use_uppercase,
                use_lowercase=use_lowercase
            )
            print(f"Password {i+1}: {password}")

        print("-" * 50)
        print("Note: These passwords are generated using cryptographically secure methods.")
        print("Store them securely and don't share them.")

    except ValueError as e:
        print(f"Error: {e}")
        print("Please enter valid positive integers.")
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
