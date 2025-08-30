"""
Chat Data Models in Mojo - High Performance Port from Python

Ported from automated-auditing Python data models to leverage Mojo's
zero-cost abstractions and compile-time optimizations.
"""

from collections import List, Dict
from utils import StringRef

@value
struct MessageRole:
    """Enum-like struct for message roles with compile-time optimization."""
    var value: StringRef
    
    fn __init__(inout self, value: StringRef):
        self.value = value
    
    @staticmethod
    fn user() -> MessageRole:
        return MessageRole("user")
    
    @staticmethod
    fn assistant() -> MessageRole:
        return MessageRole("assistant")
    
    @staticmethod
    fn system() -> MessageRole:
        return MessageRole("system")
    
    fn __eq__(self, other: MessageRole) -> Bool:
        return self.value == other.value
    
    fn __str__(self) -> String:
        return str(self.value)

@value
struct ChatMessage:
    """High-performance chat message with zero-copy string operations."""
    var role: MessageRole
    var content: String
    
    fn __init__(inout self, role: MessageRole, content: String):
        self.role = role
        self.content = content
    
    fn __str__(self) -> String:
        return str(self.role) + ": " + self.content
    
    fn to_dict(self) -> Dict[String, String]:
        """Convert to dictionary format for API calls."""
        var result = Dict[String, String]()
        result["role"] = str(self.role)
        result["content"] = self.content
        return result

struct Prompt:
    """Optimized prompt container with efficient message handling."""
    var messages: List[ChatMessage]
    
    fn __init__(inout self):
        self.messages = List[ChatMessage]()
    
    fn __init__(inout self, messages: List[ChatMessage]):
        self.messages = messages
    
    fn add_message(inout self, role: MessageRole, content: String):
        """Add a message to the prompt."""
        self.messages.append(ChatMessage(role, content))
    
    fn add_system_message(inout self, content: String):
        """Add a system message."""
        self.add_message(MessageRole.system(), content)
    
    fn add_user_message(inout self, content: String):
        """Add a user message."""
        self.add_message(MessageRole.user(), content)
    
    fn add_assistant_message(inout self, content: String):
        """Add an assistant message."""
        self.add_message(MessageRole.assistant(), content)
    
    fn __str__(self) -> String:
        """Convert all messages to formatted string."""
        var result = String("")
        for i in range(len(self.messages)):
            if i > 0:
                result += "\n"
            result += str(self.messages[i])
        return result
    
    fn to_msg_dict_list(self) -> List[Dict[String, String]]:
        """Convert messages to list of dictionaries for API calls."""
        var result = List[Dict[String, String]]()
        for i in range(len(self.messages)):
            result.append(self.messages[i].to_dict())
        return result
    
    fn get_system_prompt(self) -> String:
        """Extract system prompt if present."""
        if len(self.messages) > 0 and self.messages[0].role == MessageRole.system():
            return self.messages[0].content
        return ""
    
    fn has_system_prompt(self) -> Bool:
        """Check if prompt has a system message."""
        return len(self.messages) > 0 and self.messages[0].role == MessageRole.system()
    
    fn get_non_system_messages(self) -> List[ChatMessage]:
        """Get all messages except system message."""
        var result = List[ChatMessage]()
        let start_idx = 1 if self.has_system_prompt() else 0
        
        for i in range(start_idx, len(self.messages)):
            result.append(self.messages[i])
        return result
    
    fn message_count(self) -> Int:
        """Get total number of messages."""
        return len(self.messages)
    
    fn validate(self) -> Bool:
        """Validate prompt structure."""
        if len(self.messages) == 0:
            return False
        
        # Check for alternating user/assistant pattern after system message
        let start_idx = 1 if self.has_system_prompt() else 0
        var expecting_user = True
        
        for i in range(start_idx, len(self.messages)):
            let msg = self.messages[i]
            if expecting_user and msg.role != MessageRole.user():
                return False
            elif not expecting_user and msg.role != MessageRole.assistant():
                return False
            expecting_user = not expecting_user
        
        return True

# Utility functions for prompt construction
fn create_simple_prompt(system_content: String, user_content: String) -> Prompt:
    """Create a simple prompt with system and user messages."""
    var prompt = Prompt()
    if system_content != "":
        prompt.add_system_message(system_content)
    prompt.add_user_message(user_content)
    return prompt

fn create_conversation_prompt(
    system_content: String, 
    conversation_pairs: List[Tuple[String, String]]
) -> Prompt:
    """Create a prompt from conversation pairs (user, assistant)."""
    var prompt = Prompt()
    if system_content != "":
        prompt.add_system_message(system_content)
    
    for i in range(len(conversation_pairs)):
        let pair = conversation_pairs[i]
        prompt.add_user_message(pair.get[0, String]())
        prompt.add_assistant_message(pair.get[1, String]())
    
    return prompt

# Demo function
fn demo_chat_models():
    """Demonstrate the high-performance chat models."""
    print("🚀 Mojo Chat Models Demo")
    print("=" * 40)
    
    # Create a prompt
    var prompt = Prompt()
    prompt.add_system_message("You are a helpful AI assistant specialized in Mojo programming.")
    prompt.add_user_message("How do I optimize matrix operations in Mojo?")
    prompt.add_assistant_message("You can use SIMD operations and vectorization for optimal performance.")
    prompt.add_user_message("Can you show me an example?")
    
    print("Prompt structure:")
    print(str(prompt))
    print()
    
    print("System prompt:", prompt.get_system_prompt())
    print("Message count:", prompt.message_count())
    print("Has system prompt:", prompt.has_system_prompt())
    print("Is valid:", prompt.validate())
    print()
    
    # Create simple prompt
    let simple = create_simple_prompt(
        "You are a Mojo expert.",
        "What are the benefits of Mojo over Python?"
    )
    print("Simple prompt:")
    print(str(simple))
    
    print("\n✅ Chat models demo completed!")

fn main():
    """Main entry point."""
    demo_chat_models()
