class PromptValidator:

    @staticmethod
    def check_length(prompt, max_words=500):
        """
        Check if the prompt does not exceed max_words.
        """
        return len(prompt.split()) <= max_words

    @staticmethod
    def has_instruction(prompt):
        """
        Check if prompt contains keywords that indicate instructions.
        """
        keywords = ["instruction", "task", "explain", "please"]
        return any(word in prompt.lower() for word in keywords)

    @staticmethod
    def validate(prompt):
        """
        Return a dictionary with validation results.
        """
        results = {
            "length_ok": PromptValidator.check_length(prompt),
            "has_instruction": PromptValidator.has_instruction(prompt)
        }
        return results