from midmind import PromptBuilder, PromptTemplates, PromptValidator

# Build a prompt
pb = PromptBuilder()
pb.set_instructions("You are an AI assistant.")
pb.set_context("User wants to learn about AI prompt engineering.")
pb.set_task(PromptTemplates.reasoning("Explain reinforcement learning step by step."))

prompt = pb.build()
print("Prompt:")
print(prompt)

# Validate the prompt
validation = PromptValidator.validate(prompt)
print("\nValidation results:", validation)