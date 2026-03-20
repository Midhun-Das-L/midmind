class PromptBuilder:

    def __init__(self, role="assistant"):
        self.role = role
        self.instructions = ""
        self.context = ""
        self.task = ""

    def set_instructions(self, instructions):
        self.instructions = instructions

    def set_context(self, context):
        self.context = context

    def set_task(self, task):
        self.task = task

    def build(self):
        prompt = f"""
Role: {self.role}

Instructions:
{self.instructions}

Context:
{self.context}

Task:
{self.task}
"""
        return prompt.strip()