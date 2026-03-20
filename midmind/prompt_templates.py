class PromptTemplates:

    @staticmethod
    def summarization(text):
        """
        Create a summarization prompt for any text.
        """
        return f"Summarize the following text:\n{text}"

    @staticmethod
    def classification(text, labels):
        """
        Create a classification prompt for a text given a list of labels.
        """
        labels_str = ", ".join(labels)
        return f"Classify the text into one of the categories ({labels_str}):\n{text}"

    @staticmethod
    def reasoning(question):
        """
        Create a reasoning prompt that encourages step-by-step thinking.
        """
        return f"Think step by step and answer the question:\n{question}"