from midmind import RAGPromptBuilder, autoragx

# One line: folder + question -> grounded prompt
print(autoragx("data", "What is artificial intelligence?"))

print("\n" + "=" * 60 + "\n")

# Full control
rag = RAGPromptBuilder.from_folder("data", chunk_size=500, overlap=50)
for chunk in rag.retrieve("human intelligence machines", top_k=3):
    print(chunk)
print(rag.build_messages("What is artificial intelligence?", template="qa"))
