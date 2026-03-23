def autoragx(data_path, query):
    """
    Minimal AutoRAGX placeholder.
    Version 0.1 – simple file loader + search.
    """

    import os

    text_data = ""

    for file in os.listdir(data_path):
        path = os.path.join(data_path, file)
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text_data += f.read()

    if query.lower() in text_data.lower():
        return "Query found in documents."
    
    return text_data[:1000]