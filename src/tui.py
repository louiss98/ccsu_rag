from src.chunker.embedder import DocumentEmbedder

class RAGApp:
    def __init__(self):
        # Initialize the embedder and load the go2_robot vector store
        self.embedder = DocumentEmbedder()
        self.vector_store = self.embedder.load_vector_store("unitree_vector_store/go2_robot")

    def retrieve_context(self, query):
        # Use the vector store to retrieve relevant context for the query
        results = self.embedder.search_similar(query, k=5)
        return results

    def run(self):
        print("Welcome to the RAG App!")
        while True:
            query = input("Enter your query (or 'exit' to quit): ")
            if query.lower() == 'exit':
                break

            # Retrieve context and display results
            context = self.retrieve_context(query)
            print("\nRelevant Context:")
            for i, doc in enumerate(context):
                print(f"[{i+1}] {doc.page_content}\n")

if __name__ == "__main__":
    app = RAGApp()
    app.run()