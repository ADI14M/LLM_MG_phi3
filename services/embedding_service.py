from langchain_ollama import OllamaEmbeddings

class EmbeddingService:
    @staticmethod
    def get_embeddings_model():
        """
        Returns the Ollama embedding model instance used across the system.
        """
        return OllamaEmbeddings(model="nomic-embed-text")
