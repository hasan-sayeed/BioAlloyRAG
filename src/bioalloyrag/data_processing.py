# src/rag_skeleton/data_processing.py
import os

import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


class DataProcessor:
    """
    Handles loading, processing, and creating vector databases for documents.
    """

    def __init__(
        self,
        vectordb_path="vectordb",
        input_data_path="data/raw",
        output_data_path="data/processed",
        embedding_model="Alibaba-NLP/gte-large-en-v1.5",
    ):
        """
        Initialize the DataProcessor with default values.
        """
        self.input_data_path = input_data_path
        self.output_data_path = output_data_path
        self.vectordb_path = vectordb_path
        self.embedding = HuggingFaceEmbeddings(
            model_name=embedding_model, model_kwargs={"trust_remote_code": True}
        )
        self.vector_store = None

    def remove_references(self, input_file_path, output_file_path):
        """
        Remove references section from a PDF file and save it to the output file path.
        """
        doc = fitz.open(input_file_path)
        references_found = False

        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            if "References" in page.get_text() and not references_found:
                references_found = True
                text_instances = page.search_for("References")
                for inst in text_instances:
                    redact_area = fitz.Rect(
                        inst.x0, inst.y0, page.rect.width, page.rect.height
                    )
                    page.add_redact_annot(redact_area)
                page.apply_redactions()
            elif references_found:
                page.add_redact_annot(page.rect)
                page.apply_redactions()

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        doc.save(output_file_path)
        doc.close()
        # print(f"References removed and saved to: {output_file_path}")

    def clean_documents(self):
        """
        Removes references from all PDFs in the specified input data path and saves them to the output data path.
        """
        for root, _, files in os.walk(
            self.input_data_path
        ):  # Recursively traverse directories
            for file in files:
                if file.endswith(".pdf"):
                    input_file_path = os.path.join(root, file)

                    # Create corresponding output path
                    relative_path = os.path.relpath(
                        input_file_path, self.input_data_path
                    )
                    output_file_path = os.path.join(
                        self.output_data_path, relative_path
                    )

                    # Remove references and save to the output path
                    self.remove_references(
                        input_file_path=input_file_path,
                        output_file_path=output_file_path,
                    )

    def load_documents(self, enrich_metadata=False):
        """
        Loads processed PDF documents from the output data path.
        """
        docs = []
        for root, _, files in os.walk(
            self.output_data_path
        ):  # Recursively traverse directories
            for file in files:
                if file.endswith(".pdf"):
                    file_path = os.path.join(root, file)
                    loader = PyMuPDFLoader(file_path)
                    loaded_docs = loader.load()

                    # Enrich metadata if the flag is set
                    if enrich_metadata:
                        for doc in loaded_docs:
                            doc.metadata["name"] = os.path.splitext(file)[
                                0
                            ]  # Get file name without extension
                            doc.metadata["year"] = 2024  # Set year as 2024 for now

                    docs.extend(loaded_docs)

        print(
            f"Loaded {len(docs)} documents from {self.output_data_path} and its subdirectories."
        )
        return docs

    def split_documents(self, docs, chunk_size=1500, chunk_overlap=100):
        """
        Splits documents into chunks for vectorization.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        return splitter.split_documents(docs)

    def create_vector_db(self, docs):
        """
        Creates and stores the vector database in FAISS.
        """
        if not os.path.exists(self.vectordb_path):
            os.makedirs(self.vectordb_path)

        self.vector_store = FAISS.from_documents(
            documents=docs,
            embedding=self.embedding,
        )

        # Save the FAISS vector store to the specified directory
        self.vector_store.save_local(folder_path=self.vectordb_path)
        print(f"Knowledge base created and saved in directory: {self.vectordb_path}")

    def process_and_create_db(self):
        """Main method to remove references, load, split, and create vectorDB."""
        self.clean_documents()  # Remove references first
        docs = self.load_documents()
        splits = self.split_documents(docs)
        self.create_vector_db(splits)


# import os
# from langchain_community.document_loaders import PyMuPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# import fitz  # PyMuPDF

# class DataProcessor:
#     """
#     Handles loading, processing, and creating vector databases for documents.
#     """

#     def __init__(self, vectordb_path='vectordb', input_data_path="data/raw", output_data_path="data/processed", embedding_model="Alibaba-NLP/gte-large-en-v1.5"):
#         """
#         Initialize the DataProcessor with default values.

#         Parameters:

#         - data_path: str, path to the directory containing raw PDF files. Default is "data/raw".

#         - vectordb_path: str, path to the directory where the vector database will be stored. Default is "vectordb".

#         - embedding_model: str, the embedding model to be used for vectorization. Default is "Alibaba-NLP/gte-large-en-v1.5".

#         Note:
#         These are the default values. We suggest models from the MTEB leaderboard
#         (https://huggingface.co/spaces/mteb/leaderboard) based on the `Retrieval Average` score
#         and `Memory Usage`. Balancing retrieval quality and available resources is recommended
#         to optimize both accuracy and efficiency in your specific environment.
#         """
#         self.input_data_path = input_data_path
#         self.output_data_path = output_data_path
#         self.vectordb_path = vectordb_path
#         self.embedding = HuggingFaceEmbeddings(model_name=embedding_model, model_kwargs={"trust_remote_code": True})
#         self.vector_store = None

#     def remove_references(self, input_file_path, output_file_path):
#         """
#         Remove references section from a PDF file and overwrite it.

#         Parameters:
#         - file_path (str): Path to the PDF file to process.
#         """
#         # input_file_path = self.input_data_path
#         # output_file_path = self.output_data_path
#         doc = fitz.open(input_file_path)
#         references_found = False

#         for page_num in range(doc.page_count):
#             page = doc.load_page(page_num)
#             if "References" in page.get_text() and not references_found:
#                 references_found = True
#                 # Redact text starting from the "References" heading to the bottom of the page
#                 text_instances = page.search_for("References")
#                 for inst in text_instances:
#                     redact_area = fitz.Rect(inst.x0, inst.y0, page.rect.width, page.rect.height)
#                     page.add_redact_annot(redact_area)
#                 page.apply_redactions()
#             elif references_found:
#                 # Redact entire pages after the references start
#                 page.add_redact_annot(page.rect)
#                 page.apply_redactions()

#         # Save changes, overwriting the file
#         doc.save(output_file_path)
#         doc.close()
#         print(f"References removed from: {input_file_path}")

#     def clean_documents(self):
#         """
#         Removes references from all PDFs in the specified data path and its subdirectories.
#         """
#         for root, _, files in os.walk(self.input_data_path):  # Recursively traverse directories
#             for file in files:
#                 if file.endswith(".pdf"):
#                     file_path = os.path.join(root, file)
#                     self.remove_references(input_file_path=self.input_data_path, output_file_path=self.output_data_path)

#     def load_documents(self, enrich_metadata=False):
#         """
#         Loads PDF documents from the specified data path and its subdirectories, optionally enriching metadata.

#         Parameters:

#         - enrich_metadata (bool): If True, add metadata to each document (e.g., name and year).

#         Returns:

#         - list: List of loaded documents with optional metadata.
#         """
#         docs = []
#         for root, _, files in os.walk(self.output_data_path):  # Recursively traverse directories
#             for file in files:
#                 if file.endswith(".pdf"):
#                     file_path = os.path.join(root, file)
#                     loader = PyMuPDFLoader(file_path)
#                     loaded_docs = loader.load()

#                     # Enrich metadata if the flag is set
#                     if enrich_metadata:
#                         for doc in loaded_docs:
#                             doc.metadata["name"] = os.path.splitext(file)[0]  # Get file name without extension
#                             doc.metadata["year"] = 2024  # Set year as 2024 for now

#                     docs.extend(loaded_docs)

#         print(f"Loaded {len(docs)} documents from {self.output_data_path} and its subdirectories.")
#         return docs

#     def split_documents(self, docs, chunk_size=1500, chunk_overlap=100):
#         """
#         Splits documents into chunks for vectorization.

#         Parameters:

#         - docs: list, documents to split.

#         - chunk_size: int, size of each chunk. Default is 1500.

#         - chunk_overlap: int, overlap between chunks. Default is 100.

#         Returns:

#         - list: List of document chunks.
#         """
#         splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#         return splitter.split_documents(docs)

#     def create_vector_db(self, docs):
#         """
#         Creates and stores the vector database in FAISS.

#         Parameters:

#         - docs: list, document chunks to vectorize and store.
#         """
#         if not os.path.exists(self.vectordb_path):
#             os.makedirs(self.vectordb_path)

#         self.vector_store = FAISS.from_documents(
#             documents=docs,
#             embedding=self.embedding,
#         )

#         # Save the FAISS vector store to the specified directory
#         self.vector_store.save_local(folder_path=self.vectordb_path)
#         print(f"Knowledge base created and saved in directory: {self.vectordb_path}")

#     def process_and_create_db(self):
#         """Main method to remove references, load, split, and create vectorDB."""
#         self.clean_documents()  # Remove references first
#         docs = self.load_documents()
#         splits = self.split_documents(docs)
#         self.create_vector_db(splits)


# # src/rag_skeleton/data_processing.py
# import os
# from langchain_community.document_loaders import PyMuPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS

# class DataProcessor:
#     """
#     Handles loading, processing, and creating vector databases for documents.
#     """

#     def __init__(self, vectordb_path='vectordb', data_path="data/raw", embedding_model="Alibaba-NLP/gte-large-en-v1.5"):
#         """
#         Initialize the DataProcessor with default values.

#         Parameters:

#         - data_path: str, path to the directory containing raw PDF files. Default is "data/raw".

#         - vectordb_path: str, path to the directory where the vector database will be stored. Default is "vectordb".

#         - embedding_model: str, the embedding model to be used for vectorization. Default is "Alibaba-NLP/gte-large-en-v1.5".

#         Note:
#         These are the default values. We suggest models from the MTEB leaderboard
#         (https://huggingface.co/spaces/mteb/leaderboard) based on the `Retrieval Average` score
#         and `Memory Usage`. Balancing retrieval quality and available resources is recommended
#         to optimize both accuracy and efficiency in your specific environment.
#         """
#         self.data_path = data_path
#         self.vectordb_path = vectordb_path
#         self.embedding = HuggingFaceEmbeddings(model_name=embedding_model, model_kwargs={"trust_remote_code":True})   # https://github.com/langchain-ai/langchain/issues/6080#issuecomment-1963311548
#         self.vector_store = None

#     def load_documents(self, enrich_metadata=False):
#         """
#         Loads PDF documents from the specified data path and its subdirectories, optionally enriching metadata.

#         Parameters:

#         - enrich_metadata (bool): If True, add metadata to each document (e.g., name and year).

#         Returns:

#         - list: List of loaded documents with optional metadata.
#         """
#         docs = []
#         for root, _, files in os.walk(self.data_path):  # Recursively traverse directories
#             for file in files:
#                 if file.endswith(".pdf"):
#                     file_path = os.path.join(root, file)
#                     loader = PyMuPDFLoader(file_path)
#                     loaded_docs = loader.load()

#                     # Enrich metadata if the flag is set
#                     if enrich_metadata:
#                         for doc in loaded_docs:
#                             doc.metadata["name"] = os.path.splitext(file)[0]  # Get file name without extension
#                             doc.metadata["year"] = 2024  # Set year as 2024 for now

#                     docs.extend(loaded_docs)

#         print(f"Loaded {len(docs)} documents from {self.data_path} and its subdirectories.")
#         return docs

#     def split_documents(self, docs, chunk_size=1500, chunk_overlap=100):
#         """
#         Splits documents into chunks for vectorization.

#         Parameters:

#         - docs: list, documents to split.

#         - chunk_size: int, size of each chunk. Default is 1500.

#         - chunk_overlap: int, overlap between chunks. Default is 100.

#         Returns:

#         - list: List of document chunks.

#         """
#         splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#         return splitter.split_documents(docs)

#     def create_vector_db(self, docs):
#         """
#         Creates and stores the vector database in ChromaDB.

#         Parameters:

#         - docs: list, document chunks to vectorize and store.

#         """
#         if not os.path.exists(self.vectordb_path):
#             os.makedirs(self.vectordb_path)

#         self.vector_store = FAISS.from_documents(
#             documents=docs,
#             embedding=self.embedding,
#             # persist_directory=self.vectordb_path
#         )

#         # Save the FAISS vector store to the specified directory
#         self.vector_store.save_local(folder_path=self.vectordb_path)


#         print(f"Knowledge base created and saved in directory: {self.vectordb_path}")

#     def process_and_create_db(self):
#         """Main method to load, split, and create vectorDB."""
#         docs = self.load_documents()
#         splits = self.split_documents(docs)
#         self.create_vector_db(splits)
